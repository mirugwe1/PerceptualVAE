"""
latent_diagnostics.py

Diagnostics for the latent space and the decoder, run on a trained checkpoint.
Produces three figures and a JSON of summary statistics.

    python latent_diagnostics.py --config config.yaml --checkpoint runs/.../best.pt

WHY THESE PARTICULAR DIAGNOSTICS

The detector's entire mechanism is the information bottleneck: the encoder must
compress an image into a latent far smaller than the input, so the decoder can
only rebuild it from what the compression preserved, and the compression was
learned from healthy anatomy alone. Everything that can go wrong with the
method goes wrong in the latent or in the decoder's expressive range, and a
single scalar such as mean KL per dimension hides most of it.

  FIGURE A, LATENT STATISTICS, answers: is the bottleneck being enforced?
    Per-dimension KL says WHICH dimensions carry information rather than how
    much on average. A handful of dimensions carrying everything while the
    rest sit at the free-bits floor is a very different failure from uniform
    over-capacity, and the mean cannot tell them apart.
    The latent norm is compared against the chi distribution the prior
    implies, so posterior-prior mismatch is visible directly rather than
    inferred from a KL value.
    The covariance matrix shows whether dimensions are decorrelated as the
    factorised prior assumes. Strong off-diagonal structure means the latent
    is using fewer effective dimensions than it appears to.
    Cosine similarity between latents of different images is compared against
    the 1/sqrt(d) spread expected for independent draws. Values far above that
    mean all images map to a similar direction, which is prototype collapse
    measured in the latent rather than guessed at from blurry outputs.

  FIGURE B, LATENT GEOMETRY, answers: what can the decoder actually produce?
    Linear and spherical interpolation between two encoded images show whether
    the decoder produces plausible radiographs throughout the path or only at
    the endpoints. Spherical interpolation is included because linear paths
    pass near the origin, where a Gaussian prior has almost no mass, so a
    linear path can look bad for reasons that say nothing about the model.
    Single-dimension traversals decode z = a * e_i for a range of a, showing
    what each latent dimension controls. Dimensions are ordered by their KL
    contribution, so the ones actually in use appear first.

  FIGURE C, DECODER SPECTRUM, answers: can the decoder represent fine detail
  at all? The radially averaged power spectrum of what the decoder produces is
  compared against the spectrum of real radiographs. If the decoder's output
  carries no energy above some spatial frequency, then no residual computed
  from it can localise anything smaller than that, whatever the loss. This
  turns "the reconstructions look like prototypes" into a measurement.

PSNR is derivable from existing logs rather than requiring new instrumentation:
with a per-pixel mean squared error on data in [0, 1], PSNR = -10 log10(recon).
plot_logs.py computes it that way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402
import torch.nn.functional as F          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (                      # noqa: E402
    build_datamodule, get, get_device, load_checkpoint, load_config, set_seed,
)
try:                                     # the module is vae_model.py where
    from vae_model import VAE            # torchxrayvision forced a rename
except ImportError:                      # noqa: F401
    from model import VAE                # noqa: E402


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #

@torch.no_grad()
def collect_latents(model, loader, device, max_images=2000):
    """Encodes images and returns mu, sigma, a sampled z, and the inputs."""
    mus, sigmas, zs, xs = [], [], [], []
    n = 0
    model.eval()
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        mu, logvar = model.encoder(x)
        mus.append(mu.cpu())
        sigmas.append(torch.exp(0.5 * logvar).cpu())
        zs.append(model.reparameterize(mu, logvar).cpu())
        xs.append(x.cpu())
        n += x.size(0)
        if n >= max_images:
            break
    return (torch.cat(mus)[:max_images], torch.cat(sigmas)[:max_images],
            torch.cat(zs)[:max_images], torch.cat(xs)[:max_images])


def per_dim_kl(mu: torch.Tensor, sigma: torch.Tensor) -> np.ndarray:
    """KL(q||N(0,1)) per latent dimension, averaged over images, in nats."""
    logvar = 2 * torch.log(sigma.clamp_min(1e-8))
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return kl.mean(dim=0).numpy()


def radial_spectrum(imgs: np.ndarray, n_bins: int = 48):
    """Radially averaged power spectrum of a batch of (N, H, W) images."""
    F2 = np.abs(np.fft.fftshift(np.fft.fft2(imgs, axes=(-2, -1)), axes=(-2, -1))) ** 2
    F2 = F2.mean(axis=0)
    h, w = F2.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(cy, cx)
    edges = np.linspace(0, rmax, n_bins + 1)
    prof, freq = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.sum():
            prof.append(F2[m].mean())
            freq.append((a + b) / 2 / rmax)          # 0 to 1, Nyquist at 1
    return np.array(freq), np.array(prof)


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation. Linear paths cross the origin, where a
    Gaussian prior in high dimensions has almost no mass, so a linear path can
    look poor for reasons unrelated to the model."""
    an, bn = a / a.norm(), b / b.norm()
    omega = torch.acos((an * bn).sum().clamp(-1 + 1e-7, 1 - 1e-7))
    so = torch.sin(omega)
    if so.abs() < 1e-6:
        return (1 - t) * a + t * b
    return (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b


# --------------------------------------------------------------------------- #
# Figure A: latent statistics
# --------------------------------------------------------------------------- #

def figure_latent_stats(mu, sigma, z, free_bits, out_path, title):
    d = mu.shape[1]
    kl = per_dim_kl(mu, sigma)
    order = np.argsort(-kl)

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.4))

    # 1. per-dimension KL, sorted
    ax[0, 0].bar(range(d), kl[order], width=1.0, color="#2b6cb0")
    ax[0, 0].axhline(free_bits, color="crimson", ls="--", lw=1.2)
    ax[0, 0].text(d * 0.98, free_bits * 1.08, f"free-bits floor = {free_bits}",
                  color="crimson", fontsize=8, ha="right")
    alive = int((kl > free_bits * 1.05).sum())
    ax[0, 0].set_title(f"A1. KL per dimension, sorted\n{alive} of {d} above the floor "
                       f"({alive/d:.0%}); total {kl.sum():.1f} nats", fontsize=9.5)
    ax[0, 0].set_xlabel("latent dimension (sorted)"); ax[0, 0].set_ylabel("nats")
    ax[0, 0].grid(axis="y", alpha=.25)

    # 2. mu and sigma spread per dimension
    mus, sig = mu.numpy()[:, order], sigma.numpy()[:, order]
    q = lambda a, p: np.percentile(a, p, axis=0)
    ax[0, 1].fill_between(range(d), q(mus, 5), q(mus, 95), alpha=.35, color="#2b6cb0", label="mu, 5-95%")
    ax[0, 1].plot(range(d), mus.mean(0), color="#2b6cb0", lw=1)
    ax[0, 1].fill_between(range(d), q(sig, 5), q(sig, 95), alpha=.35, color="#c05621", label="sigma, 5-95%")
    ax[0, 1].plot(range(d), sig.mean(0), color="#c05621", lw=1)
    ax[0, 1].axhline(1.0, color="0.4", ls=":", lw=1)
    ax[0, 1].set_title("A2. Posterior mean and standard deviation\n"
                       "sigma near 1 means that dimension is unused", fontsize=9.5)
    ax[0, 1].set_xlabel("latent dimension (sorted by KL)")
    ax[0, 1].legend(fontsize=7.5); ax[0, 1].grid(alpha=.25)

    # 3. latent norm against the chi reference implied by the prior
    norms = z.norm(dim=1).numpy()
    ax[0, 2].hist(norms, bins=60, density=True, color="#2f855a", alpha=.75, label="observed")
    ref = np.sqrt(np.random.default_rng(0).chisquare(d, 20000))
    ax[0, 2].hist(ref, bins=60, density=True, histtype="step", color="crimson",
                  lw=1.4, label=f"prior, chi_{d}")
    ax[0, 2].set_title(f"A3. Latent norm\nobserved mean {norms.mean():.2f}, "
                       f"prior mean {np.sqrt(d):.2f}", fontsize=9.5)
    ax[0, 2].set_xlabel("||z||"); ax[0, 2].legend(fontsize=7.5); ax[0, 2].grid(alpha=.25)

    # 4. cosine similarity between latents of different images
    zz = z / z.norm(dim=1, keepdim=True)
    idx = np.random.default_rng(0).choice(len(zz), size=(2, 4000))
    cos = (zz[idx[0]] * zz[idx[1]]).sum(1).numpy()
    ax[1, 0].hist(cos, bins=60, density=True, color="#805ad5", alpha=.75, label="observed")
    xs = np.linspace(-1, 1, 400)
    ax[1, 0].plot(xs, np.exp(-xs**2 * d / 2) * np.sqrt(d / (2 * np.pi)),
                  color="crimson", lw=1.4, label=f"independent draws, sd = 1/sqrt({d})")
    ax[1, 0].set_title(f"A4. Cosine between latents of different images\n"
                       f"observed sd {cos.std():.3f}, independent {1/np.sqrt(d):.3f}", fontsize=9.5)
    ax[1, 0].set_xlabel("cos(z_i, z_j)"); ax[1, 0].set_xlim(-1, 1)
    ax[1, 0].legend(fontsize=7.5); ax[1, 0].grid(alpha=.25)

    # 5. covariance of z
    zc = (z - z.mean(0)).numpy()
    cov = (zc.T @ zc) / (len(zc) - 1)
    cov = cov[np.ix_(order, order)]
    v = np.percentile(np.abs(cov), 99)
    im = ax[1, 1].imshow(cov, cmap="RdBu_r", vmin=-v, vmax=v)
    off = cov[~np.eye(d, dtype=bool)]
    ax[1, 1].set_title("A5. Covariance of z (sorted by KL)\n"
                       f"mean |off-diagonal| {np.abs(off).mean():.4f}, "
                       f"mean diagonal {np.diag(cov).mean():.3f}", fontsize=9.5)
    plt.colorbar(im, ax=ax[1, 1], fraction=.046)

    # 6. Mahalanobis distance of mu under the empirical covariance of z
    try:
        P = np.linalg.pinv(cov)
        mc = (mu.numpy() - mu.numpy().mean(0))[:, order]
        maha = np.sqrt(np.einsum("ni,ij,nj->n", mc, P, mc))
        ax[1, 2].hist(maha, bins=60, density=True, color="#b7791f", alpha=.75, label="mu")
        ax[1, 2].hist(ref, bins=60, density=True, histtype="step", color="crimson",
                      lw=1.4, label=f"chi_{d}")
        ax[1, 2].set_title(f"A6. Mahalanobis distance of mu\nmean {maha.mean():.2f} "
                           f"against {np.sqrt(d):.2f} expected", fontsize=9.5)
        ax[1, 2].set_xlabel("distance"); ax[1, 2].legend(fontsize=7.5)
    except np.linalg.LinAlgError:
        ax[1, 2].text(.5, .5, "covariance not invertible", ha="center", va="center")
    ax[1, 2].grid(alpha=.25)

    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=165, bbox_inches="tight")
    plt.close(fig)

    return {"kl_per_dim_total": float(kl.sum()),
            "kl_per_dim_mean": float(kl.mean()),
            "dims_above_floor": alive, "dims_total": int(d),
            "latent_norm_mean": float(norms.mean()),
            "latent_norm_prior": float(np.sqrt(d)),
            "cosine_sd_observed": float(cos.std()),
            "cosine_sd_independent": float(1 / np.sqrt(d)),
            "mean_abs_offdiag_cov": float(np.abs(off).mean()),
            "mean_diag_cov": float(np.diag(cov).mean())}



# --------------------------------------------------------------------------- #
# Figure D: aggregate posterior and per-dimension boxes
# --------------------------------------------------------------------------- #

def figure_latent_box(mu, sigma, z, free_bits, out_path, title, max_dims=64):
    """The two standard VAE latent checks, side by side with the per-dimension view.

    D1, THE AGGREGATE POSTERIOR. Pools every latent value over all dimensions
    and all images and compares the resulting density with the N(0,1) prior.
    This is the question "does q(z) match p(z) overall?", and it is the check
    most often shown for a VAE. It is necessary but not sufficient: a latent
    in which half the dimensions are dead at N(0,1) and the other half carry
    everything can still pool to something close to N(0,1), which is why the
    per-dimension panels beside it matter.

    D2 and D3, THE PER-DIMENSION BOXES. One box per latent dimension showing
    the spread of the posterior mean and of the posterior standard deviation
    across images. A dimension that is being used has mu varying widely across
    images and sigma pushed below 1; a dead dimension has mu pinned near 0 and
    sigma near 1, because the cheapest way to pay no KL is to return the prior.
    Reading D1 and D2 together is what separates "the latent matches the prior"
    from "the latent is half unused".

    D4, KL PER DIMENSION, on the same ordering, so the boxes can be read
    against how much each dimension actually contributes.
    """
    d = mu.shape[1]
    kl = per_dim_kl(mu, sigma)
    order = np.argsort(-kl)
    k = min(max_dims, d)
    sel = order[:k]

    mus = mu.numpy()[:, sel]
    sig = sigma.numpy()[:, sel]
    pooled = z.numpy().ravel()

    fig = plt.figure(figsize=(16.5, 9.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 1], width_ratios=[1, 1.9],
                          hspace=.42, wspace=.18)

    # ---- D1 aggregate posterior against the prior ----
    ax = fig.add_subplot(gs[0, 0])
    lo, hi = -6, 6
    xs = np.linspace(lo, hi, 600)
    ax.hist(pooled, bins=240, range=(lo, hi), density=True, color="#2b6cb0",
            alpha=.55, label="empirical q(z)")
    ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color="crimson",
            ls="--", lw=1.6, label="prior N(0,1)")
    ax.set_xlim(lo, hi); ax.set_xlabel("latent value (pooled over dimensions and images)")
    ax.set_ylabel("density"); ax.legend(fontsize=8); ax.grid(alpha=.25)
    ax.set_title(f"D1. Aggregate posterior against the prior\n"
                 f"empirical mean {pooled.mean():.3f}, sd {pooled.std():.3f}; "
                 f"prior 0 and 1", fontsize=9.5)

    # ---- D1b the same per dimension, as a heat map ----
    ax = fig.add_subplot(gs[0, 1])
    edges = np.linspace(lo, hi, 121)
    H = np.stack([np.histogram(z.numpy()[:, i], bins=edges, density=True)[0] for i in sel])
    im = ax.imshow(H, aspect="auto", origin="lower", cmap="magma",
                   extent=[lo, hi, 0, k])

    # Dimension runs up the y axis, so a curve of mu against x traces a
    # roughly vertical line, and the posterior standard deviation becomes a
    # band either side of it. Reading them against the prior reference at 0
    # and +/- 1 shows, per dimension, both where the posterior sits and how
    # wide it is, without leaving the density plot.
    yy = np.arange(k) + 0.5
    mu_d = mus.mean(axis=0)
    sd_d = sig.mean(axis=0)
    ax.axvline(0, color="white", ls=":", lw=1.1)
    ax.axvline(-1, color="#7fd4ff", ls=":", lw=.9)
    ax.axvline(1, color="#7fd4ff", ls=":", lw=.9)
    ax.fill_betweenx(yy, mu_d - sd_d, mu_d + sd_d, color="white", alpha=.20,
                     label="mu +/- sigma")
    ax.plot(mu_d, yy, color="white", lw=1.4, label="mu per dimension")
    ax.plot(mu_d - sd_d, yy, color="white", lw=.7, alpha=.8)
    ax.plot(mu_d + sd_d, yy, color="white", lw=.7, alpha=.8)
    ax.set_xlim(lo, hi); ax.set_ylim(0, k)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=.35,
              labelcolor="white", facecolor="black", edgecolor="none")
    ax.set_xlabel("latent value"); ax.set_ylabel("dimension (sorted by KL)")
    ax.set_title("D1b. Per-dimension density of z, with mu and sigma overlaid\n"
                 "a dead dimension sits on 0 with the band reaching the dotted "
                 "lines at +/- 1", fontsize=9.5)
    plt.colorbar(im, ax=ax, fraction=.03)

    # ---- D2 boxes of mu ----
    ax = fig.add_subplot(gs[1, :])
    bp = ax.boxplot([mus[:, i] for i in range(k)], showfliers=False, widths=.62,
                    patch_artist=True, medianprops=dict(color="black", lw=.9))
    for b in bp["boxes"]:
        b.set(facecolor="#2b6cb0", alpha=.65, linewidth=.6)
    ax.axhline(0, color="crimson", ls="--", lw=1)
    ax.set_ylabel("posterior mean, mu")
    ax.set_xticks(range(1, k + 1, max(1, k // 16)))
    ax.set_xticklabels([str(sel[i]) for i in range(0, k, max(1, k // 16))], fontsize=7)
    ax.set_title(f"D2. Posterior mean per dimension, {k} most informative of {d}. "
                 f"A used dimension has mu spread across images; a dead one sits on the "
                 f"red line", fontsize=9.5)
    ax.grid(axis="y", alpha=.25)

    # ---- D3 boxes of sigma, with KL overlaid ----
    ax = fig.add_subplot(gs[2, :])
    bp = ax.boxplot([sig[:, i] for i in range(k)], showfliers=False, widths=.62,
                    patch_artist=True, medianprops=dict(color="black", lw=.9))
    for b in bp["boxes"]:
        b.set(facecolor="#c05621", alpha=.65, linewidth=.6)
    ax.axhline(1.0, color="crimson", ls="--", lw=1)
    ax.text(k * .995, 1.02, "sigma = 1 means the dimension returns the prior",
            color="crimson", fontsize=7.5, ha="right")
    ax.set_ylabel("posterior sd, sigma"); ax.set_xlabel("latent dimension (sorted by KL)")
    ax.set_xticks(range(1, k + 1, max(1, k // 16)))
    ax.set_xticklabels([str(sel[i]) for i in range(0, k, max(1, k // 16))], fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(range(1, k + 1), kl[sel], color="#2f855a", lw=1.4)
    ax2.axhline(free_bits, color="#2f855a", ls=":", lw=1)
    ax2.set_ylabel("KL per dimension, nats (green)", fontsize=8.5, color="#2f855a")
    ax.set_title("D3. Posterior standard deviation per dimension, with the KL "
                 "contribution overlaid", fontsize=9.5)
    ax.grid(axis="y", alpha=.25)

    fig.suptitle(title, fontsize=12, y=.985)
    fig.savefig(out_path, dpi=165, bbox_inches="tight")
    plt.close(fig)

    dead = int((kl <= free_bits * 1.05).sum())
    return {"pooled_mean": float(pooled.mean()), "pooled_sd": float(pooled.std()),
            "dead_dims": dead,
            "mu_sd_across_images_mean": float(mu.numpy().std(axis=0).mean()),
            "sigma_median": float(np.median(sigma.numpy()))}


# --------------------------------------------------------------------------- #
# Figure B: latent geometry
# --------------------------------------------------------------------------- #

@torch.no_grad()
def figure_latent_geometry(model, mu, sigma, device, out_path, title, n_steps=9, n_dims=6):
    kl = per_dim_kl(mu, sigma)
    order = np.argsort(-kl)
    za, zb = mu[0].to(device), mu[1].to(device)
    ts = np.linspace(0, 1, n_steps)

    lin = torch.stack([(1 - t) * za + t * zb for t in ts])
    sph = torch.stack([slerp(za, zb, float(t)) for t in ts])
    rows = [("linear interpolation", model.decoder(lin)[0].cpu().numpy()[:, 0]),
            ("spherical interpolation", model.decoder(sph)[0].cpu().numpy()[:, 0])]

    # single-dimension traversals, most informative dimensions first
    amps = np.linspace(-3, 3, n_steps)
    base = mu.mean(0).to(device)
    for k in range(n_dims):
        i = int(order[k])
        zt = torch.stack([base.clone() for _ in amps])
        for j, a in enumerate(amps):
            zt[j, i] = float(a)
        rows.append((f"dim {i} (KL {kl[i]:.2f})", model.decoder(zt)[0].cpu().numpy()[:, 0]))

    fig, ax = plt.subplots(len(rows), n_steps, figsize=(1.45 * n_steps, 1.5 * len(rows)),
                           squeeze=False)
    for r, (lbl, imgs) in enumerate(rows):
        for c in range(n_steps):
            a = ax[r][c]
            a.imshow(imgs[c], cmap="gray", vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
        ax[r][0].set_ylabel(lbl, fontsize=8, labelpad=6)
    for c, t in enumerate(ts):
        ax[0][c].set_title(f"{t:.2f}", fontsize=7.5)
    fig.suptitle(title + "\nrows 1 and 2 interpolate between two encoded images; "
                 "the rest vary one latent dimension from -3 to +3 with the others held at the mean",
                 fontsize=10, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=165, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure C: decoder basis and spectrum
# --------------------------------------------------------------------------- #

@torch.no_grad()
def figure_decoder_spectrum(model, mu, sigma, x_real, device, out_path, title, n_dims=8):
    kl = per_dim_kl(mu, sigma)
    order = np.argsort(-kl)
    d = mu.shape[1]

    # decoder response to each active dimension, as a difference from the mean image
    base = mu.mean(0).to(device)
    base_img = model.decoder(base[None])[0].cpu().numpy()[0, 0]
    basis, labels = [], []
    for k in range(n_dims):
        i = int(order[k])
        z = base.clone()[None].repeat(1, 1)
        z[0, i] += 3.0
        basis.append(model.decoder(z)[0].cpu().numpy()[0, 0] - base_img)
        labels.append(f"dim {i}")

    # reconstructions and real images, for the spectrum comparison
    recon, _ = model.reconstruct(x_real[:256].to(device), n_samples=1)
    recon = recon.cpu().numpy()[:, 0]
    real = x_real[:256].numpy()[:, 0]

    fig = plt.figure(figsize=(16.5, 7.4))
    gs = fig.add_gridspec(2, n_dims, height_ratios=[1, 1.35], hspace=.32)

    for k in range(n_dims):
        a = fig.add_subplot(gs[0, k])
        v = np.abs(basis[k]).max()
        a.imshow(basis[k], cmap="RdBu_r", vmin=-v, vmax=v)
        a.set_title(labels[k], fontsize=8)
        a.set_xticks([]); a.set_yticks([])
    fig.text(0.008, 0.80, "C1. Decoder basis\n(response to +3 in one\ndimension, others at mean)",
             fontsize=9, va="center")

    a = fig.add_subplot(gs[1, :4])
    fr, pr_real = radial_spectrum(real)
    _, pr_rec = radial_spectrum(recon)
    a.semilogy(fr, pr_real, color="#2b6cb0", lw=1.8, label="real radiographs")
    a.semilogy(fr, pr_rec, color="#c05621", lw=1.8, label="reconstructions")
    a.set_xlabel("spatial frequency (1 = Nyquist)"); a.set_ylabel("power")
    a.set_title("C2. Radially averaged power spectrum\nthe gap at high frequency is the "
                "detail the decoder cannot produce", fontsize=9.5)
    a.legend(fontsize=8); a.grid(alpha=.25)

    a = fig.add_subplot(gs[1, 4:])
    ratio = pr_rec / np.maximum(pr_real, 1e-12)
    a.plot(fr, ratio, color="#2f855a", lw=1.8)
    a.axhline(1.0, color="0.4", ls=":", lw=1)
    half = fr[np.argmin(np.abs(ratio - 0.5))] if (ratio < 0.5).any() else np.nan
    if not np.isnan(half):
        a.axvline(half, color="crimson", ls="--", lw=1.2)
        a.text(half, 0.55, f"  half power at {half:.2f} Nyquist", color="crimson", fontsize=8)
    a.set_xlabel("spatial frequency (1 = Nyquist)"); a.set_ylabel("reconstruction / real")
    a.set_ylim(0, 1.25)
    a.set_title("C3. Retained power by frequency\nbelow 0.5 means more than half the "
                "energy at that scale is lost", fontsize=9.5)
    a.grid(alpha=.25)

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.savefig(out_path, dpi=165, bbox_inches="tight")
    plt.close(fig)
    return {"half_power_frequency": float(half) if not np.isnan(half) else None}


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Latent and decoder diagnostics")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-images", type=int, default=2000)
    ap.add_argument("--out", default=None, help="Defaults to the run directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"].get("device", "cuda"))

    model = VAE(cfg).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, device)["model_state"])
    model.eval()

    dm = build_datamodule(cfg)
    _, _, normal_test = dm.normal_loaders()
    mu, sigma, z, x = collect_latents(model, normal_test, device, args.max_images)
    print(f"collected {len(mu)} latents of dimension {mu.shape[1]}")

    out = Path(args.out) if args.out else Path(args.checkpoint).parent.parent
    name = cfg["experiment"]["name"]
    free_bits = float(get(cfg, "loss.free_bits", 0.5))

    stats = figure_latent_stats(mu, sigma, z, free_bits,
                                out / "latent_stats.png", f"Latent statistics: {name}")
    stats.update(figure_latent_box(mu, sigma, z, free_bits,
                 out / "latent_box.png", f"Aggregate posterior and per-dimension boxes: {name}"))
    figure_latent_geometry(model, mu, sigma, device,
                           out / "latent_geometry.png", f"Latent geometry: {name}")
    stats.update(figure_decoder_spectrum(model, mu, sigma, x, device,
                 out / "decoder_spectrum.png", f"Decoder basis and spectrum: {name}"))

    with open(out / "latent_diagnostics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\nsummary")
    for k, v in stats.items():
        print(f"  {k:26s} {v}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()