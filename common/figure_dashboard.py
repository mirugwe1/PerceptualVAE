"""
figure_dashboard.py

A single composite diagnostic page for a trained VAE, in the layout used by
the group: image rows, per-latent-variable distributions, latent geometry,
and the loss and PSNR history.

    python ../common/figure_dashboard.py --config config.yaml   --checkpoint runs/vae_vgg19_imagenet/20260929_091115/checkpoints/best.pt

    # with reference curves from other runs on the loss panel
    python ../common/figure_dashboard.py --config config.yaml \
        --checkpoint .../best.pt \
        --reference ../stage2_vgg19/runs/*/*/train.log

LAYOUT

  BLOCK 1, six image rows
      data x            held-out test images
      recon x_hat_z     decoded from a SAMPLED latent
      recon x_hat_mu    decoded from the posterior mean
      x_train           training images
      recon x_train     their reconstructions
      gen x_hat_post    decoded from prior draws, z ~ N(0, I)

    The two reconstruction rows are shown separately because they answer
    different questions. Decoding from mu is what the anomaly score uses and
    shows the model at its best. Decoding from a sampled z shows how much the
    posterior width costs, which is the part of the model the KL term
    controls. The prior row is the honest test of whether the latent has
    learned a distribution at all rather than a lookup of the training set.

  BLOCK 2, per-latent-variable distributions of mu, sigma and z
    One row per latent dimension, colour showing the density of values that
    dimension takes across images. A dimension carrying information shows a
    broad band; one returning the prior shows a band identical to N(0,1) in mu
    and z, and sitting at 1 in sigma. Overlaid dots give each dimension's
    mean and standard deviation, defined in the panel titles. The summary
    figures at the top of each panel report the overall mean and standard
    deviation, which is where the comparison against the prior is read.

  BLOCK 3, latent geometry
    Norms of z and mu divided by sqrt(latent dimension), so the prior sits at
    1 regardless of size; cosine similarity between latents of different
    images against the prior expectation; and the covariance of z, where
    off-diagonal structure means the dimensions are not independent as the
    factorised prior assumes.

  BLOCK 4, loss and PSNR
    Training and test loss on a logarithmic left axis, PSNR on the right.
    PSNR is derived rather than logged: with a per-pixel mean squared error on
    data in [0, 1], PSNR = -10 log10(recon).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402


# --------------------------------------------------------------------------- #
# Figure construction. Pure numpy, so it can be exercised without a GPU.
# --------------------------------------------------------------------------- #

ROW_LABELS = [r"data $x$", r"recon $\hat{x}_z$", r"recon $\hat{x}_\mu$",
              r"$x_{train}$", r"recon $x_{train}$", r"gen $\hat{x}_{post}$"]


def montage(rows: list[np.ndarray], pad: int = 2) -> np.ndarray:
    """Stacks equal-length lists of images into one padded array."""
    n, h, w = rows[0].shape
    R, C = len(rows), n
    out = np.ones((R * (h + pad) + pad, C * (w + pad) + pad))
    for r, imgs in enumerate(rows):
        for c in range(C):
            y = pad + r * (h + pad)
            x = pad + c * (w + pad)
            out[y:y + h, x:x + w] = np.clip(imgs[c], 0, 1)
    return out


def density_panel(ax, values, lo, hi, title, n_bins=160, cmap="rainbow",
                  dots="mean_sd"):
    """One row per latent dimension, colour showing the density of that
    dimension's values across images."""
    d = values.shape[1]
    edges = np.linspace(lo, hi, n_bins + 1)
    H = np.stack([np.histogram(values[:, i], bins=edges, density=True)[0]
                  for i in range(d)])
    # A dimension returning the prior concentrates in one bin, which would
    # saturate the colour scale for every other dimension. Clip at a high
    # percentile so the informative rows stay readable.
    ax.imshow(H, aspect="auto", origin="lower", cmap=cmap,
              extent=[lo, hi, 0, d], interpolation="nearest",
              vmin=0, vmax=max(np.percentile(H, 99.0), 1e-9))
    # Two per-dimension statistics overlaid as dot columns. Because dimension
    # runs up the y axis, each reads as a near-vertical line.
    #
    # The default matches the reference implementation, which plots the mean of
    # each dimension's histogram in white and its standard deviation in black.
    # For a latent matching the prior the white column therefore sits on 0 and
    # the black column on 1, and any departure is visible per dimension. That
    # is the point of plotting the mean even though it is near zero: a
    # dimension whose mean has drifted off the axis shows up immediately.
    rows = np.arange(d) + 0.5
    mean = values.mean(axis=0)
    sd = values.std(axis=0)
    p95 = np.percentile(np.abs(values), 95, axis=0)
    if dots == "mean_sd":
        a, b = mean, sd
        lab_a, lab_b = "per-dim mean", "per-dim sd"
        lab = "white: per-dimension mean    black: per-dimension sd"
    elif dots == "sd_p95":
        a, b = sd, p95
        lab_a, lab_b = "per-dim sd", "per-dim p95 of |value|"
        lab = "white: per-dimension sd    black: per-dimension 95th percentile of |value|"
    elif dots == "sd_only":
        a, b = sd, None
        lab_a, lab_b = "per-dim sd", None
        lab = "white: per-dimension sd"
    else:
        raise ValueError(f"dots must be mean_sd, sd_p95 or sd_only; got {dots!r}")
    # Plain white and black dots vanish against a rainbow field, so each
    # marker carries the opposite colour as an outline. That keeps both
    # columns legible wherever they happen to fall.
    ax.scatter(a, rows, s=11, c="white", edgecolors="black", linewidths=.55,
               zorder=5, label=lab_a)
    if b is not None:
        ax.scatter(b, rows, s=11, c="black", edgecolors="white", linewidths=.55,
                   zorder=5, label=lab_b)
    ax.set_xlim(lo, hi); ax.set_ylim(0, d)
    ax.set_ylabel("Latent variables", fontsize=9)
    leg = ax.legend(fontsize=7, loc="upper right", framealpha=.88,
                    markerscale=1.2, handletextpad=.4, borderpad=.3, labelspacing=.3)
    leg.set_zorder(6)
    ax.set_title(f"{title}, ave: {values.mean():.2f}, std: {values.std():.2f}\n{lab}",
                 fontsize=9)
    ax.tick_params(labelsize=8)


def build_figure(D: dict, out_path: str, title: str, dots: str = "mean_sd",
                 cov_style: str = "matched", cov_gamma: float = 2.5) -> None:
    mu, sigma, z = D["mu"], D["sigma"], D["z"]
    d = mu.shape[1]

    fig = plt.figure(figsize=(16.5, 21.5))
    gs = fig.add_gridspec(4, 3, height_ratios=[3.05, 2.25, 2.05, 2.35],
                          hspace=.30, wspace=.22,
                          left=.055, right=.975, top=.930, bottom=.035)

    # ---------------- block 1: image rows ----------------
    ax = fig.add_subplot(gs[0, :])
    grid = montage([D["x"], D["rec_z"], D["rec_mu"],
                    D["x_train"], D["rec_train"], D["gen"]])
    ax.imshow(grid, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    h = D["x"].shape[1] + 2
    for r, lab in enumerate(ROW_LABELS):
        ax.text(-6, 2 + r * h + h / 2, lab, ha="right", va="center", fontsize=9)
    ax.set_title("Rows: " + ",  ".join(ROW_LABELS), fontsize=9.5, pad=10)

    # ---------------- block 2: per-dimension densities ----------------
    density_panel(fig.add_subplot(gs[1, 0]), mu, -4, 4, r"$\mu$", dots=dots)
    density_panel(fig.add_subplot(gs[1, 1]), sigma, 0, 1.5, r"$\sigma$", dots=dots)
    density_panel(fig.add_subplot(gs[1, 2]), z, -4, 4, r"$z$", dots=dots)

    # ---------------- block 3: latent geometry ----------------
    ax = fig.add_subplot(gs[2, 0])
    nz = np.linalg.norm(z, axis=1) / np.sqrt(d)
    nm = np.linalg.norm(mu, axis=1) / np.sqrt(d)
    prior = np.linalg.norm(np.random.default_rng(0).normal(size=(20000, d)),
                           axis=1) / np.sqrt(d)
    bins = np.linspace(0, 2, 180)
    for arr, c, lab in [(prior, "#1f77b4", "prior"), (nz, "#ff7f0e", r"$\|z\|$"),
                        (nm, "#2ca02c", r"$\|\mu\|$")]:
        hh, _ = np.histogram(arr, bins=bins, density=True)
        ax.plot(bins[:-1], hh / hh.sum(), lw=1.4 if lab != "prior" else 1.0,
                ls="--" if lab == "prior" else "-", color=c, label=lab)
    ax.set_xlabel(r"Normalised by $\sqrt{N_z}$", fontsize=9)
    ax.set_title(f"Norm $z$: {nz.mean():.1f} ({nz.std():.1f}), "
                 f"$\\mu$: {nm.mean():.1f} ({nm.std():.1f})", fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.tick_params(labelsize=8)

    ax = fig.add_subplot(gs[2, 1])
    zz = z / np.linalg.norm(z, axis=1, keepdims=True)
    idx = np.random.default_rng(0).choice(len(zz), size=(2, 8000))
    cos = (zz[idx[0]] * zz[idx[1]]).sum(1)
    edges = np.linspace(-1, 1, 201)
    centres = (edges[:-1] + edges[1:]) / 2
    ref = np.exp(-centres ** 2 * d / 2)
    hh, _ = np.histogram(cos, bins=edges, density=True)
    ax.plot(centres, ref / ref.sum(), ls="--", lw=1.0, color="#1f77b4", label="prior")
    ax.plot(centres, hh / hh.sum(), lw=1.6, color="#ff7f0e", label="$z$")
    ax.set_title(r"$\cos(z_i, z_j)$, $i$ and $j$ random" +
                 f"\nobserved sd {cos.std():.3f}, prior {1/np.sqrt(d):.3f}", fontsize=9.5)
    ax.set_xlim(-1, 1); ax.legend(fontsize=8); ax.grid(alpha=.3); ax.tick_params(labelsize=8)

    ax = fig.add_subplot(gs[2, 2])
    zc = z - z.mean(0)
    cov = (zc.T @ zc) / (len(zc) - 1)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(sd, sd)
    mask = ~np.eye(d, dtype=bool)
    off_cov = np.abs(cov[mask])
    off_r = np.abs(corr[mask])
    floor = np.sqrt(2 / np.pi) / np.sqrt(len(z))

    if cov_style == "matched":
        # The reference rendering: absolute covariance on a black-to-yellow
        # scale with vmax at the diagonal. The diagonal therefore glows and
        # everything else sits near black unless it is an appreciable
        # fraction of the variance. Directly comparable between runs, which
        # is the point of matching it.
        #
        # A LINEAR scale is what makes weak structure look strong. Off-diagonal
        # cells at a few percent of the diagonal still land in the dark-red part
        # of 'hot', which reads as correlation across the whole field even when
        # the values are negligible. A power-law normalisation with gamma above
        # 1 pushes the low end toward black while leaving the diagonal and any
        # genuinely strong cell bright, so the panel shows what stands out
        # rather than what merely exists.
        from matplotlib.colors import PowerNorm
        ax.imshow(np.abs(cov), cmap="hot", interpolation="nearest",
                  norm=PowerNorm(gamma=cov_gamma, vmin=0.0,
                                 vmax=max(np.median(np.diag(cov)), 1e-9)))
    else:
        # Correlation, diagonal removed, diverging scale from the off-diagonal
        # percentile. Harder to compare against the reference but easier to
        # read: sign is visible and the numbers are scale free.
        shown = np.where(mask, corr, np.nan)
        v = max(np.nanpercentile(np.abs(shown), 99), 1e-3)
        im = ax.imshow(shown, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=.046, pad=.02).ax.tick_params(labelsize=7)

    # Whichever rendering is used, the title carries the numbers, including the
    # mean absolute correlation that uncorrelated data of this size would
    # produce anyway. Without that floor a modest value reads as structure.
    ax.set_title(f"Cov($z$)\n"
                 f"mean |off-diag| {off_cov.mean():.4f}, diagonal {np.diag(cov).mean():.3f}   "
                 f"|r| {off_r.mean():.3f} vs floor {floor:.3f} ({off_r.mean()/floor:.1f}x)",
                 fontsize=9)
    ax.tick_params(labelsize=8)

    # ---------------- block 4: loss and PSNR ----------------
    ax = fig.add_subplot(gs[3, :])
    H = D["history"]
    e = H["epoch"]
    ax.semilogy(e, H["train"], lw=1.3, color="#1f77b4", label="Training Loss (LHS-log)")
    ax.semilogy(e, H["val"], lw=1.3, color="#ff7f0e", label="Testing Loss (LHS-log)")
    for ref in D.get("references", []):
        ax.semilogy(ref["epoch"], ref["val"], lw=1.0, ls="--", alpha=.8,
                    label=f"reference: {ref['label']}")
    ax.set_xlabel("Epochs", fontsize=9); ax.tick_params(labelsize=8)
    ax.grid(alpha=.3, which="both")

    if np.isfinite(H["psnr"]).any():
        ax2 = ax.twinx()
        ax2.plot(e, H["psnr"], lw=1.2, color="black", label="Testing PSNR (RHS)")
        ax2.set_yscale("log"); ax2.tick_params(labelsize=8)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], fontsize=7.5, loc="center right")
        ptxt = (f", PSNR: train={H['psnr_train']:.3f}, test={H['psnr'][-1]:.3f}"
                if "psnr_train" in H else f", PSNR test={H['psnr'][-1]:.3f}")
    else:
        ax.legend(fontsize=7.5, loc="center right"); ptxt = ""
    ax.set_title(f"Total loss: train={H['train'][-1]:.3f}, test={H['val'][-1]:.3f}{ptxt}",
                 fontsize=9.5)

    fig.suptitle(title, fontsize=11, y=.982)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #

EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+).*?total\s+(?P<total>[-\d.]+)"
    r".*?recon\s+(?P<recon>[-\d.]+).*?val\s+(?P<val>[-\d.]+)", re.IGNORECASE)


def read_history(path: Path) -> dict:
    e, tr, va, rec = [], [], [], []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            e.append(int(m.group("epoch"))); tr.append(float(m.group("total")))
            va.append(float(m.group("val"))); rec.append(float(m.group("recon")))
    if not e:
        raise SystemExit(f"No epoch lines matched in {path}")
    rec = np.array(rec)
    psnr = (-10 * np.log10(np.clip(rec, 1e-12, None))
            if np.all(rec > 0) else np.full_like(rec, np.nan))
    return {"epoch": np.array(e), "train": np.array(tr), "val": np.array(va),
            "psnr": psnr, "label": Path(path).parent.name}


def collect(config_path, checkpoint, n_cols, max_images, references):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils import build_datamodule, get_device, load_checkpoint, load_config, set_seed
    try:
        from vae_model import VAE
    except ImportError:
        from model import VAE

    cfg = load_config(config_path)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"].get("device", "cuda"))
    model = VAE(cfg).to(device)
    model.load_state_dict(load_checkpoint(checkpoint, device)["model_state"])
    model.eval()

    dm = build_datamodule(cfg)
    train_loader, _, test_loader = dm.normal_loaders()

    def first(loader, n):
        xs = []
        for b in loader:
            xs.append(b["image"])
            if sum(t.size(0) for t in xs) >= n:
                break
        return torch.cat(xs)[:n]

    x = first(test_loader, n_cols).to(device)
    xt = first(train_loader, n_cols).to(device)

    with torch.no_grad():
        o = model(x, sample=True)
        rec_z = o["mean"]
        rec_mu = model(x, sample=False)["mean"]
        rec_tr = model(xt, sample=False)["mean"]
        gen = model.decoder(torch.randn(n_cols, model.z_dim, device=device))[0]

        mus, sigs, zs, n = [], [], [], 0
        for b in test_loader:
            xb = b["image"].to(device)
            m, lv = model.encoder(xb)
            mus.append(m.cpu()); sigs.append(torch.exp(0.5 * lv).cpu())
            zs.append(model.reparameterize(m, lv).cpu())
            n += xb.size(0)
            if n >= max_images:
                break

    run_dir = Path(checkpoint).parent.parent
    hist = read_history(run_dir / "train.log")
    refs = [read_history(Path(p)) for p in references] if references else []

    np_ = lambda t: t.detach().cpu().numpy()[:, 0]
    return cfg, run_dir, {
        "x": np_(x), "rec_z": np_(rec_z), "rec_mu": np_(rec_mu),
        "x_train": np_(xt), "rec_train": np_(rec_tr), "gen": np_(gen),
        "mu": torch.cat(mus)[:max_images].numpy(),
        "sigma": torch.cat(sigs)[:max_images].numpy(),
        "z": torch.cat(zs)[:max_images].numpy(),
        "history": hist, "references": refs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Composite VAE diagnostic page")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cols", type=int, default=16, help="Images per row")
    ap.add_argument("--max-images", type=int, default=2000)
    ap.add_argument("--reference", nargs="*", default=None,
                    help="Other train.log files to overlay on the loss panel")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dots", choices=["mean_sd", "sd_p95", "sd_only"], default="mean_sd",
                    help="Which per-dimension statistics the overlaid dot columns show")
    ap.add_argument("--cov-style", choices=["matched", "correlation"], default="matched",
                    help="matched reproduces the reference rendering; correlation is "
                         "easier to read but not directly comparable to it")
    ap.add_argument("--cov-gamma", type=float, default=2.5,
                    help="Power-law exponent for the matched covariance panel. 1 is "
                         "linear; higher pushes weak cells toward black. 2 to 4 is useful.")
    args = ap.parse_args()

    cfg, run_dir, D = collect(args.config, args.checkpoint, args.cols,
                              args.max_images, args.reference)
    out = args.out or str(run_dir / "dashboard.png")
    epochs = int(D["history"]["epoch"][-1])
    title = (f"{run_dir.name}_{cfg['experiment']['name']}, epoch: {epochs}\n"
             f"z_dim {D['mu'].shape[1]}, {D['mu'].shape[0]} held-out images")
    build_figure(D, out, title, dots=args.dots, cov_style=args.cov_style,
                 cov_gamma=args.cov_gamma)
    print(f"written to {out}")


if __name__ == "__main__":
    main()