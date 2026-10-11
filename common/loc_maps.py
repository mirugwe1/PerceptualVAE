"""
loc_maps.py  --  Tier 1 anomaly maps, all computed post hoc from a trained VAE.

Every function takes a batch x of shape (B,1,H,W) in [0,1] and returns a map
of the same shape where LARGER means MORE ANOMALOUS. Nothing here trains.

    pixel           |x - x_hat|, the baseline every other map is judged against
    normative       (|x - x_hat| - mu(u)) / sd(u), per-pixel z-score against
                    healthy residuals. Ribs are high in every healthy image, so
                    their sd is high and they normalise away.
    prior_sub       max(0, |x - x_hat| - mu(u)), removes structure present in
                    every image without rescaling it
    ssim            (1 - SSIM(x, x_hat)) / 2, structural rather than intensity
                    error, less driven by bright edges
    feat_multiscale feature-space residual at several backbone layers, each
                    normalised by its own energy and averaged after upsampling
    kl_grad         |d KL / d x|, Zimmerer et al. 2018-2019
    elbo_grad       |d (reconstruction + beta KL) / d x|
    sample_var      variance over decodings of K samples from q(z|x)
    ensemble_var    variance of reconstructions across several trained models
    mars_spatial    per-position Mahalanobis on the feature residual, read
                    from a stage-9 fit

Post-processing variants, applied to any map at evaluation time:

    smooth          Gaussian filtering, favouring blob-shaped regions
    lung            out-of-lung pixels set to the lowest score
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #

def gaussian_blur(m: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian, reflect padding so borders are not darkened."""
    if sigma <= 0:
        return m
    r = max(1, int(3 * sigma + 0.5))
    t = torch.arange(-r, r + 1, device=m.device, dtype=m.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    m = F.conv2d(F.pad(m, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    m = F.conv2d(F.pad(m, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return m


def ssim_map(x, y, sigma=1.5, c1=0.01 ** 2, c2=0.03 ** 2):
    """Local SSIM dissimilarity, data range 1. Returns (1 - SSIM) / 2 in [0,1]."""
    mx, my = gaussian_blur(x, sigma), gaussian_blur(y, sigma)
    sxx = gaussian_blur(x * x, sigma) - mx ** 2
    syy = gaussian_blur(y * y, sigma) - my ** 2
    sxy = gaussian_blur(x * y, sigma) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return ((1 - s) / 2).clamp(0, 1)


# --------------------------------------------------------------------------- #
# Lung segmentation
# --------------------------------------------------------------------------- #

class LungMasker:
    """TorchXRayVision's pretrained PSPNet, lungs only.

    Imported lazily because torchxrayvision ships a package called `model`,
    which collided with this project's model.py earlier. If it cannot be
    imported, the lung variants are skipped with a warning rather than the
    whole evaluation failing.

    The mask is DILATED a few pixels. Opacity commonly sits at the lung
    margin, and a tight mask would cut off exactly the boxes being scored.
    """

    def __init__(self, device, dilate: int = 3):
        import torchxrayvision as xrv                     # noqa: F401
        self.xrv = xrv
        self.net = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
        names = [t.lower() for t in self.net.targets]
        self.idx = [i for i, t in enumerate(names) if "lung" in t]
        if not self.idx:
            raise RuntimeError(f"no lung channel among {self.net.targets}")
        self.dilate = dilate

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1]
        z = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        z = z * 2048.0 - 1024.0                           # xrv's expected range
        p = self.net(z)
        if p.min() < 0 or p.max() > 1:                    # logits, not probs
            p = torch.sigmoid(p)
        lung = p[:, self.idx].amax(1, keepdim=True)
        lung = F.interpolate(lung, size=(h, h), mode="bilinear", align_corners=False) > 0.5
        if self.dilate > 0:
            k = 2 * self.dilate + 1
            lung = F.max_pool2d(lung.float(), k, stride=1, padding=self.dilate) > 0.5
        return lung


# --------------------------------------------------------------------------- #
# Map computations
# --------------------------------------------------------------------------- #

def recon(model, x):
    xh, _ = model.reconstruct(x)
    return xh.clamp(0, 1)


def pixel(model, x, xh=None):
    return (x - (recon(model, x) if xh is None else xh)).abs()


@torch.no_grad()
def feat_multiscale(extractor, x, xh):
    """Per-layer squared feature residual, normalised by that layer's own
    energy so that shallow layers, which emit far more elements, do not
    dominate, then averaged after upsampling to the image size."""
    h = x.shape[-1]
    out = 0
    fx, fr = extractor(x), extractor(xh)
    for a, b in zip(fx, fr):
        d = ((a - b) ** 2).sum(1, keepdim=True)
        e = (a ** 2).sum(1, keepdim=True).mean((2, 3), keepdim=True).clamp(min=1e-8)
        out = out + F.interpolate(d / e, size=(h, h), mode="bilinear", align_corners=False)
    return out / len(fx)


def _kl(mu, logvar):
    dims = tuple(range(1, mu.dim()))
    return 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(dims)


def kl_grad(model, x):
    """|d KL / d x|. The encoder Jacobian contracted with the KL gradient."""
    with torch.enable_grad():
        xi = x.detach().clone().requires_grad_(True)
        mu, logvar = model.encoder(xi)
        g, = torch.autograd.grad(_kl(mu, logvar).sum(), xi)
    return g.abs()


def elbo_grad(model, x, beta: float):
    """|d (per-pixel MSE + beta KL / HW) / d x|, the objective's own weighting."""
    with torch.enable_grad():
        xi = x.detach().clone().requires_grad_(True)
        mu, logvar = model.encoder(xi)
        xh, _ = model.decoder(mu)
        hw = x.shape[-1] * x.shape[-2]
        loss = ((xi - xh) ** 2).mean((1, 2, 3)) + beta * _kl(mu, logvar) / hw
        g, = torch.autograd.grad(loss.sum(), xi)
    return g.abs()


@torch.no_grad()
def sample_var(model, x, k: int):
    mu, logvar = model.encoder(x)
    xs = torch.stack([model.decoder(model.reparameterize(mu, logvar))[0].clamp(0, 1)
                      for _ in range(k)])
    return xs.var(0, unbiased=True)


@torch.no_grad()
def ensemble_var(models, x):
    xs = torch.stack([recon(m, x) for m in models])
    return xs.var(0, unbiased=True)


# --------------------------------------------------------------------------- #
# Tier 2: change the reconstruction itself
# --------------------------------------------------------------------------- #
#
# Tier 1 re-scored the VAE's output. Tier 2 changes WHAT is reconstructed, so
# that x_hat becomes "this patient, healthy" and the residual isolates the
# lesion. Each function returns the RESTORED IMAGE, not a map, so that one
# restoration can be scored several ways (L1 and SSIM) without recomputing it.


class _frozen:
    """Disable parameter gradients while optimising the INPUT, so autograd
    does not build or store gradients for millions of weights it will never
    use, then restore the previous state exactly."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.prev = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad_(False)

    def __exit__(self, *a):
        for p, r in zip(self.model.parameters(), self.prev):
            p.requires_grad_(r)


def project(model, x, steps=30, lr=5e-3, lam=0.1, beta=1.0):
    """Gradient-descent projection onto the healthy manifold (Dehaene 2020).

        x* = argmin_x'  ||x' - D(mu(x'))||^2  +  beta KL(x')/HW  +  lam ||x' - x||_1

    The first two terms pull the input toward something the VAE explains
    well, that is, toward healthy anatomy. The L1 term holds it to the
    original. Its subgradient is a constant pull of size lam, so a pixel
    moves only where the healthy-manifold gradient exceeds that pull: the
    change is SPARSE, concentrated where the input is least plausible.

    lam sets the trade-off and is the parameter to sweep. Too large and
    nothing moves; too small and the whole image is projected, which
    reduces the residual to the ordinary reconstruction error.
    """
    x0 = x.detach()
    xp = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([xp], lr=lr)
    hw = x.shape[-1] * x.shape[-2]
    with _frozen(model), torch.enable_grad():
        for _ in range(int(steps)):
            mu, logvar = model.encoder(xp)
            xh, _ = model.decoder(mu)
            loss = (((xp - xh) ** 2).mean((1, 2, 3))
                    + beta * _kl(mu, logvar) / hw
                    + lam * (xp - x0).abs().mean((1, 2, 3)))
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            opt.step()
            with torch.no_grad():
                xp.clamp_(0, 1)
    return xp.detach()


@torch.no_grad()
def restore(model, x, pix_mu, pix_sd, iters=3, z_thr=2.0):
    """Normative-guided pseudo-healthy restoration.

        m_k = 1[ z(x, x_hat_{k-1}) > tau ]
        x_k = m_k * x_hat_{k-1} + (1 - m_k) * x
        x_hat_k = D(mu(x_k))

    Suspected pathology is replaced by its reconstruction and the patient's
    own tissue is kept everywhere else; then the image is re-encoded. Each
    pass removes suspected pathology BEFORE encoding, so the encoder sees
    less of it and the reconstruction drifts toward this patient, healthy.

    The mask uses the NORMATIVE score, the residual z-scored per pixel
    against healthy validation residuals. That calibrates the threshold on
    healthy data and, crucially, ignores ribs: they are poorly reconstructed
    in every healthy image too, so their healthy sd is high and their z low.
    A raw pixel-residual mask would instead replace the patient's real ribs
    with blurred ones, destroying the anatomy the method is meant to keep.
    """
    xh = recon(model, x)
    for _ in range(int(iters)):
        z = ((x - xh).abs() - pix_mu) / (pix_sd + 1e-4)
        xk = torch.where(z > z_thr, xh, x)
        xh = recon(model, xk)
    return xh


@torch.no_grad()
def truncate(model, x, scale=0.5):
    """Shrink the latent toward the prior mean before decoding.

        x_hat_s = D(s * mu(x)),   0 < s < 1

    The prior mean is the most typical point in latent space. Pathology is
    encoded in rare directions, far from it, so shrinking suppresses those
    directions more than the common anatomy that dominates every code.
    s = 1 is ordinary reconstruction; s -> 0 decodes the average chest.
    """
    mu, _ = model.encoder(x)
    xh, _ = model.decoder(scale * mu)
    return xh.clamp(0, 1)
