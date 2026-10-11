"""
losses.py

The training objective, rewritten so that every weight is interpretable.

WHY THE REDUCTION CHANGED. The reproduction summed the Gaussian log
likelihood over all 16,384 pixels and summed the KL over latent dimensions,
then averaged over the batch. Mathematically that is a valid ELBO. Practically
it makes every hyperparameter resolution dependent and mutually
incomparable: the reconstruction term sits near 1.5e4, so a perceptual term
near 1e0 needs a weight near 1e3 to matter, and that weight has to be
re-derived if the image size changes.

Here:

    recon      is a MEAN over pixels
    kl         is summed over latent dimensions, then divided by the pixel count
    perceptual is a mean over feature maps

All three are order 1, so beta and the perceptual weight are dimensionless,
mean the same thing at any resolution, and can be reasoned about instead of
tuned blind.

    total = recon + beta_t * kl + w_perc * perceptual

FREE BITS. Without them, a KL term this cheap will drive unused latent
dimensions to exactly the prior and leave them dead. Free bits clamp the
per-dimension KL from below, so the objective stops rewarding collapse once a
dimension carries `free_bits` nats and spends its effort elsewhere.

KL WARMUP. beta is ramped linearly from 0 over the first `warmup_epochs`. A
VAE that meets full KL pressure from step 0 frequently collapses to the mean
image and never recovers, which for an anomaly detector produces a model that
reconstructs everything equally badly and separates nothing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_2PI = math.log(2.0 * math.pi)


def gaussian_nll(x, mean, log_sigma):
    """Per-pixel negative log likelihood under N(mean, sigma).

    With a learned sigma this is the term that lets the model say "this pixel
    was never predictable", which is exactly what you want over rib margins
    and the diaphragm.
    """
    inv_var = torch.exp(-2.0 * log_sigma)
    return 0.5 * (LOG_2PI + 2.0 * log_sigma + (x - mean) ** 2 * inv_var)


def reconstruction_term(x, mean, log_sigma, kind: str):
    if kind == "gaussian_nll":
        if log_sigma is None:
            raise ValueError("recon: gaussian_nll requires model.predict_sigma: true")
        return gaussian_nll(x, mean, log_sigma).flatten(1).mean(dim=1)
    if kind == "mse":
        return F.mse_loss(mean, x, reduction="none").flatten(1).mean(dim=1)
    if kind == "l1":
        return F.l1_loss(mean, x, reduction="none").flatten(1).mean(dim=1)
    raise ValueError(f"unknown reconstruction kind {kind!r}")


def kl_term(mu, logvar, free_bits: float = 0.0):
    """KL(q||N(0,I)), per latent dimension, with an optional free-bits floor.

    Returns (kl_per_sample, mean_kl_per_dimension). The second value is worth
    logging: comparing it against free_bits tells you immediately whether
    dimensions are alive or pinned at the floor, which aggregate KL hides.
    """
    per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    raw_mean = per_dim.flatten(1).mean(dim=1)
    if free_bits > 0:
        per_dim = torch.clamp(per_dim, min=free_bits)
    return per_dim.flatten(1).sum(dim=1), raw_mean


def beta_at_epoch(beta: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return beta
    return beta * min(1.0, epoch / float(warmup_epochs))


class VAEObjective(nn.Module):
    def __init__(self, cfg: dict, perceptual_fn=None):
        super().__init__()
        loss_cfg = cfg["loss"]
        self.kind = loss_cfg.get("recon", "gaussian_nll")
        self.beta = float(loss_cfg.get("beta", 1.0))
        self.free_bits = float(loss_cfg.get("free_bits", 0.0))
        self.warmup_epochs = int(loss_cfg.get("kl_warmup_epochs", 0))
        self.n_pixels = cfg["data"]["image_size"] ** 2

        perc_cfg = loss_cfg.get("perceptual", {}) or {}
        self.perceptual_fn = perceptual_fn
        self.w_perc = float(perc_cfg.get("weight", 0.0)) if perceptual_fn else 0.0

    def forward(self, x, out: dict, epoch: int) -> dict:
        recon = reconstruction_term(x, out["mean"], out["log_sigma"], self.kind).mean()

        kl_sum, kl_per_dim = kl_term(out["mu"], out["logvar"], self.free_bits)
        # Divided by the pixel count so that beta is dimensionless and directly
        # comparable against the per-pixel reconstruction term.
        kl = (kl_sum / self.n_pixels).mean()

        beta_t = beta_at_epoch(self.beta, epoch, self.warmup_epochs)

        if self.perceptual_fn is not None and self.w_perc > 0:
            perc = self.perceptual_fn(out["mean"], x)
        else:
            perc = x.new_zeros(())

        total = recon + beta_t * kl + self.w_perc * perc

        weighted_kl = beta_t * kl
        weighted_perc = self.w_perc * perc
        denom = total.detach().abs().clamp_min(1e-12)

        return {
            "total": total,
            "recon": recon.detach(),
            "kl": kl.detach(),
            "kl_per_dim": kl_per_dim.mean().detach(),
            "perceptual": perc.detach(),
            "beta_t": torch.tensor(beta_t, device=x.device),
            # Shares are logged so that a term that contributes nothing is
            # visible in epoch 1 rather than discovered after the run.
            "share_kl": (weighted_kl.detach().abs() / denom),
            "share_perc": (weighted_perc.detach().abs() / denom),
        }
