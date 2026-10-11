"""
corruption.py  --  training-time corruptions for denoising VAEs (Tier 3).

The model is shown a CORRUPTED image and trained to reconstruct the CLEAN
one. It thereby learns that departures from healthy anatomy are to be
removed rather than reproduced, which is the property every earlier stage
lacked: the plain VAE reconstructs pathology into x_hat, so the residual
loses it. A denoising VAE should output "this patient, healthy" at test
time, leaving an unseen lesion in the residual.

TWO CORRUPTIONS

coarse_noise    Gaussian noise drawn on a low-resolution grid and upsampled,
                after Kascenas et al. (MIDL 2022). Low-frequency noise
                resembles the scale of real lesions far better than pixel
                noise, which a model learns to remove trivially and which
                therefore teaches it nothing about anatomy.

foreign_patch   A square region blended with the same region of ANOTHER
                image in the batch, after the patch interpolation of Tan et
                al.'s FPI (MOOD 2020). Produces corruptions with realistic
                radiographic texture at a lesion-like scale. FPI itself
                trains a per-pixel classifier; here the patch is used as a
                denoising corruption for the VAE instead, which keeps the
                method reconstruction-based.

The TARGET is always the clean image. Corrupting the target as well would
teach the model to reproduce corruptions, the opposite of the intent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def coarse_noise(x, res: int = 16, std: float = 0.2):
    """Low-frequency Gaussian noise. res is the noise grid side; at 128 px,
    res 16 gives blobs of roughly 8 px, comparable to small opacities."""
    B = x.shape[0]
    n = torch.randn(B, 1, res, res, device=x.device, dtype=x.dtype) * std
    n = F.interpolate(n, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return (x + n).clamp(0, 1)


def foreign_patch(x, min_size: int = 16, max_size: int = 48,
                  min_alpha: float = 0.3, max_alpha: float = 0.8):
    """Blend one random square per image with the same square of another
    image in the batch. The donor is a different patient, so the patch
    carries real but WRONG anatomy for its location."""
    B, _, H, W = x.shape
    out = x.clone()
    perm = torch.roll(torch.arange(B, device=x.device), shifts=1)   # never itself
    for b in range(B):
        s = int(torch.randint(min_size, max_size + 1, (1,)))
        y0 = int(torch.randint(0, H - s + 1, (1,)))
        x0 = int(torch.randint(0, W - s + 1, (1,)))
        a = float(torch.empty(1).uniform_(min_alpha, max_alpha))
        out[b, :, y0:y0 + s, x0:x0 + s] = (
            (1 - a) * x[b, :, y0:y0 + s, x0:x0 + s]
            + a * x[perm[b], :, y0:y0 + s, x0:x0 + s])
    return out


def corrupt(x, cfg: dict | None):
    """Applies the configured corruption, or returns x unchanged.

    cfg is training.corruption from the config, for example
        {"type": "coarse_noise", "res": 16, "std": 0.2, "prob": 1.0}
    prob is the fraction of images corrupted; the rest are passed through
    clean, so the model also keeps learning ordinary reconstruction.
    """
    if not cfg or not cfg.get("enabled", False):
        return x
    kind = cfg.get("type", "coarse_noise")
    if kind == "coarse_noise":
        xc = coarse_noise(x, int(cfg.get("res", 16)), float(cfg.get("std", 0.2)))
    elif kind == "foreign_patch":
        xc = foreign_patch(x, int(cfg.get("min_size", 16)), int(cfg.get("max_size", 48)),
                           float(cfg.get("min_alpha", 0.3)), float(cfg.get("max_alpha", 0.8)))
    else:
        raise ValueError(f"unknown corruption type {kind!r}")
    p = float(cfg.get("prob", 1.0))
    if p >= 1.0:
        return xc
    keep = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < p).to(x.dtype)
    return keep * xc + (1 - keep) * x
