"""
scoring.py

Turning a reconstruction into an anomaly score. This module, not the
architecture, is where most of the detection performance lives.

THE POOLING PROBLEM. The reproduction scored an image by the mean absolute
residual over every pixel. For focal disease that is close to the worst
available choice: a lesion covering three percent of the frame has its signal
divided by roughly thirty-three before it reaches the score. It is the direct
cause of image-level separations in the region of a few percent.

Top-k pooling, the mean of the largest k percent of residual pixels, keeps
the focal signal. Both are computed here and reported side by side, because
the comparison itself says whether a model's failure is focal or global.

THE MASKING PROBLEM. Each class in this dataset comes from a different source
repository, so borders, padding and burned-in markers carry class
information. Those regions also reconstruct badly, so they enter the residual
and inflate the score for reasons unrelated to pathology. Scoring inside the
lung mask removes the easiest of these shortcuts. The model still sees the
whole image; only the score is restricted.

THE SCALE PROBLEM. When the model predicts a per-pixel sigma, the raw
residual is the wrong quantity: it is largest exactly where the image is
intrinsically unpredictable (rib margins, diaphragm, mediastinal border). The
normalised residual |x - mean| / sigma discounts those regions and is the
default here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Residual maps
# --------------------------------------------------------------------------- #

def pixel_residual(x, mean, log_sigma=None, normalize: bool = True) -> torch.Tensor:
    """Per-pixel anomaly map, shape (B, 1, H, W)."""
    res = (x - mean).abs()
    if normalize and log_sigma is not None:
        res = res / torch.exp(log_sigma)
    return res.mean(dim=1, keepdim=True)


@torch.no_grad()
def feature_residual(extractor, x, mean, out_size: int) -> torch.Tensor:
    """Anomaly map in the feature space of a frozen backbone.

    Each selected layer contributes a spatial map of squared feature
    distance, channel-averaged, upsampled back to image resolution and
    averaged across layers. Feature-space residuals are far less dominated by
    high-frequency texture than pixel residuals, which matters on radiographs
    where rib edges dominate the pixel view.
    """
    f_x = extractor(x)
    f_r = extractor(mean)
    maps = []
    for a, b in zip(f_r, f_x):
        if a.dim() == 3:  # ViT patch tokens (B, N, D) -> (B, D, g, g)
            bsz, n, dim = a.shape
            g = int(round(n ** 0.5))
            a = a.transpose(1, 2).reshape(bsz, dim, g, g)
            b = b.transpose(1, 2).reshape(bsz, dim, g, g)
        d = (a - b).pow(2).mean(dim=1, keepdim=True)
        maps.append(F.interpolate(d, size=(out_size, out_size),
                                  mode="bilinear", align_corners=False))
    return torch.stack(maps).mean(0)


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #

def pool_score(amap: torch.Tensor, mask: torch.Tensor | None, mode: str) -> torch.Tensor:
    """Reduce an anomaly map to one score per sample.

    mode: "mean", "max", or "topN" where N is a percentage, e.g. "top1",
    "top5". Pixels outside the mask are excluded entirely rather than set to
    zero, which would otherwise drag the mean down in proportion to how much
    of the frame the lungs occupy, making the score depend on body habitus
    and framing.
    """
    b = amap.shape[0]
    flat = amap.reshape(b, -1)

    if mask is not None:
        mflat = mask.reshape(b, -1) > 0.5
    else:
        mflat = torch.ones_like(flat, dtype=torch.bool)

    scores = []
    for i in range(b):
        vals = flat[i][mflat[i]]
        if vals.numel() == 0:          # empty mask: fall back to the full frame
            vals = flat[i]
        if mode == "mean":
            scores.append(vals.mean())
        elif mode == "max":
            scores.append(vals.max())
        elif mode.startswith("top"):
            pct = float(mode[3:])
            k = max(1, int(round(vals.numel() * pct / 100.0)))
            scores.append(torch.topk(vals, k).values.mean())
        else:
            raise ValueError(f"unknown pooling mode {mode!r}")
    return torch.stack(scores)


DEFAULT_POOLINGS = ["mean", "top10", "top5", "top1", "max"]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

@torch.no_grad()
def score_loader(model, loader, device, cfg, extractor=None) -> dict:
    """Scores every sample in a loader under several pooling modes at once.

    Returns a dict of pooling name -> numpy array of scores, plus the mean
    residual map statistics. Computing all poolings in one pass costs almost
    nothing and removes the temptation to re-run evaluation until a pooling
    looks good, which would be selection on the test set.
    """
    import numpy as np

    sc = cfg.get("scoring", {})
    poolings = sc.get("poolings", DEFAULT_POOLINGS)
    use_mask = sc.get("use_lung_mask", True)
    normalize = sc.get("normalize_by_sigma", True)
    n_samples = int(sc.get("recon_samples", 1))
    use_features = bool(sc.get("feature_space", False)) and extractor is not None

    out: dict[str, list] = {f"pixel/{p}": [] for p in poolings}
    if use_features:
        out.update({f"feature/{p}": [] for p in poolings})
    paths: list[str] = []

    model.eval()
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True) if use_mask else None
        paths.extend(batch["path"])

        mean, log_sigma = model.reconstruct(x, n_samples=n_samples)

        pmap = pixel_residual(x, mean, log_sigma, normalize=normalize)
        for p in poolings:
            out[f"pixel/{p}"].append(pool_score(pmap, m, p).cpu())

        if use_features:
            fmap = feature_residual(extractor, x, mean, x.shape[-1])
            for p in poolings:
                out[f"feature/{p}"].append(pool_score(fmap, m, p).cpu())

    result = {k: torch.cat(v).numpy() for k, v in out.items()}
    result["_paths"] = np.array(paths)
    return result
