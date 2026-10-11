"""
metrics.py

Reconstruction-fidelity and detection metrics.

CONVENTIONS, fixed once so every variant reports the same thing:

  * Pathology is the positive class. Higher score means more anomalous.
  * AUPRC is always returned with its chance baseline, which is positive
    prevalence, not 0.5. With 1,345 pathology images against ~1,020 held-out
    normals, chance is 0.569, so an AUPRC of 0.65 is a weak result rather
    than a moderate one. The baseline also moves between pathology classes,
    so AUPRC values are not comparable across classes without it.
  * FPR@95TPR is the fraction of normal images flagged when the threshold
    catches 95 percent of pathology. It is a much harsher metric than AUROC
    and a much better proxy for whether anything is clinically usable.

CONFIDENCE INTERVALS. Every detection metric is returned with a bootstrap
interval. Comparing three variants on a single AUROC point estimate invites
reading noise as signal, and with test sets of one to six thousand images the
intervals are wide enough to matter.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


# --------------------------------------------------------------------------- #
# Reconstruction fidelity
# --------------------------------------------------------------------------- #

def _gaussian_window(size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(channels, 1, size, size).contiguous()


def batch_ssim(x, y, data_range: float = 1.0, size: int = 11, sigma: float = 1.5,
               batch_size: int = 256) -> float:
    """Gaussian-windowed SSIM, standard constants. No clamping is applied:
    the decoder is bounded by construction here, so inputs are already in
    range and a clamp would hide it if that ever stopped being true."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    pad, ch = size // 2, x.size(1)
    scores = []
    for i in range(0, x.size(0), batch_size):
        a, b = x[i:i + batch_size], y[i:i + batch_size]
        w = _gaussian_window(size, sigma, ch, a.device, a.dtype)
        mu_a = F.conv2d(a, w, padding=pad, groups=ch)
        mu_b = F.conv2d(b, w, padding=pad, groups=ch)
        va = F.conv2d(a * a, w, padding=pad, groups=ch) - mu_a ** 2
        vb = F.conv2d(b * b, w, padding=pad, groups=ch) - mu_b ** 2
        vab = F.conv2d(a * b, w, padding=pad, groups=ch) - mu_a * mu_b
        s = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / \
            ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
        scores.append(s.flatten(1).mean(dim=1))
    return float(torch.cat(scores).mean())


class LPIPSMetric:
    def __init__(self, device, net: str = "alex", batch_size: int = 64):
        import lpips
        self.model = lpips.LPIPS(net=net).to(device).eval()
        self.device, self.batch_size = device, batch_size

    @torch.no_grad()
    def __call__(self, x, y) -> float:
        out = []
        for i in range(0, x.size(0), self.batch_size):
            a, b = x[i:i + self.batch_size].to(self.device), y[i:i + self.batch_size].to(self.device)
            if a.size(1) == 1:
                a, b = a.repeat(1, 3, 1, 1), b.repeat(1, 3, 1, 1)
            out.append(self.model(a, b, normalize=True).flatten())
        return float(torch.cat(out).mean())


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def _pack(neg: np.ndarray, pos: np.ndarray):
    y = np.concatenate([np.zeros(len(neg), int), np.ones(len(pos), int)])
    s = np.concatenate([neg, pos])
    return y, s


def fpr_at_tpr(neg, pos, level: float) -> float:
    y, s = _pack(neg, pos)
    fpr, tpr, _ = roc_curve(y, s)
    idx = int(np.searchsorted(tpr, level, side="left"))
    return 1.0 if idx >= len(tpr) else float(fpr[idx])


def detection_metrics(neg: np.ndarray, pos: np.ndarray,
                      n_boot: int = 1000, seed: int = 0) -> dict:
    """AUROC, AUPRC, FPR@95, FPR@99, each with a bootstrap 95 percent interval."""
    y, s = _pack(neg, pos)

    point = {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "fpr95": fpr_at_tpr(neg, pos, 0.95),
        "fpr99": fpr_at_tpr(neg, pos, 0.99),
    }

    rng = np.random.default_rng(seed)
    boots: dict[str, list] = {k: [] for k in point}
    for _ in range(n_boot):
        # Stratified resample: the class ratio is held fixed, so the interval
        # reflects sampling noise rather than fluctuation in prevalence.
        ni = rng.integers(0, len(neg), len(neg))
        pi = rng.integers(0, len(pos), len(pos))
        bn, bp = neg[ni], pos[pi]
        by, bs = _pack(bn, bp)
        if by.sum() in (0, len(by)):
            continue
        boots["auroc"].append(roc_auc_score(by, bs))
        boots["auprc"].append(average_precision_score(by, bs))
        boots["fpr95"].append(fpr_at_tpr(bn, bp, 0.95))
        boots["fpr99"].append(fpr_at_tpr(bn, bp, 0.99))

    out = dict(point)
    for k, vals in boots.items():
        if vals:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out[f"{k}_ci95"] = [float(lo), float(hi)]
    out["auprc_baseline"] = float(len(pos) / (len(pos) + len(neg)))
    out["n_normal"], out["n_pathology"] = int(len(neg)), int(len(pos))
    return out


# --------------------------------------------------------------------------- #
# Pixel-level localisation (RSNA only, needs gt_mask)
# --------------------------------------------------------------------------- #

def pixel_localisation(anomaly_maps: np.ndarray, gt_masks: np.ndarray,
                       n_thresholds: int = 64) -> dict:
    """Localisation quality against annotated boxes.

    This is the measurement image-level AUROC cannot make, and it is the whole
    reason for moving to a dataset with boxes. An image-level score says
    whether a radiograph is abnormal; only this says whether the model found
    the right REGION.

    Both arrays are (N, H, W). Only images carrying at least one box
    contribute, since a negative image has no region to find.

    Returned:
      pixel_auprc     average precision over all pixels of positive images.
      pixel_baseline  fraction of those pixels inside a box, i.e. the chance
                      level. Boxes cover roughly a fifth of a frame, so an
                      AUPRC of 0.25 is close to nothing and must be read
                      against this number, exactly as image-level AUPRC is.
      pixel_auroc     included for comparability with published work, but it
                      is optimistic under this class imbalance and AUPRC is
                      the honest figure.
      best_dice       Dice at the single global threshold that maximises it,
                      swept over `n_thresholds` quantiles of the score
                      distribution. An oracle threshold, so an upper bound.
      dice_threshold  the score value achieving it.
      mean_iou        IoU at that same threshold.
      n_images        how many annotated images contributed.
    """
    keep = gt_masks.reshape(len(gt_masks), -1).sum(axis=1) > 0
    if keep.sum() == 0:
        return {"pixel_auprc": None, "n_images": 0,
                "note": "no annotated images in this class"}

    maps = anomaly_maps[keep].astype(np.float64).ravel()
    gts = (gt_masks[keep] > 0.5).astype(np.uint8).ravel()

    out = {
        "n_images": int(keep.sum()),
        "pixel_auprc": float(average_precision_score(gts, maps)),
        "pixel_baseline": float(gts.mean()),
        "pixel_auroc": float(roc_auc_score(gts, maps)),
    }

    # Dice sweep over quantiles of the score distribution rather than a linear
    # range, so the thresholds land where the data actually is.
    qs = np.quantile(maps, np.linspace(0.50, 0.999, n_thresholds))
    best = (-1.0, None)
    gt_sum = gts.sum()
    for t in qs:
        pred = maps >= t
        inter = float(np.logical_and(pred, gts).sum())
        denom = float(pred.sum()) + float(gt_sum)
        if denom == 0:
            continue
        dice = 2.0 * inter / denom
        if dice > best[0]:
            union = float(np.logical_or(pred, gts).sum())
            best = (dice, float(t), inter / union if union else 0.0)
    if best[1] is not None:
        out["best_dice"] = float(best[0])
        out["dice_threshold"] = float(best[1])
        out["mean_iou"] = float(best[2])
    return out