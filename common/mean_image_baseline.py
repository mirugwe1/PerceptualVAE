"""
mean_image_baseline.py

The trivial control: score every image by its distance from the AVERAGE
training normal, with no model at all.

    python ../common/mean_image_baseline.py --config config.yaml

WHY THIS HAS TO BE RUN BEFORE ANY CAPACITY RESULT IS WRITTEN UP

A collapsed VAE reconstructs roughly the mean chest whatever it is given, so
its residual IS a distance from that mean. In this project the collapsed
model, carrying about 10 bits of latent, scored 0.7455 on COVID Lung Opacity
against 0.60 for the fully trained one. That makes the obvious question
unavoidable: does a literal mean image do the same?

If it does, the honest headline is that the method reduces to a trivial
baseline on this benchmark, and every AUROC in the paper needs reading
against this row rather than against 0.5. If it does not, the capacity
result stands and this becomes the floor that proves it.

Either way a reviewer asks within the first page, and the answer costs
minutes on a CPU.

IT ALSO LOCALISES. On RSNA the same residual is scored against the
radiologist boxes, which is the measurement the whole second dataset exists
for. A mean-image pixel AUPRC near the 0.124 the trained models reach would
say that column has a trivial explanation.

WHAT IS AND IS NOT MATCHED TO THE MODELS

Matched: the same splits, the same lung masking, the same pooling functions,
the same bootstrap, the same metric code. The only difference is what
produces the reconstruction.

Not matched: there is no training, so no seed variation and no early
stopping. The bootstrap interval here is sampling uncertainty only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import detection_metrics, pixel_localisation   # noqa: E402
from utils import build_datamodule, get_logger, load_config, set_seed  # noqa: E402


POOLINGS = {
    "mean": lambda r, m: (r * m).sum((1, 2, 3)) / m.sum((1, 2, 3)).clamp(min=1),
    "top10": lambda r, m: _topk_frac(r, m, 0.10),
    "top5": lambda r, m: _topk_frac(r, m, 0.05),
    "top1": lambda r, m: _topk_frac(r, m, 0.01),
    "max": lambda r, m: (r * m).amax((1, 2, 3)),
}


def _topk_frac(r, m, frac):
    out = []
    for i in range(r.shape[0]):
        v = r[i][m[i] > 0].flatten()
        if v.numel() == 0:
            out.append(torch.tensor(0.0)); continue
        k = max(1, int(v.numel() * frac))
        out.append(v.topk(k).values.mean())
    return torch.stack(out)


def unpack(batch):
    """Returns (image, mask, gt_mask) whatever shape the batch arrives in.

    THE TWO DATAMODULES DISAGREE, which is why this exists rather than an
    attribute access. RSNADataModule yields a dict with image, mask and
    gt_mask; the COVID CXRDataModule yields a bare tensor, or a (image, mask)
    tuple where lung masks are enabled. Reading one format and silently
    mis-scoring the other is the failure mode this guards against, so the
    detected format is printed once at startup.
    """
    if isinstance(batch, dict):
        x = batch["image"]
        m = batch.get("mask")
        return x, (m if m is not None else torch.ones_like(x)), batch.get("gt_mask")
    if isinstance(batch, (list, tuple)):
        x = batch[0]
        m = batch[1] if len(batch) > 1 and torch.is_tensor(batch[1]) \
            and batch[1].shape == x.shape else torch.ones_like(x)
        return x, m, None
    return batch, torch.ones_like(batch), None


def loaders(dm):
    """Both naming conventions: normal_loaders (RSNA) and get_normal_loaders
    (COVID). Picking the wrong one is an AttributeError at best."""
    n = getattr(dm, "normal_loaders", None) or getattr(dm, "get_normal_loaders")
    p = getattr(dm, "pathology_loaders", None) or getattr(dm, "get_pathology_loaders")
    return n(), p()


@torch.no_grad()
def mean_training_image(loader) -> torch.Tensor:
    s, n = None, 0
    for b in loader:
        x, _, _ = unpack(b)
        s = x.sum(0) if s is None else s + x.sum(0)
        n += x.size(0)
    return s / n


@torch.no_grad()
def score(loader, mean_img, use_mask: bool, want_maps: bool = False):
    per_pool = {k: [] for k in POOLINGS}
    maps, gts = [], []
    for b in loader:
        x, m, gt = unpack(b)
        if not use_mask:
            m = torch.ones_like(x)
        r = (x - mean_img).abs()
        for k, fn in POOLINGS.items():
            per_pool[k].append(fn(r, m).cpu())
        if want_maps and gt is not None:
            maps.append((r * m)[:, 0].cpu())
            gts.append(gt[:, 0].cpu() if gt.ndim == 4 else gt.cpu())
    out = {k: torch.cat(v).numpy() for k, v in per_pool.items()}
    if want_maps and maps:
        return out, torch.cat(maps).numpy(), torch.cat(gts).numpy()
    return out, None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Mean-image baseline control")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="mean_image_baseline.json")
    ap.add_argument("--no-mask", action="store_true",
                    help="Ignore lung masks even when the dataset supplies them")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    logger = get_logger("mean_baseline", Path(args.out).with_suffix(".log"))

    dm = build_datamodule(cfg)
    (train_loader, _, test_loader), path_loaders = loaders(dm)
    use_mask = (not args.no_mask) and cfg["scoring"].get("use_lung_mask", False)

    # Report the detected batch format once, so a silent mis-read is visible.
    probe = next(iter(test_loader))
    px, pm, pg = unpack(probe)
    logger.info(f"batch format: {type(probe).__name__}; image {tuple(px.shape)}; "
                f"mask {'supplied' if pm.abs().min() == 0 else 'all ones'}; "
                f"gt_mask {'present' if pg is not None else 'absent'}")

    mean_img = mean_training_image(train_loader)
    logger.info(f"mean image over {len(train_loader.dataset)} training normals, "
                f"range [{float(mean_img.min()):.4f}, {float(mean_img.max()):.4f}], "
                f"lung mask {'on' if use_mask else 'off'}")

    normal, _, _ = score(test_loader, mean_img, use_mask)
    results = {"n_normal": int(len(normal["mean"])),
               "lung_mask": bool(use_mask),
               "n_train_images": int(len(train_loader.dataset))}

    want_pixel = bool(cfg["evaluation"].get("pixel_max_images", None) is not None
                      or cfg["data"].get("module") == "rsna")

    for cls, loader in path_loaders.items():
        pos, maps, gts = score(loader, mean_img, use_mask, want_maps=want_pixel)
        block = {"n_samples": int(len(pos["mean"])), "by_score": {}}
        for k in POOLINGS:
            block["by_score"][f"pixel/{k}"] = detection_metrics(normal[k], pos[k])
        block["primary"] = block["by_score"][cfg["scoring"]["primary"]]

        # The measurement image-level AUROC cannot make. Only images carrying
        # at least one box contribute, so a class with no annotations is
        # skipped rather than scored against an empty ground truth.
        if maps is not None and gts is not None and gts.sum() > 0:
            block["localisation"] = pixel_localisation(maps, gts)

        results[cls] = block
        a = block["primary"]
        line = (f"{cls:30s} AUROC {a['auroc']:.4f} "
                f"[{a['auroc_ci95'][0]:.4f}, {a['auroc_ci95'][1]:.4f}]  "
                f"AUPRC {a['auprc']:.4f} (chance {a['auprc_baseline']:.3f})")
        if "localisation" in block:
            L = block["localisation"]
            line += (f"  | pixel AUPRC {L['pixel_auprc']:.4f} "
                     f"(chance {L['pixel_baseline']:.4f}, "
                     f"lift {L['pixel_auprc'] - L['pixel_baseline']:+.4f})")
        logger.info(line)

    Path(args.out).write_text(json.dumps(results, indent=2))
    logger.info(f"written to {args.out}")
    logger.info("Read every trained model's AUROC against THESE numbers, not "
                "against 0.5.")


if __name__ == "__main__":
    main()