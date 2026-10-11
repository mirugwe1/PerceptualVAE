"""
evaluate.py

Reconstruction fidelity on held-out normals, and detection metrics for every
pathology class under every pooling mode.

WHY ALL POOLINGS ARE REPORTED. Mean pooling and top-k pooling answer
different questions: whether the whole image reconstructs poorly, and whether
some region does. Reporting both is more informative than either, and
computing them in a single pass removes the temptation to re-run evaluation
until one looks good, which would be selection on the test set. The pooling
declared primary in the config is the one to quote; the rest are context.

Every detection metric carries a bootstrap 95 percent interval. Three
variants separated by 0.02 AUROC with intervals half a point wide are not
three different results.

Usage:
    python evaluate.py --config config.yaml --checkpoint <path/best.pt>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import (                                    # noqa: E402
    LPIPSMetric, batch_ssim, detection_metrics, pixel_localisation,
)
from perceptual import build_extractor                  # noqa: E402
from scoring import pixel_residual, score_loader        # noqa: E402
from utils import (                                     # noqa: E402
    build_datamodule, get, get_device, get_logger, has_pixel_ground_truth,
    load_checkpoint, load_config, set_seed,
)
try:                                                    # renamed because
    from vae_model import VAE                           # torchxrayvision
except ImportError:                                     # ships a package
    from model import VAE                               # called 'model'



@torch.no_grad()
def fidelity(model, loader, device, cfg, logger) -> dict:
    """Reconstruction quality on held-out normals only.

    These describe how well the model rebuilds its own training distribution.
    They are not detection metrics and say nothing on their own about whether
    pathology is found, which is worth stating because a perceptual loss
    reliably improves them whether or not detection improves at all.
    """
    xs, rs = [], []
    n_samples = int(get(cfg, "scoring.recon_samples", 1))
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        mean, _ = model.reconstruct(x, n_samples=n_samples)
        xs.append(x.cpu())
        rs.append(mean.cpu())
    x, r = torch.cat(xs), torch.cat(rs)

    out = {"n_samples": int(x.size(0))}
    metrics = [m.lower() for m in cfg["evaluation"]["metrics"]]
    if "ssim" in metrics:
        out["ssim"] = batch_ssim(x, r)
    if "lpips" in metrics:
        out["lpips"] = LPIPSMetric(device)(x, r)

    # Anything else named in evaluation.metrics that fidelity_metrics knows
    # about. These come from the chest-radiograph column of Kastryulin et al.'s
    # benchmark against radiologist judgement, where SSIM ranks tenth of
    # fourteen and PSNR twelfth. Since the central claim of this work is that
    # reconstruction improved while detection did not, that claim should not
    # rest on the two weakest measures available.
    #
    # LPIPS is kept but read with care: it is approximately what the
    # perceptual variants minimise, so improving it is partly circular. The
    # others are neither weak on this modality nor the training objective.
    extra = [m for m in metrics
             if m in ("ms_ssim", "haarpsi", "gmsd", "dists", "fsim", "vsi", "psnr")]
    if extra:
        try:
            from fidelity_metrics import fidelity_metrics
            fm = fidelity_metrics(r.to(device), x.to(device), include=extra)
            if "error" in fm:
                logger.warning(f"  extended fidelity unavailable: {fm['error']}")
            else:
                for k, v in fm.items():
                    if k.startswith("_"):
                        continue
                    if k.endswith("_note"):
                        logger.warning(f"  {k}: {v}")
                    out[k] = v
        except ImportError as exc:
            logger.warning(f"  extended fidelity skipped: {exc}")

    logger.info(f"  fidelity: " + ", ".join(
        f"{k} {v:.4f}" for k, v in out.items()
        if isinstance(v, float)))
    return out


@torch.no_grad()
def collect_maps(model, loader, device, cfg, max_images=None):
    """Anomaly maps and box ground truth for a pathology class.

    Only called when the dataset supplies gt_mask. Maps are kept in float16:
    9,555 images at 128 by 128 is about 310 MB that way and 630 MB in float32,
    and the metrics downstream cast back to float64 anyway.
    """
    maps, gts, n = [], [], 0
    normalize = bool(get(cfg, "scoring.normalize_by_sigma", False))
    samples = int(get(cfg, "scoring.recon_samples", 1))
    model.eval()
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        mean, log_sigma = model.reconstruct(x, n_samples=samples)
        amap = pixel_residual(x, mean, log_sigma, normalize=normalize)
        maps.append(amap[:, 0].half().cpu().numpy())
        gts.append(batch["gt_mask"][:, 0].numpy().astype("uint8"))
        n += x.size(0)
        if max_images and n >= max_images:
            break
    return np.concatenate(maps), np.concatenate(gts)


def main(config_path: str, checkpoint_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"].get("device", "cuda"))

    run_dir = Path(checkpoint_path).parent.parent
    logger = get_logger("evaluate", run_dir / "evaluate.log")
    logger.info(f"Checkpoint: {checkpoint_path}")

    ckpt = load_checkpoint(checkpoint_path, device)
    model = VAE(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dm = build_datamodule(cfg)
    dm.validate_layout()
    _, _, normal_test = dm.normal_loaders()
    path_loaders = dm.pathology_loaders()

    # ------------------------------------------------------------------- #
    # Optional feature-space scoring.
    #
    # The backbone is used here as a MEASUREMENT, not as a training loss, and
    # that inverts which backbones are useful. A network fine-tuned to
    # discriminate Normal from pathology develops features that amplify the
    # difference between them. In a reconstruction loss that is harmful: it
    # instructs the decoder to render precisely the structures that mark
    # disease. In a scoring function it is exactly what is wanted, because the
    # score is the distance between a pathological input and its
    # healthy-looking reconstruction.
    #
    # Nothing here touches the trained model, so this runs on an existing
    # checkpoint without retraining.
    # ------------------------------------------------------------------- #
    extractor = None
    if get(cfg, "scoring.feature_space", False):
        fb = get(cfg, "scoring.feature_backbone", "vgg19_imagenet")
        fb_ckpt = get(cfg, "scoring.feature_checkpoint")
        fb_norm = get(cfg, "scoring.feature_normalization")
        if fb_norm is not None:
            fb_norm = (tuple(fb_norm[0]), tuple(fb_norm[1]))
        extractor = build_extractor(
            fb,
            layers=get(cfg, "scoring.feature_layers"),
            checkpoint=fb_ckpt,
            normalization=fb_norm,
        ).to(device)
        logger.info(f"Feature-space scoring via {fb}")
        if hasattr(extractor, "load_report"):
            logger.info(f"  backbone load report: {extractor.load_report}")
            if not extractor.load_report.get("normalization_from_config", True):
                logger.warning(
                    "  scoring.feature_normalization is not set: falling back to "
                    "ImageNet statistics. If this backbone was fine-tuned on a "
                    "different input range, every feature map is shifted and the "
                    "feature-space scores measure something other than intended.")

    primary = get(cfg, "scoring.primary", "pixel/top1")
    n_boot = int(get(cfg, "evaluation.bootstrap", 1000))
    results: dict = {"primary_score": primary}

    logger.info("Held-out Normal...")
    results["normal"] = fidelity(model, normal_test, device, cfg, logger)
    normal_scores = score_loader(model, normal_test, device, cfg, extractor)
    results["normal"]["mean_pixel_residual"] = float(
        normal_scores.get("pixel/mean", np.array([np.nan])).mean())

    score_keys = sorted(k for k in normal_scores if not k.startswith("_"))

    for cls, loader in path_loaders.items():
        logger.info(f"Pathology class: {cls}...")
        path_scores = score_loader(model, loader, device, cfg, extractor)

        block: dict = {"n_samples": int(len(path_scores[score_keys[0]]))}
        by_score = {}
        for key in score_keys:
            m = detection_metrics(normal_scores[key], path_scores[key],
                                  n_boot=n_boot, seed=cfg["experiment"]["seed"])
            by_score[key] = m
            marker = "  <== primary" if key == primary else ""
            ci = m.get("auroc_ci95", [float("nan")] * 2)
            logger.info(
                f"    {key:16s} AUROC {m['auroc']:.4f} [{ci[0]:.3f}, {ci[1]:.3f}]  "
                f"AUPRC {m['auprc']:.4f} (chance {m['auprc_baseline']:.3f})  "
                f"FPR95 {m['fpr95']:.3f}{marker}")

        block["by_score"] = by_score
        if primary in by_score:
            block["primary"] = by_score[primary]

        # ------------------------------------------------------------- #
        # Pixel-level localisation, where the dataset provides boxes.
        # This is the measurement image-level AUROC cannot make.
        # ------------------------------------------------------------- #
        if has_pixel_ground_truth(cfg):
            cap = get(cfg, "evaluation.pixel_max_images", None)
            amaps, gtm = collect_maps(model, loader, device, cfg, max_images=cap)
            loc = pixel_localisation(amaps, gtm)
            block["localisation"] = loc
            if loc.get("pixel_auprc") is not None:
                logger.info(
                    f"    localisation  pixel AUPRC {loc['pixel_auprc']:.4f} "
                    f"(chance {loc['pixel_baseline']:.4f})  "
                    f"best Dice {loc.get('best_dice', float('nan')):.4f}  "
                    f"IoU {loc.get('mean_iou', float('nan')):.4f}  "
                    f"n={loc['n_images']}")
            else:
                logger.info(f"    localisation  no annotated images in {cls}")
            del amaps, gtm
        block["mean_pixel_residual"] = float(
            path_scores.get("pixel/mean", np.array([np.nan])).mean())
        results[cls] = block

    out_path = run_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Written to {out_path}")

    logger.info("")
    logger.info("Reminder: run controls.py before quoting any of these. On this "
                "dataset a strong AUROC may reflect source differences rather "
                "than pathology.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate the CXR anomaly-detection VAE")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--checkpoint", type=str, required=True)
    args = ap.parse_args()
    main(args.config, args.checkpoint)