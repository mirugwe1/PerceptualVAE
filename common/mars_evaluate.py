"""
mars_evaluate.py  --  scores the test split with the fitted Gaussians.

    python ../common/mars_evaluate.py --config config.yaml --run-dir <run_dir>

Writes eval_results.json in the SAME schema as evaluate.py, so the existing
aggregation and table scripts read it without modification.

SCORES REPORTED, AND WHAT EACH COMPARISON ANSWERS

    l2_residual     uniform norm of the global residual
    mars            Mahalanobis on the global residual, the published method
    maha_features   Mahalanobis on raw features, no reconstruction at all
    spatial/mean    spatial residual Mahalanobis, mean-pooled to an image score
    spatial/max     the same, max-pooled

    mars - l2_residual      was the uniform norm the bottleneck?
    mars - maha_features    does reconstruction add anything over density
                            estimation on raw features? Zero or negative
                            means the VAE is doing no work.
    spatial vs PaDiM        does modelling the residual rather than raw
                            features localise better? The novel comparison.

The spatial mode also yields pixel-level localisation on RSNA, through the
same metric code as every other stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mars import (GlobalGaussian, SpatialGaussian, global_embed,   # noqa: E402
                  loaders, spatial_embed, unpack)
from mars_setup import setup                                        # noqa: E402
from metrics import detection_metrics, pixel_localisation           # noqa: E402
from utils import build_datamodule, get_logger, set_seed            # noqa: E402


@torch.no_grad()
def score_loader(model, extractor, loader, device, gg, gf, sg, size,
                 use_mask, want_maps):
    out = {k: [] for k in ("l2_residual", "mars", "maha_features",
                           "spatial/mean", "spatial/max")}
    maps, gts = [], []
    for b in loader:
        x, gt = unpack(b)
        x = x.to(device, non_blocking=True)
        xh, _ = model.reconstruct(x)
        xh = xh.clamp(0, 1)
        if gg is not None:
            fx = global_embed(extractor, x).double().cpu().numpy()
            fr = global_embed(extractor, xh).double().cpu().numpy()
            r = fx - fr
            out["l2_residual"].append((r ** 2).sum(1))
            out["mars"].append(gg.score(r))
            out["maha_features"].append(gf.score(fx))
        if sg is not None:
            sx, g = spatial_embed(extractor, x, sg.grid)
            sr, _ = spatial_embed(extractor, xh, sg.grid)
            amap = sg.score_map(sx - sr, size)
            if use_mask and isinstance(b, dict) and b.get("mask") is not None:
                mm = b["mask"].to(device)
                amap = amap * mm
                denom = mm.sum((1, 2, 3)).clamp(min=1)
                out["spatial/mean"].append((amap.sum((1, 2, 3)) / denom).cpu().numpy())
            else:
                out["spatial/mean"].append(amap.mean((1, 2, 3)).cpu().numpy())
            out["spatial/max"].append(amap.amax((1, 2, 3)).cpu().numpy())
            if want_maps and gt is not None:
                maps.append(amap[:, 0].half().cpu().numpy())
                gts.append((gt[:, 0] if gt.ndim == 4 else gt).numpy().astype("uint8"))
    res = {k: np.concatenate(v) for k, v in out.items() if v}
    if want_maps and maps:
        return res, np.concatenate(maps), np.concatenate(gts)
    return res, None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate MaRS")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    cfg, model, extractor, device, src = setup(args.config)
    set_seed(cfg["experiment"]["seed"])
    run_dir = Path(args.run_dir)
    logger = get_logger("mars_eval", run_dir / "evaluate.log")
    z = np.load(run_dir / "mars_fit.npz")

    gg = gf = sg = None
    if "g_mu" in z.files:
        gg = GlobalGaussian.load(z, "g")
        # The raw-feature control is fitted here rather than in mars_fit so
        # that both Gaussians come from the identical pass over the data.
    if "sp_mu" in z.files:
        sg = SpatialGaussian.load(z, device)

    dm = build_datamodule(cfg)
    (train_l, val_l, test_l), path_l = loaders(dm)

    if gg is not None:
        fit_l = train_l if cfg["mars"].get("fit_on", "train") == "train" else val_l
        F = []
        with torch.no_grad():
            for b in fit_l:
                F.append(global_embed(extractor, unpack(b)[0].to(device)).double().cpu())
        gf = GlobalGaussian.fit(torch.cat(F).numpy())

    size = cfg["data"]["image_size"]
    use_mask = bool(cfg.get("scoring", {}).get("use_lung_mask", False))
    want_maps = cfg["data"].get("module") == "rsna"

    S_n, _, _ = score_loader(model, extractor, test_l, device, gg, gf, sg,
                             size, use_mask, False)
    results = {"seed": cfg["experiment"]["seed"], "method": "MaRS",
               "source_checkpoint": src, "primary_score": "mars",
               "normal": {"n_samples": int(len(next(iter(S_n.values()))))}}

    logger.info(f"  {'class':30s} " + " ".join(f"{k:>14s}" for k in S_n))
    for cls, loader in path_l.items():
        S_p, maps, gts = score_loader(model, extractor, loader, device, gg, gf, sg,
                                      size, use_mask, want_maps)
        block = {"n_samples": int(len(next(iter(S_p.values())))), "by_score": {}}
        for k in S_n:
            block["by_score"][k] = detection_metrics(S_n[k], S_p[k])
        if maps is not None and gts is not None and gts.sum() > 0:
            block["localisation"] = pixel_localisation(maps, gts)
        results[cls] = block
        logger.info(f"  {cls:30s} " + " ".join(
            f"{block['by_score'][k]['auroc']:14.4f}" for k in S_n))
        if "localisation" in block:
            L = block["localisation"]
            logger.info(f"  {'':30s} pixel AUPRC {L['pixel_auprc']:.4f} "
                        f"(chance {L['pixel_baseline']:.4f})  "
                        f"pixel AUROC {L['pixel_auroc']:.4f}")

    (run_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    logger.info(f"written {run_dir/'eval_results.json'}")


if __name__ == "__main__":
    main()
