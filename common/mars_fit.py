"""
mars_fit.py  --  the stage-9 equivalent of training.

    python ../common/mars_fit.py --config config.yaml

There is no gradient training in MaRS. "Fitting" means estimating the
distribution of the feature residual over healthy images: one Gaussian for
the global mode, one per grid position for the spatial mode. The result is
written to <run_dir>/mars_fit.npz and read by mars_evaluate.py.

FITTED ON TRAINING NORMALS BY DEFAULT

The residual distribution should describe what the source VAE does to
healthy images it has seen the like of, which is the training split. Fitting
on validation is offered for the case where one wants the statistics
estimated on images the VAE did not train on; note that the VAE reconstructs
its own training images slightly better, so a Gaussian fitted there is
marginally tighter. Report which was used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mars import (GlobalGaussian, SpatialGaussian, global_embed,   # noqa: E402
                  loaders, spatial_embed, unpack)
from mars_setup import setup                                        # noqa: E402
from utils import build_datamodule, get_logger, set_seed            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit MaRS residual Gaussians")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg, model, extractor, device, src = setup(args.config)
    set_seed(cfg["experiment"]["seed"])
    m = cfg["mars"]

    run_dir = (Path(cfg["paths"]["output_root"]).expanduser() /
               cfg["experiment"]["name"] /
               f"{time.strftime('%Y%m%d_%H%M%S')}_seed{cfg['experiment']['seed']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("mars_fit", run_dir / "fit.log")
    logger.info(f"source VAE : {src}")
    logger.info(f"backbone   : {m['backbone']}  layers {m['layers']}")
    logger.info(f"modes      : {m['modes']}   fit on {m.get('fit_on','train')}")

    dm = build_datamodule(cfg)
    (train_l, val_l, _), _ = loaders(dm)
    fit_l = train_l if m.get("fit_on", "train") == "train" else val_l

    payload = {}
    want_g, want_s = "global" in m["modes"], "spatial" in m["modes"]
    G = []
    S = SpatialGaussian() if want_s else None
    t0 = time.time()

    with torch.no_grad():
        for b in fit_l:
            x = unpack(b)[0].to(device, non_blocking=True)
            xh, _ = model.reconstruct(x)
            xh = xh.clamp(0, 1)
            if want_g:
                G.append((global_embed(extractor, x) -
                          global_embed(extractor, xh)).double().cpu())
            if want_s:
                fx, g = spatial_embed(extractor, x)
                fr, _ = spatial_embed(extractor, xh, g)
                R = fx - fr
                if S._S1 is None:
                    D = R.shape[1]
                    d = min(int(m.get("spatial_dim", 100)), D)
                    gen = torch.Generator().manual_seed(cfg["experiment"]["seed"])
                    idx = torch.randperm(D, generator=gen)[:d].to(device)
                    S.begin(idx, g, device)
                    logger.info(f"spatial    : {g}x{g} grid, {D} channels -> {d} kept")
                S.update(R)

    summary = {"source_checkpoint": src, "backbone": m["backbone"],
               "layers": m["layers"], "fit_on": m.get("fit_on", "train"),
               "seed": cfg["experiment"]["seed"]}
    if want_g:
        X = torch.cat(G).numpy()
        gg = GlobalGaussian.fit(X)
        payload.update(gg.save(None, "g"))
        ev = gg.eigvals
        # Condition number of the residual covariance: the anisotropy MaRS
        # exploits, reported as one number. Near 1 means L2 and Mahalanobis
        # will agree and MaRS cannot help.
        cond = float(ev[0] / max(ev[-1], 1e-12))
        top10 = float(ev[:10].sum() / ev.sum())
        summary.update(global_D=int(X.shape[1]), global_N=int(X.shape[0]),
                       shrinkage=gg.shrinkage, condition_number=cond,
                       top10_variance_share=top10)
        logger.info(f"global     : D {X.shape[1]}, N {X.shape[0]}, "
                    f"Ledoit-Wolf shrinkage {gg.shrinkage:.3f}")
        logger.info(f"anisotropy : condition number {cond:.2e}, "
                    f"top-10 directions carry {top10:.1%} of residual variance")
    if want_s:
        n = S.finish(float(m.get("spatial_eps", 0.01)))
        payload.update(S.save())
        summary.update(spatial_grid=int(S.grid), spatial_dim=int(len(S.idx)),
                       spatial_N=int(n))

    np.savez_compressed(run_dir / "mars_fit.npz", **payload)
    (run_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"fitted in {time.time()-t0:.0f} s -> {run_dir/'mars_fit.npz'}")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
