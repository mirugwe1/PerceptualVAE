"""
build_rsna_cache.py

Decodes every RSNA DICOM once and writes a uint8 .npy at the target
resolution, so training reads arrays instead of re-decoding 1024x1024 DICOMs
every epoch.

Without this, the first epoch spends roughly half an hour on decoding and the
figure is invisible in the loss curve, which makes it easy to mistake for slow
training. Run it once per image_size.

    python build_rsna_cache.py --config ../stage1_vae_rsna/config.yaml
    python build_rsna_cache.py --config ../stage1_vae_rsna/config.yaml --workers 12

Safe to interrupt and re-run: existing cache entries are skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import load_config          # noqa: E402
from rsna_dataset import RSNAIndex     # noqa: E402


def _one(args) -> str | None:
    """Decode a single study. Returns an error string, or None on success."""
    pid, images_dir, cache_dir, size = args
    out = Path(cache_dir) / f"{pid}_{size}.npy"
    if out.is_file():
        return None
    try:
        import pydicom
        import torch
        import torch.nn.functional as F

        dcm = pydicom.dcmread(Path(images_dir) / f"{pid}.dcm")
        arr = dcm.pixel_array
        # MONOCHROME1 is a photographic negative and must be inverted, or those
        # studies look like enormous anomalies to the detector.
        if getattr(dcm, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            arr = arr.max() - arr
        arr = arr.astype(np.float32)
        hi = float(arr.max())
        if hi > 0:
            arr = arr / hi
        t = torch.from_numpy(arr)[None, None]
        t = F.interpolate(t, size=(size, size), mode="bilinear",
                          align_corners=False, antialias=True)
        np.save(out, (t[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8))
        return None
    except Exception as exc:                       # noqa: BLE001
        return f"{pid}: {type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-build the RSNA image cache")
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=6,
                    help="Match this to the cores you actually have")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["data"].get("module", "covid").lower() != "rsna":
        raise SystemExit("This config is not data.module: rsna")

    cache_dir = cfg["data"].get("cache_dir")
    if not cache_dir:
        raise SystemExit("data.cache_dir is not set, so there is nothing to build")
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    size = cfg["data"]["image_size"]
    index = RSNAIndex(cfg["paths"]["data_root"])

    # Everything the pipeline will touch: the training class and every
    # pathology class, so no evaluation run pays the decoding cost either.
    pids = []
    for cls in index.by_class:
        pids.extend(index.by_class[cls])
    todo = [p for p in pids if not (cache_dir / f"{p}_{size}.npy").is_file()]

    print(f"cache dir : {cache_dir}")
    print(f"resolution: {size}x{size}")
    print(f"total     : {len(pids)}   already cached: {len(pids) - len(todo)}   to do: {len(todo)}")
    if not todo:
        print("nothing to do")
        return

    jobs = [(p, str(index.images_dir), str(cache_dir), size) for p in todo]
    errors, done, t0 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_one, j) for j in jobs]
        for f in as_completed(futures):
            err = f.result()
            if err:
                errors.append(err)
            done += 1
            if done % 500 == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / el
                print(f"  {done}/{len(todo)}  {rate:.0f} img/s  "
                      f"eta {(len(todo)-done)/max(rate,1e-9)/60:.1f} min")

    mb = sum(f.stat().st_size for f in cache_dir.glob(f"*_{size}.npy")) / 1024 / 1024
    print(f"\ndone in {(time.time()-t0)/60:.1f} min, cache is {mb:.0f} MB")
    if errors:
        print(f"{len(errors)} failures:")
        for e in errors[:20]:
            print("  " + e)


if __name__ == "__main__":
    main()