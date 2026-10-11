"""
mars_setup.py  --  shared loading for the stage-9 scripts.

THE SPLIT MUST MATCH THE SOURCE VAE

Stage 9 scores a VAE trained in another stage. The data split is seeded, so
if stage 9 built its datamodule from its own config with a different seed,
it would evaluate on a test set that overlaps the source model's training
set and every number would be inflated. The full configuration is therefore
taken from the SOURCE CHECKPOINT, which train.py saves alongside the
weights, and only the stage-9 blocks (mars, paths, evaluation) are overlaid
on top. That guarantees the identical split by construction rather than by
remembering to copy a seed.
"""

from __future__ import annotations

import copy
import glob
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perceptual import build_extractor                          # noqa: E402
from utils import get_device, load_checkpoint, load_config      # noqa: E402

try:
    from vae_model import VAE
except ImportError:
    from model import VAE


def resolve_checkpoint(spec: str) -> str:
    """Accepts an explicit path or a glob; a glob resolves to the newest match."""
    p = os.path.expanduser(spec)
    if os.path.isfile(p):
        return p
    hits = sorted(glob.glob(p), key=os.path.getmtime)
    if not hits:
        raise SystemExit(f"No source checkpoint matches {spec!r}")
    return hits[-1]


def setup(config_path: str):
    """Returns (cfg, model, extractor, device, source_ckpt_path)."""
    stage = load_config(config_path)
    device = get_device(stage["experiment"].get("device", "cuda"))

    src = resolve_checkpoint(stage["mars"]["source_checkpoint"])
    ck = load_checkpoint(src, device)
    base = ck.get("config")
    if base is None:
        raise SystemExit(
            f"{src} carries no saved config, so the source split cannot be "
            "reproduced. Re-evaluating on a guessed split would risk test/train "
            "overlap; refusing rather than guessing.")

    # The source config supplies model, data, and the seed that fixes the
    # split. Stage 9 contributes only what is its own.
    cfg = copy.deepcopy(base)
    # mars and evaluation are stage 9's own, so they replace whole.
    for k in ("mars", "evaluation"):
        if k in stage:
            cfg[k] = stage[k]
    # paths is MERGED, not replaced. The source supplies data_root, cache
    # locations and anything else the datamodule reads; stage 9 contributes
    # only where its own outputs go. Replacing the block wholesale discards
    # data_root and the datamodule fails to find the images.
    cfg.setdefault("paths", {})
    cfg["paths"].update(stage.get("paths", {}))
    cfg["experiment"]["name"] = stage["experiment"]["name"]

    model = VAE(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    m = cfg["mars"]
    layers = m.get("layers") or (["layer2", "layer3"] if "resnet" in m["backbone"]
                                 else ["relu3_4", "relu4_4"])
    m["layers"] = layers
    extractor = build_extractor(m["backbone"], layers=layers).to(device).eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    return cfg, model, extractor, device, src