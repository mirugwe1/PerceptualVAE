"""
utils.py

Config loading, reproducibility, logging, early stopping and checkpoint IO.
Model-agnostic and shared by every script in cxr_ad/.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def _expand(obj):
    """Recursively expand ~ in any string that looks like a path."""
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str) and (obj.startswith("~") or obj.startswith("$HOME")):
        return str(Path(obj.replace("$HOME", "~")).expanduser())
    return obj


def load_config(path: str | Path) -> dict:
    """Load YAML, expand ~ anywhere it appears, and support a `_base_` key.

    `_base_: base.yaml` merges the current file over that base, so variant
    configs only state their differences. This is deliberate: the point of
    the experiment is that variants differ in one place, and a config format
    that forces you to restate everything makes accidental divergence easy.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    base_name = cfg.pop("_base_", None)
    if base_name:
        base = load_config(path.parent / base_name)
        cfg = deep_merge(base, cfg)

    return _expand(cfg)


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def get(cfg: dict, dotted: str, default=None):
    """cfg lookup by dotted path, e.g. get(cfg, 'loss.perceptual.weight', 0.0)."""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Slower, but removes cuDNN algorithm selection as a source of
        # run-to-run variation. Worth it when comparing variants that are
        # expected to differ by small margins.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #

@dataclass
class EarlyStopping:
    """Stops when the monitored value has not improved by `min_delta` for
    `patience` epochs.

    min_delta defaults to something that can actually fire. A value like 1e-9
    makes early stopping decorative: essentially any numerical wobble counts
    as improvement and resets the counter, so every run costs the full epoch
    budget whether or not it converged.
    """

    patience: int = 30
    min_delta: float = 1e-4
    mode: str = "min"
    best: float | None = None
    counter: int = 0
    should_stop: bool = False

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = (self.best - value) if self.mode == "min" else (value - self.best)
        if improved > self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

def save_checkpoint(state: dict[str, Any], directory: str | Path, filename: str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    torch.save(state, path)
    return path


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def find_latest_checkpoint(runs_dir: str | Path, filename: str = "best.pt") -> Path | None:
    """Newest run under runs_dir. Run names are timestamps, so lexicographic
    sort is chronological."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return None
    candidates = sorted(runs_dir.glob(f"*/checkpoints/{filename}"))
    return candidates[-1] if candidates else None


# --------------------------------------------------------------------------- #
# Data module selection
# --------------------------------------------------------------------------- #

def build_datamodule(cfg: dict):
    """Returns the data module named by data.module.

    Selecting the loader from config rather than by editing imports means the
    COVID-19 Radiography pipeline and the RSNA pipeline can coexist, and a run
    directory's config.resolved.yaml records which dataset actually produced
    it. Both modules expose the same interface: validate_layout(),
    normal_loaders(), pathology_loaders() and normal_control_loaders(), and
    both yield dicts with at least "image", "mask", "label" and "path".

    RSNA additionally yields "gt_mask", the rasterised annotation boxes. That
    is localisation GROUND TRUTH, never an input to scoring.
    """
    which = cfg["data"].get("module", "covid").lower()
    if which == "covid":
        from dataset import CXRDataModule
        return CXRDataModule(cfg)
    if which == "rsna":
        from rsna_dataset import RSNADataModule
        return RSNADataModule(cfg)
    raise ValueError(
        f"Unknown data.module {which!r}. Expected 'covid' or 'rsna'."
    )


def has_pixel_ground_truth(cfg: dict) -> bool:
    """True when the dataset supplies gt_mask, so pixel-level localisation
    metrics can be computed."""
    return cfg["data"].get("module", "covid").lower() == "rsna"