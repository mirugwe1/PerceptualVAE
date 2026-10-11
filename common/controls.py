"""
controls.py

Validity checks that run before any result is believed. None of these
involve the trained model's quality; they ask whether the benchmark measures
disease at all.

Motivation. The COVID-19 Radiography Database assembles each class from
different source repositories. DeGrave, Janizek and Lee (Nature Machine
Intelligence, 2021) showed that COVID chest-radiograph models built this way
learn acquisition shortcuts rather than pathology, and appear accurate while
failing in new hospitals. An unsupervised detector is not exempt: a model
trained on one source's contrast curve, border padding and marker style will
reconstruct images from another source worse for reasons that have nothing to
do with lungs.

Three controls:

  1. NORMAL VERSUS NORMAL. Score two halves of the held-out normal test set
     against each other. AUROC must be near 0.5. If it is not, the scoring
     pipeline separates images of the same class, and no pathology number
     from that pipeline can be trusted.

  2. LOW-RESOLUTION SHORTCUT PROBE. Fit a logistic regression on 8x8
     downsampled images. At that resolution no anatomy survives, only global
     intensity and framing. If it separates Normal from a pathology class,
     that separation is available without pathology and the anomaly detector
     can find it too.

  3. BORDER-ONLY PROBE. The same classifier on the outer frame with the lung
     field masked out. This isolates padding, letterboxing and burned-in
     markers. On a dataset with no lung masks the mask is all ones, so this
     probe degenerates into the low-resolution one and is skipped.

  4. METADATA PROBE (RSNA only). View position, age and sex read from the
     DICOM headers. Single-source data removes the repository confound but
     not the clinical one: AP films are usually portable studies on sicker
     patients, so view position alone can separate the classes. A dataset can
     pass probes 2 and 3 and still fail this one.

An AUROC well above chance on probe 2 or 3 does not invalidate the
experiment, but it does mean image-level AUROC on this dataset cannot be
reported as a pathology-detection result without heavy qualification. It is
the main argument for masked scoring, and for moving the primary benchmark to
a same-source dataset with localisation ground truth.

Usage:
    python controls.py --config config.yaml                  # probes only
    python controls.py --config config.yaml --checkpoint ... # + normal vs normal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.model_selection import cross_val_score   # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402
from sklearn.pipeline import make_pipeline            # noqa: E402

from metrics import detection_metrics                 # noqa: E402
from scoring import score_loader                      # noqa: E402
from utils import (                                   # noqa: E402
    build_datamodule, get_device, get_logger, load_checkpoint, load_config,
    set_seed,
)


def _collect(loader, device, mode: str, size: int = 8, limit: int = 2000):
    """Flattened features for the shortcut probes.

    mode 'lowres'  : the whole image at size x size
    mode 'border'  : the image with the lung field zeroed, then size x size
    """
    feats = []
    n = 0
    for batch in loader:
        x = batch["image"].to(device)
        if mode == "border":
            x = x * (1.0 - batch["mask"].to(device))
        x = F.adaptive_avg_pool2d(x, size)
        feats.append(x.flatten(1).cpu().numpy())
        n += x.size(0)
        if n >= limit:
            break
    return np.concatenate(feats)[:limit]


def shortcut_probe(normal_loader, path_loader, device, mode: str, seed: int) -> dict:
    a = _collect(normal_loader, device, mode)
    b = _collect(path_loader, device, mode)
    X = np.concatenate([a, b])
    y = np.concatenate([np.zeros(len(a), int), np.ones(len(b), int)])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    return {
        "mode": mode,
        "auroc_mean": float(auc.mean()),
        "auroc_std": float(auc.std()),
        "n_normal": int(len(a)),
        "n_pathology": int(len(b)),
    }


def metadata_probe(cfg, normal_ids, path_ids, seed: int, sample: int = 1500) -> dict:
    """Can DICOM header fields alone separate the classes?

    Reads ViewPosition, PatientAge and PatientSex, one-hot encodes the
    categoricals, and fits the same cross-validated logistic regression. No
    pixels are involved at all, so anything above chance here is a clinical or
    acquisition confound rather than anatomy.
    """
    import random
    import numpy as np
    try:
        import pydicom
    except ImportError:
        return {"skipped": "pydicom not installed"}

    root = Path(cfg["paths"]["data_root"]).expanduser() / "stage_2_train_images"
    if not root.is_dir():
        return {"skipped": f"{root} not found (metadata probe is RSNA-only)"}

    rng = random.Random(seed)
    a = rng.sample(list(normal_ids), min(sample, len(normal_ids)))
    b = rng.sample(list(path_ids), min(sample, len(path_ids)))

    rows, y = [], []
    for group, lab in ((a, 0), (b, 1)):
        for pid in group:
            f = root / f"{pid}.dcm"
            if not f.is_file():
                continue
            h = pydicom.dcmread(f, stop_before_pixels=True)
            view = str(getattr(h, "ViewPosition", "?"))
            sex = str(getattr(h, "PatientSex", "?"))
            try:
                age = float(getattr(h, "PatientAge", "nan"))
            except (TypeError, ValueError):
                age = float("nan")
            rows.append({"AP": view == "AP", "PA": view == "PA",
                         "M": sex == "M", "F": sex == "F", "age": age})
            y.append(lab)

    if not rows:
        return {"skipped": "no headers read"}

    X = np.array([[r["AP"], r["PA"], r["M"], r["F"], r["age"]] for r in rows],
                 dtype=float)
    col_mean = np.nanmean(X[:, 4])
    X[np.isnan(X[:, 4]), 4] = col_mean
    y = np.array(y)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")

    ap_n = float(np.mean(X[y == 0, 0])) * 100
    ap_p = float(np.mean(X[y == 1, 0])) * 100
    return {"auroc_mean": float(auc.mean()), "auroc_std": float(auc.std()),
            "pct_AP_normal": ap_n, "pct_AP_pathology": ap_p,
            "n_normal": int((y == 0).sum()), "n_pathology": int((y == 1).sum())}


def main(config_path: str, checkpoint: str | None) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"].get("device", "cuda"))

    out_dir = Path(checkpoint).parent.parent if checkpoint else \
        Path(cfg["paths"]["output_root"]) / cfg["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("controls", out_dir / "controls.log")

    dm = build_datamodule(cfg)
    dm.validate_layout()
    _, _, normal_test = dm.normal_loaders()
    path_loaders = dm.pathology_loaders()

    results: dict = {}

    # ------------------------------------------------------------------ #
    # 2 and 3: shortcut probes (no model needed)
    # ------------------------------------------------------------------ #
    logger.info("Shortcut probes: can a linear model separate the classes without anatomy?")
    results["shortcut_probes"] = {}
    for cls, loader in path_loaders.items():
        per_class = {}
        modes = ("lowres",) if not cfg["data"].get("use_masks", True) else ("lowres", "border")
        for mode in modes:
            r = shortcut_probe(normal_test, loader, device, mode, cfg["experiment"]["seed"])
            per_class[mode] = r
            verdict = ("SEVERE" if r["auroc_mean"] > 0.85 else
                       "substantial" if r["auroc_mean"] > 0.70 else
                       "mild" if r["auroc_mean"] > 0.60 else "none detected")
            logger.info(f"  {cls} / {mode}: AUROC {r['auroc_mean']:.3f} "
                        f"+/- {r['auroc_std']:.3f}   shortcut risk: {verdict}")
        results["shortcut_probes"][cls] = per_class

        # DICOM-header probe: no pixels at all. RSNA-only; skipped elsewhere.
        if cfg["data"].get("module", "covid").lower() == "rsna":
            md = metadata_probe(cfg, dm.index.by_class[dm.normal_class],
                                dm.index.by_class[cls], cfg["experiment"]["seed"])
            per_class["metadata"] = md
            if "skipped" in md:
                logger.info(f"  {cls} / metadata: skipped ({md['skipped']})")
            else:
                verdict = ("SEVERE" if md["auroc_mean"] > 0.85 else
                           "substantial" if md["auroc_mean"] > 0.70 else
                           "mild" if md["auroc_mean"] > 0.60 else "none detected")
                logger.info(
                    f"  {cls} / metadata: AUROC {md['auroc_mean']:.3f} "
                    f"+/- {md['auroc_std']:.3f}   confound risk: {verdict}   "
                    f"(AP films: {md['pct_AP_normal']:.0f}% of Normal vs "
                    f"{md['pct_AP_pathology']:.0f}% of {cls})")

    # ------------------------------------------------------------------ #
    # 1: normal versus normal (needs a trained model)
    # ------------------------------------------------------------------ #
    if checkpoint:
        from vae_model import VAE
        ckpt = load_checkpoint(checkpoint, device)
        model = VAE(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        a_loader, b_loader = dm.normal_control_loaders()
        sa = score_loader(model, a_loader, device, cfg)
        sb = score_loader(model, b_loader, device, cfg)

        logger.info("Normal versus normal: AUROC must be near 0.5")
        nvn = {}
        for key in sorted(k for k in sa if not k.startswith("_")):
            m = detection_metrics(sa[key], sb[key], n_boot=200,
                                  seed=cfg["experiment"]["seed"])
            nvn[key] = {"auroc": m["auroc"], "auroc_ci95": m.get("auroc_ci95")}
            flag = "" if abs(m["auroc"] - 0.5) < 0.05 else "   <-- LEAK"
            logger.info(f"  {key:20s} AUROC {m['auroc']:.3f}{flag}")
        results["normal_vs_normal"] = nvn

    path = out_dir / "controls.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Written to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark validity controls")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--checkpoint", type=str, default=None)
    args = ap.parse_args()
    main(args.config, args.checkpoint)