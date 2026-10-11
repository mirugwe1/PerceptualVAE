"""
plot_tensorboard.py

Renders a train.log in the TensorBoard scalar style: one small panel per
metric, the raw series drawn faint behind an exponentially smoothed line,
light grid, step on the x axis.

    python ../common/plot_tensorboard.py --log runs/vae_baseline/<ts>/train.log
    python ../common/plot_tensorboard.py --log a/train.log b/train.log --smooth 0.8

Several logs can be given and are overlaid with one colour each, which is the
comparison TensorBoard gives when several runs share a log directory.

WHY THE SMOOTHING IS SHOWN THIS WAY

TensorBoard's smoothing is an exponential moving average with a debiasing
correction for the first few steps, so an early value is not dragged toward
zero by the initialisation. That correction is reproduced here rather than a
plain EMA, because without it the first ten or twenty epochs of a loss curve
are visibly wrong, which is exactly the region where a run goes bad.

The raw series is always drawn behind at low opacity. A smoothed curve alone
hides the variance, and for these runs the variance between epochs is
informative: it is what distinguishes a model still converging from one whose
validation loss is bouncing around a floor.

WHAT IS PLOTTED

The parser reads the same fields as plot_logs.py, so it works on any train.log
this project has written. PSNR is derived rather than logged: with a per-pixel
mean squared error on data in [0, 1], PSNR = -10 log10(recon), so no rerun is
needed to obtain it.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)"
    r".*?total\s+(?P<total>[-\d.]+)"
    r".*?recon\s+(?P<recon>[-\d.]+)"
    r".*?kl\s+(?P<kl>[-\d.]+)"
    r"(?:.*?kl_per_dim\s+(?P<kl_per_dim>[-\d.]+))?"
    r"(?:.*?beta_t\s+(?P<beta_t>[-\d.]+))?"
    r"(?:.*?share_kl\s+(?P<share_kl>[-\d.]+))?"
    r"(?:.*?perc\s+(?P<perc>[-\d.]+))?"
    r"(?:.*?share_perc\s+(?P<share_perc>[-\d.]+))?"
    r".*?val\s+(?P<val>[-\d.]+)"
    r"(?:.*?lr\s+(?P<lr>[-\d.eE+]+))?",
    re.IGNORECASE)

FIELDS = ["total", "recon", "kl", "kl_per_dim", "beta_t",
          "share_kl", "perc", "share_perc", "val", "lr"]

# Panel definitions: key, title, y label, whether a log y axis suits it.
PANELS = [
    ("total",      "loss/total (train)",      "objective",          False),
    ("val",        "loss/validation",         "objective",          False),
    ("recon",      "loss/reconstruction",     "mean squared error", True),
    ("psnr",       "quality/PSNR (derived)",  "dB",                 False),
    ("kl",         "loss/KL divergence",      "nats per pixel",     False),
    ("kl_per_dim", "latent/KL per dimension", "nats",               False),
    ("share_kl",   "share/KL",                "fraction",           False),
    ("perc",       "loss/perceptual",         "value",              True),
    ("share_perc", "share/perceptual",        "fraction",           False),
    ("gap",        "loss/generalisation gap", "val minus train",    False),
    ("beta_t",     "schedule/beta",           "beta",               False),
    ("lr",         "schedule/learning rate",  "lr",                 True),
]


def parse(path: Path) -> dict:
    """Reads a train.log into arrays, deriving PSNR and the train-val gap."""
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            rows.append({k: (float(m.group(k)) if m.group(k) else np.nan)
                         for k in FIELDS} | {"epoch": int(m.group("epoch"))})
    if not rows:
        raise SystemExit(f"No epoch lines matched in {path}. Check the log format.")

    d = {k: np.array([r[k] for r in rows], dtype=float) for k in FIELDS}
    d["epoch"] = np.array([r["epoch"] for r in rows], dtype=int)
    # PSNR is derivable: a per-pixel MSE on [0, 1] data gives -10 log10(recon).
    # Skipped when the reconstruction term is negative, which means a Gaussian
    # log-likelihood was used and the identity does not hold.
    d["psnr"] = (-10.0 * np.log10(np.clip(d["recon"], 1e-12, None))
                 if np.all(d["recon"] > 0) else np.full_like(d["recon"], np.nan))
    d["gap"] = d["val"] - d["total"]
    d["label"] = Path(path).parent.name
    return d


def ema(y: np.ndarray, weight: float) -> np.ndarray:
    """TensorBoard's smoothing: an exponential moving average with the
    debiasing correction, so early points are not pulled toward zero."""
    out = np.empty_like(y, dtype=float)
    last, debias = 0.0, 0.0
    for i, v in enumerate(y):
        if not np.isfinite(v):
            out[i] = np.nan
            continue
        last = last * weight + (1 - weight) * v
        debias = debias * weight + (1 - weight)
        out[i] = last / debias if debias > 0 else v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="TensorBoard-style training curves")
    ap.add_argument("--log", nargs="+", required=True)
    ap.add_argument("--out", default="training_tensorboard.png")
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="0 shows the raw series, 0.99 is heavy. TensorBoard defaults to 0.6")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    runs = [parse(Path(p)) for p in args.log]
    palette = ["#ff7043", "#42a5f5", "#66bb6a", "#ab47bc", "#ffa726", "#26c6da"]

    live = [p for p in PANELS
            if any(np.isfinite(r[p[0]]).any() and np.nanstd(r[p[0]]) > 0 for r in runs)]
    cols = args.cols
    rows = math.ceil(len(live) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 2.9 * rows),
                             facecolor="white")
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, title, ylab, logy) in zip(axes, live):
        for i, r in enumerate(runs):
            c = palette[i % len(palette)]
            y, e = r[key], r["epoch"]
            if not np.isfinite(y).any():
                continue
            ax.plot(e, y, color=c, lw=1.0, alpha=.22)                 # raw, behind
            ax.plot(e, ema(y, args.smooth), color=c, lw=1.7,
                    label=r["label"] if len(runs) > 1 else None)      # smoothed
        ax.set_title(title, fontsize=9.5, loc="left", pad=6)
        ax.set_xlabel("epoch", fontsize=8); ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(labelsize=7.5)
        if logy and np.all(np.concatenate([r[key][np.isfinite(r[key])] for r in runs]) > 0):
            ax.set_yscale("log")
        ax.set_facecolor("#fafafa")
        ax.grid(color="#e0e0e0", lw=.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bdbdbd")
        if key == "gap":
            ax.axhline(0, color="#9e9e9e", ls="--", lw=.9)
        if len(runs) > 1:
            ax.legend(fontsize=7, frameon=False)

    for ax in axes[len(live):]:
        ax.axis("off")

    sub = ", ".join(r["label"] for r in runs)
    fig.suptitle(f"Training curves: {sub}    (smoothing {args.smooth}, "
                 f"raw series faint behind)", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"written to {args.out}")

    for r in runs:
        best = int(np.nanargmin(r["val"]))
        print(f"\n{r['label']}")
        print(f"  epochs run       : {len(r['epoch'])}")
        print(f"  best val         : {r['val'][best]:.5f} at epoch {r['epoch'][best]}")
        print(f"  final train      : {r['total'][-1]:.5f}")
        print(f"  gap at best      : {r['gap'][best]:+.5f}   at end {r['gap'][-1]:+.5f}")
        if np.isfinite(r["psnr"]).any():
            print(f"  PSNR             : {r['psnr'][0]:.2f} dB -> {r['psnr'][-1]:.2f} dB "
                  f"(best {np.nanmax(r['psnr']):.2f})")
        if np.isfinite(r["share_perc"]).any() and np.nanmax(r["share_perc"]) > 0:
            print(f"  perceptual share : {np.nanmean(r['share_perc']):.1%} mean, "
                  f"{r['share_perc'][-1]:.1%} at the end")
        if np.isfinite(r["kl_per_dim"]).any():
            print(f"  KL per dimension : {r['kl_per_dim'][-1]:.2f} nats at the end")


if __name__ == "__main__":
    main()