"""
plot_loss.py

One figure: training and validation loss on the same axes, TensorBoard style.

    python ../common/plot_loss.py --log runs/vae_baseline/<ts>/train.log
    python ../common/plot_loss.py --log a/train.log b/train.log --smooth 0.8

Several logs overlay, one colour per run, training solid and validation
dashed.

The raw series is drawn faint behind an exponentially smoothed line, using
TensorBoard's own smoothing: an exponential moving average with the debiasing
correction, so the first few epochs are not dragged toward zero. Without that
correction the early part of a loss curve is visibly wrong, which is where a
bad run shows itself.

The best validation epoch is marked, since that is the checkpoint the rest of
the pipeline evaluates, and the gap between the two curves at that point is
annotated. That gap is the quantity to watch: for an anomaly detector,
overfitting is not merely a loss of accuracy but a loss of the mechanism, as a
model that fits its training normals too closely reconstructs pathology about
as well as health and the residual stops separating anything.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+).*?total\s+(?P<total>[-\d.]+).*?val\s+(?P<val>[-\d.]+)",
    re.IGNORECASE)


def parse(path: Path):
    e, tr, va = [], [], []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            e.append(int(m.group("epoch")))
            tr.append(float(m.group("total")))
            va.append(float(m.group("val")))
    if not e:
        raise SystemExit(f"No epoch lines matched in {path}")
    return (np.array(e), np.array(tr), np.array(va), Path(path).parent.name)


def ema(y, weight: float):
    """TensorBoard smoothing, with the debiasing correction."""
    out = np.empty_like(y, dtype=float)
    last = debias = 0.0
    for i, v in enumerate(y):
        last = last * weight + (1 - weight) * v
        debias = debias * weight + (1 - weight)
        out[i] = last / debias if debias > 0 else v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Training and validation loss")
    ap.add_argument("--log", nargs="+", required=True)
    ap.add_argument("--out", default="loss_curve.png")
    ap.add_argument("--smooth", type=float, default=0.6)
    ap.add_argument("--logy", action="store_true", help="Logarithmic y axis")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    runs = [parse(Path(p)) for p in args.log]
    palette = ["#ff7043", "#42a5f5", "#66bb6a", "#ab47bc", "#ffa726", "#26c6da"]

    fig, ax = plt.subplots(figsize=(9.2, 5.4), facecolor="white")
    for i, (e, tr, va, lab) in enumerate(runs):
        c = palette[i % len(palette)]
        one = len(runs) == 1
        ax.plot(e, tr, color=c, lw=1.0, alpha=.20)
        ax.plot(e, va, color=c, lw=1.0, alpha=.20)
        ax.plot(e, ema(tr, args.smooth), color=c, lw=2.0,
                label="training" if one else f"{lab}, training")
        ax.plot(e, ema(va, args.smooth), color=c, lw=2.0, ls="--",
                label="validation" if one else f"{lab}, validation")

        b = int(np.argmin(va))
        ax.plot(e[b], va[b], "o", color=c, ms=8, mfc="white", mew=2, zorder=5)
        ax.annotate(f"best val {va[b]:.5f}\nepoch {e[b]}, gap {va[b]-tr[b]:+.5f}",
                    (e[b], va[b]), textcoords="offset points", xytext=(12, 14),
                    fontsize=8.5, color=c,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=c, lw=.8, alpha=.9))

    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylabel("objective", fontsize=10)
    if args.logy:
        ax.set_yscale("log")
    ax.set_facecolor("#fafafa")
    ax.grid(color="#e0e0e0", lw=.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbdbd")
    ax.legend(fontsize=9, frameon=False)
    title = args.title or (f"Training and validation loss: {runs[0][3]}"
                           if len(runs) == 1 else "Training and validation loss")
    ax.set_title(f"{title}\nsmoothing {args.smooth}, raw series faint behind",
                 fontsize=11, loc="left", pad=10)

    fig.tight_layout()
    fig.savefig(args.out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"written to {args.out}\n")

    for e, tr, va, lab in runs:
        b = int(np.argmin(va))
        print(f"{lab}")
        print(f"  epochs run   : {len(e)}")
        print(f"  best val     : {va[b]:.6f} at epoch {e[b]}")
        print(f"  train there  : {tr[b]:.6f}   gap {va[b]-tr[b]:+.6f}")
        print(f"  final        : train {tr[-1]:.6f}, val {va[-1]:.6f}, "
              f"gap {va[-1]-tr[-1]:+.6f}")
        growth = (va[-1] - tr[-1]) - (va[b] - tr[b])
        print(f"  gap growth after the best epoch: {growth:+.6f}"
              f"{'   (overfitting)' if growth > 0 else ''}\n")


if __name__ == "__main__":
    main()