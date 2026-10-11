"""
plot_logs.py

Parses one or more cxr_ad train.log files and renders a diagnostic panel grid,
plus a printed summary of the things that are easy to miss by scrolling.

    # newest run in this stage
    python ../common/plot_logs.py

    # a specific run
    python ../common/plot_logs.py \
  --log stage3_vgg19_cxr/runs/vae_vgg19_cxr/20260913_165820/train.log \
  --out ./training_curves.png



    # compare stages side by side on the same axes
    python ../common/plot_logs.py \
        --log ../stage1_vae/runs/vae_baseline/*/train.log \
        --log ../stage2_vgg19/runs/vae_vgg19_imagenet/*/train.log \
        --labels stage1 stage2

Writes training_curves.png next to the first log (or --out), and metrics.csv
alongside it so the numbers can go straight into a table.

WHAT THE PANELS ARE FOR

  Total, train vs val   The overfitting check. Train falling while val flattens
                        is the failure that matters most for anomaly detection,
                        because a VAE that generalises too well reconstructs
                        pathology just as happily as health and the residual
                        stops separating anything.

  Generalisation gap    The same thing as one number, which makes the epoch it
                        opens up easy to read off.

  KL per dimension      Against the free-bits floor. Sitting on the floor means
                        dimensions are dead and the latent carries little. Far
                        above it with a tiny KL share means the bottleneck is
                        barely constraining anything.

  Term shares           What fraction of the objective each term actually
                        contributes. A perceptual term at under one percent is
                        a variant that is secretly the baseline.

  Epoch duration        Mostly for planning walltime, but a sudden change also
                        flags a node problem or thermal throttling.
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402


EPOCH_RE = re.compile(
    r"(?P<ts>\d{2}:\d{2}:\d{2})\s*\|\s*INFO\s*\|\s*epoch\s+(?P<epoch>\d+)\s*\|\s*"
    r"total\s+(?P<total>[-\d.eE+]+)\s+"
    r"recon\s+(?P<recon>[-\d.eE+]+)\s+"
    r"kl\s+(?P<kl>[-\d.eE+]+)\s*\("
    r"per-dim\s+(?P<kl_per_dim>[-\d.eE+]+),\s*"
    r"beta_t\s+(?P<beta_t>[-\d.eE+]+),\s*"
    r"share\s+(?P<share_kl>[-\d.]+)%\)\s*"
    r"perc\s+(?P<perc>[-\d.eE+]+)\s*\(share\s+(?P<share_perc>[-\d.]+)%\)\s*\|\s*"
    r"val\s+(?P<val>[-\d.eE+]+)\s*\|\s*lr\s+(?P<lr>[-\d.eE+]+)"
)

HEADERS = {
    "run_dir":    re.compile(r"Run directory:\s*(.+)"),
    "device":     re.compile(r"Device:\s*(.+)"),
    "counts":     re.compile(r"Class counts:\s*(\{.+\})"),
    "split":      re.compile(r"Normal split:\s*train\s+(\d+),\s*val\s+(\d+),\s*test\s+(\d+)"),
    "params":     re.compile(r"VAE parameters:\s*([\d,]+)\s*\((.+)\)"),
    "backbone":   re.compile(r"Perceptual backbone\s+(\S+):\s*([\d,]+) frozen"),
    "early_stop": re.compile(r"Early stopping at epoch\s+(\d+)\s*\(best val\s+([-\d.]+)\)"),
    "test":       re.compile(r"Held-out normal test objective:\s*([-\d.]+)"),
}

NUMERIC = ["total", "recon", "kl", "kl_per_dim", "beta_t",
           "share_kl", "perc", "share_perc", "val", "lr"]


def parse(path: Path) -> dict:
    meta: dict = {"path": str(path), "label": path.parent.name}
    epochs: list[dict] = []
    times: list[datetime] = []

    for line in path.read_text(errors="replace").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            d = m.groupdict()
            row = {"epoch": int(d["epoch"])}
            for k in NUMERIC:
                row[k] = float(d[k])
            epochs.append(row)
            times.append(datetime.strptime(d["ts"], "%H:%M:%S"))
            continue
        for key, rx in HEADERS.items():
            mm = rx.search(line)
            if mm:
                meta[key] = mm.groups() if len(mm.groups()) > 1 else mm.group(1)

    if not epochs:
        raise ValueError(f"No epoch lines matched in {path}. Is this a train.log?")

    data = {k: np.array([e[k] for e in epochs], dtype=float)
            for k in ["epoch"] + NUMERIC}

    # Epoch wall time, tolerating a run that crosses midnight.
    secs = []
    for a, b in zip(times[:-1], times[1:]):
        dt = (b - a).total_seconds()
        secs.append(dt + 86400 if dt < 0 else dt)
    data["epoch_seconds"] = np.array([secs[0] if secs else np.nan] + secs)

    return {"meta": meta, "data": data}


def summarise(run: dict) -> list[str]:
    d, m = run["data"], run["meta"]
    n = len(d["epoch"])
    best_i = int(np.argmin(d["val"]))
    gap = d["val"] - d["total"]

    lines = [f"  epochs run            : {n}"]
    if "params" in m:
        lines.append(f"  model                 : {m['params'][0]} params ({m['params'][1]})")
    if "split" in m:
        tr, va, te = m["split"]
        lines.append(f"  normal split          : train {tr} / val {va} / test {te}")
    lines.append(f"  best val              : {d['val'][best_i]:.5f} at epoch {int(d['epoch'][best_i])}")
    lines.append(f"  final train / val     : {d['total'][-1]:.5f} / {d['val'][-1]:.5f}")
    lines.append(f"  generalisation gap    : {gap[best_i]:+.4f} at best, {gap[-1]:+.4f} at end")
    lines.append(f"  KL share of objective : {d['share_kl'][-1]:.1f}%  (min {d['share_kl'].min():.1f}, max {d['share_kl'].max():.1f})")
    lines.append(f"  KL per dim            : {d['kl_per_dim'][0]:.3f} -> {d['kl_per_dim'][-1]:.3f}")
    if d["perc"].max() > 0:
        lines.append(f"  perceptual share      : {d['share_perc'][-1]:.1f}%  (max {d['share_perc'].max():.1f})")
    else:
        lines.append("  perceptual            : not used")
    tot = np.nansum(d["epoch_seconds"])
    lines.append(f"  time                  : {np.nanmedian(d['epoch_seconds']):.1f}s/epoch, "
                 f"{tot/60:.1f} min total")
    if "early_stop" in m:
        lines.append(f"  early stopped         : epoch {m['early_stop'][0]} (best val {m['early_stop'][1]})")
    if "test" in m:
        lines.append(f"  held-out test         : {m['test']}")

    # --- flags -------------------------------------------------------- #
    flags = []
    wasted = n - (best_i + 1)
    if wasted > 0:
        flags.append(f"{wasted} epochs after the best validation score "
                     f"({100*wasted/n:.0f}% of the run)")
    if gap[-1] > 3 * abs(gap[best_i]) and gap[-1] > 0.05:
        flags.append(f"generalisation gap grew from {gap[best_i]:+.3f} to {gap[-1]:+.3f}: "
                     f"the model is fitting the training normals rather than learning "
                     f"a general model of health, which is the failure mode that makes "
                     f"reconstruction stop separating pathology")
    if d["share_kl"].max() < 5.0:
        flags.append(f"KL never exceeds {d['share_kl'].max():.1f}% of the objective: "
                     f"the bottleneck is barely constraining the model. Consider "
                     f"raising loss.beta")
    if d["perc"].max() > 0 and d["share_perc"].max() < 1.0:
        flags.append("perceptual share stayed under 1%: this run is effectively "
                     "the no-perceptual baseline")
    lr_drop = d["lr"][0] / max(d["lr"][-1], 1e-12)
    if lr_drop < 2:
        flags.append(f"learning rate only fell {lr_drop:.2f}x: the cosine schedule "
                     f"was set for more epochs than were run, so the run stopped "
                     f"mid-schedule at a still-high learning rate")

    return lines, flags


def plot(runs: list[dict], out_path: Path, free_bits: float) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 11.5))
    ax = axes.ravel()

    def style(a, title, ylabel, xlabel="epoch"):
        a.set_title(title, fontsize=11, pad=6)
        a.set_ylabel(ylabel, fontsize=9)
        a.set_xlabel(xlabel, fontsize=9)
        a.grid(alpha=0.25, linewidth=0.6)
        a.tick_params(labelsize=8)

    for i, run in enumerate(runs):
        d, lab, c = run["data"], run["meta"]["label"], colors[i % 10]
        e = d["epoch"]
        best = int(np.argmin(d["val"]))

        # 1 objective
        ax[0].plot(e, d["total"], color=c, lw=1.4, label=f"{lab} train")
        ax[0].plot(e, d["val"], color=c, lw=1.2, ls="--", alpha=0.85, label=f"{lab} val")
        ax[0].scatter([e[best]], [d["val"][best]], color=c, s=45, zorder=5,
                      marker="v", edgecolor="black", linewidth=0.5)

        # 2 gap
        ax[1].plot(e, d["val"] - d["total"], color=c, lw=1.4, label=lab)

        # 3 reconstruction
        ax[2].plot(e, d["recon"], color=c, lw=1.4, label=lab)

        # 4 KL (weighted contribution)
        ax[3].plot(e, d["kl"], color=c, lw=1.4, label=f"{lab} kl")
        ax[3].plot(e, d["beta_t"] * d["kl"], color=c, lw=1.0, ls=":",
                   alpha=0.8, label=f"{lab} beta*kl")

        # 5 KL per dim
        ax[4].plot(e, d["kl_per_dim"], color=c, lw=1.4, label=lab)

        # 6 shares
        ax[5].plot(e, d["share_kl"], color=c, lw=1.4, label=f"{lab} KL")
        if d["perc"].max() > 0:
            ax[5].plot(e, d["share_perc"], color=c, lw=1.4, ls="--",
                       label=f"{lab} perceptual")

        # 7 lr and beta schedule
        ax[6].plot(e, d["lr"], color=c, lw=1.4, label=f"{lab} lr")

        # 8 perceptual term
        if d["perc"].max() > 0:
            ax[7].plot(e, d["perc"], color=c, lw=1.4, label=lab)

        # 9 epoch duration
        ax[8].plot(e, d["epoch_seconds"], color=c, lw=1.0, alpha=0.8, label=lab)

    style(ax[0], "Objective: train vs validation", "total loss")
    ax[0].legend(fontsize=7, ncol=2)

    style(ax[1], "Generalisation gap (val minus train)", "gap")
    ax[1].axhline(0, color="0.4", lw=0.8, ls="--")
    ax[1].legend(fontsize=7)

    style(ax[2], "Reconstruction term", "recon")
    ax[2].legend(fontsize=7)

    style(ax[3], "KL term (raw and beta-weighted)", "kl")
    ax[3].legend(fontsize=7, ncol=2)

    style(ax[4], f"KL per latent dimension (free-bits floor = {free_bits})", "nats/dim")
    ax[4].axhline(free_bits, color="crimson", lw=1.1, ls="--", alpha=0.8)
    ax[4].text(0.99, free_bits, " floor", color="crimson", fontsize=8,
               va="bottom", ha="right", transform=ax[4].get_yaxis_transform())
    ax[4].legend(fontsize=7)

    style(ax[5], "Share of the objective", "percent")
    ax[5].axhline(1.0, color="crimson", lw=1.0, ls="--", alpha=0.7)
    ax[5].text(0.99, 1.0, " 1% floor for a term to matter", color="crimson",
               fontsize=7.5, va="bottom", ha="right",
               transform=ax[5].get_yaxis_transform())
    ax[5].legend(fontsize=7)

    style(ax[6], "Learning rate", "lr")
    ax[6].set_yscale("log")
    ax[6].legend(fontsize=7)

    style(ax[7], "Perceptual term", "perceptual")
    if not ax[7].lines:
        ax[7].text(0.5, 0.5, "no perceptual term in these runs", ha="center",
                   va="center", fontsize=10, color="0.5", transform=ax[7].transAxes)
    else:
        ax[7].legend(fontsize=7)

    style(ax[8], "Epoch wall time", "seconds")
    ax[8].legend(fontsize=7)

    title = "  |  ".join(r["meta"]["label"] for r in runs)
    fig.suptitle(f"Training diagnostics: {title}", fontsize=13, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_csv(runs: list[dict], path: Path) -> None:
    cols = ["run", "epoch"] + NUMERIC + ["epoch_seconds"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            d, lab = r["data"], r["meta"]["label"]
            for i in range(len(d["epoch"])):
                w.writerow([lab] + [d[c][i] for c in cols[1:]])


def discover() -> list[Path]:
    found = sorted(Path("runs").glob("*/*/train.log"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return found[:1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot cxr_ad training logs")
    ap.add_argument("--log", action="append", nargs="+", default=None,
                    help="train.log path(s). Repeat the flag to compare runs. "
                         "Globs are fine. Defaults to the newest run under ./runs/")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Legend label per log, in the same order")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG. Defaults to training_curves.png beside the first log")
    ap.add_argument("--free-bits", type=float, default=0.5,
                    help="Drawn as a reference line on the KL-per-dimension panel")
    args = ap.parse_args()

    if args.log:
        paths = [Path(p) for group in args.log for p in group]
    else:
        paths = discover()
        if not paths:
            raise SystemExit("No train.log found under ./runs/*/*/. Pass --log explicitly.")

    runs = []
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"Not a file: {p}")
        runs.append(parse(p))

    if args.labels:
        if len(args.labels) != len(runs):
            raise SystemExit(f"{len(args.labels)} labels for {len(runs)} logs")
        for r, lab in zip(runs, args.labels):
            r["meta"]["label"] = lab

    print("=" * 74)
    for r in runs:
        lines, flags = summarise(r)
        print(f"\n{r['meta']['label']}  ({r['meta']['path']})")
        print("-" * 74)
        for ln in lines:
            print(ln)
        if flags:
            print("\n  WORTH LOOKING AT:")
            for fl in flags:
                print(f"    * {fl}")
    print("\n" + "=" * 74)

    out = Path(args.out) if args.out else paths[0].parent / "training_curves.png"
    plot(runs, out, args.free_bits)
    csv_path = out.with_name("metrics.csv")
    write_csv(runs, csv_path)
    print(f"Figure : {out}")
    print(f"CSV    : {csv_path}")


if __name__ == "__main__":
    main()