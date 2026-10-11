"""
visualize.py

Qualitative figures, one per class. Rows:

    Input
    Reconstruction
    |Residual|             (the anomaly map)
    Masked                 (what the score actually sees)

The masked row shows exactly what enters the score, so a figure and a number
cannot disagree without it being visible.

The predicted-sigma row was dropped along with the per-pixel uncertainty head
(see base.yaml, revision 1). If predict_sigma is ever turned back on, the
residual row is sigma-normalised again and a sigma row is worth restoring,
since reading the two together is what distinguishes a genuine anomaly from a
region the model always fails on.

RESIDUAL SCALE is shared across classes by default and printed in the
caption. Per-figure normalisation makes a faint normal residual and a strong
pathology residual render identically, so the figure would appear to show a
separation the numbers do not support.

Usage:
    python visualize.py --config config.yaml --checkpoint <path/best.pt>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scoring import pixel_residual, pool_score  # noqa: E402
from utils import (                          # noqa: E402
    build_datamodule, get, get_device, load_checkpoint, load_config, set_seed,
)
from vae_model import VAE                        # noqa: E402

BASE_ROWS = ["Input", "Reconstruction", "|Residual|", "Masked"]
GT_ROW = "Residual + annotation"

# Index of the residual panel within the list returned by gather(). The shared
# colour scale is computed from it, so this must track the row order.
RESIDUAL_IDX = 2


@torch.no_grad()
def gather(model, loader, device, n, use_sigma_norm):
    """Returns [input, reconstruction, residual, masked] and, when the dataset
    supplies annotation boxes, a fifth array of ground-truth masks."""
    xs, ms, gs = [], [], []
    for batch in loader:
        xs.append(batch["image"])
        ms.append(batch["mask"])
        if "gt_mask" in batch:
            gs.append(batch["gt_mask"])
        if sum(t.size(0) for t in xs) >= n:
            break
    x = torch.cat(xs)[:n].to(device)
    m = torch.cat(ms)[:n].to(device)
    mean, log_sigma = model.reconstruct(x, n_samples=1)
    # log_sigma is None when predict_sigma is off; pixel_residual then ignores
    # the normalize flag and returns the raw absolute residual.
    amap = pixel_residual(x, mean, log_sigma, normalize=use_sigma_norm)
    out = [t.cpu().numpy()[:, 0] for t in (x, mean, amap, amap * m)]
    if gs:
        gt = torch.cat(gs)[:n]
        # Only append the row when at least one panel is actually annotated.
        if float(gt.sum()) > 0:
            out.append(gt.cpu().numpy()[:, 0])
    return out


def plot(label, panels, vmax, out_path, note):
    n = panels[0].shape[0]
    has_gt = len(panels) == 5
    rows = BASE_ROWS + ([GT_ROW] if has_gt else [])
    fig, axes = plt.subplots(len(rows), n, figsize=(1.7 * n, 1.75 * len(rows)), squeeze=False)
    styles = [
        dict(cmap="gray", vmin=0, vmax=1),
        dict(cmap="gray", vmin=0, vmax=1),
        dict(cmap="inferno", vmin=0, vmax=vmax),
        dict(cmap="inferno", vmin=0, vmax=vmax),
    ]
    if has_gt:
        # The last row redraws the residual with the annotation outlined, so
        # the model's map and the radiologist's box can be read together.
        styles.append(dict(cmap="inferno", vmin=0, vmax=vmax))

    for r, (data, st) in enumerate(zip(panels[:4] + ([panels[2]] if has_gt else []), styles)):
        for c in range(n):
            ax = axes[r][c]
            ax.imshow(data[c], **st)
            if has_gt and r == 4:
                ax.contour(panels[4][c], levels=[0.5], colors="cyan", linewidths=1.1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r][0].set_ylabel(rows[r], fontsize=10, labelpad=8)
    fig.suptitle(label, fontsize=13, y=0.995)
    fig.text(0.5, 0.010, note, ha="center", fontsize=8, color="0.35")
    fig.subplots_adjust(left=0.065, right=0.995, top=0.95, bottom=0.045,
                        wspace=0.03, hspace=0.03)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(config_path, checkpoint_path, n_samples, percentile):
    cfg = load_config(config_path)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"].get("device", "cuda"))

    ckpt = load_checkpoint(checkpoint_path, device)
    model = VAE(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dm = build_datamodule(cfg)
    _, _, normal_test = dm.normal_loaders()
    loaders = {"Normal (held-out)": normal_test, **dm.pathology_loaders()}
    use_sigma_norm = bool(get(cfg, "scoring.normalize_by_sigma", True))

    collected = {}
    for label, loader in loaders.items():
        collected[label] = gather(model, loader, device, n_samples, use_sigma_norm)

    pooled = np.concatenate([c[RESIDUAL_IDX].ravel() for c in collected.values()])
    vmax = float(np.percentile(pooled, percentile))

    print(f"shared residual vmax (p{percentile:g}) = {vmax:.4f}")
    for label, panels in collected.items():
        res = panels[RESIDUAL_IDX]
        print(f"  {label:24s} residual mean {res.mean():.4f}  "
              f"p{percentile:g} {np.percentile(res, percentile):.4f}")

    run_dir = Path(checkpoint_path).parent.parent
    note = (f"Anomaly map on a shared scale across all classes: vmax = {vmax:.4f} "
            f"({percentile:g}th percentile of pooled maps). "
            f"{'Sigma-normalised.' if use_sigma_norm else 'Raw residual.'}")
    for label, panels in collected.items():
        safe = label.split(" (")[0].replace(" ", "_")
        plot(label, panels, vmax, run_dir / f"qualitative_{safe}.png", note)
        print(f"Saved {run_dir / f'qualitative_{safe}.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Qualitative anomaly figures")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--percentile", type=float, default=99.0)
    args = ap.parse_args()
    main(args.config, args.checkpoint, args.n_samples, args.percentile)