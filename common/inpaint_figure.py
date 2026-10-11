"""
inpaint_figure.py

The four-row panel for the inpainting stage:

    row 1   input x
    row 2   masked input, one representative pass
    row 3   assembled reconstruction, every region taken from the pass in
            which it was HIDDEN
    row 4   residual |x - x_hat| over that assembled reconstruction

    python ../common/inpaint_figure.py --config config.yaml \
        --checkpoint <run>/checkpoints/best.pt --class-name Lung_Opacity --n 5

WHY ROW 3 IS NOT A SINGLE FORWARD PASS

It would be easy, and wrong, to show the reconstruction of the row-2 image.
Two thirds of that output is a region the model could see, so most of the
panel would display ordinary reconstruction and only the masked third would
show inpainting. Row 3 instead assembles the output from all n disjoint
passes, taking each region from the pass that hid it, so every pixel shown
was predicted from context alone. That is also the image the anomaly score
is computed against, which is the point of the figure.

Row 2 is therefore illustrative of the masking, not the input to row 3, and
the caption should say so.

ON SCALING

Rows 1 to 3 share vmin 0 and vmax 1, the range the data occupy, so no row is
stretched to look sharper. The residual row shares one scale across all
columns, set from its 99th percentile and printed to stdout for the caption:
without stating it, a reader cannot tell whether a bright region means a
large error or a small one on a tight colour range.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inpaint import disjoint_mask_sets, inpaint_reconstruct     # noqa: E402
from utils import (build_datamodule, get_device, load_checkpoint,  # noqa: E402
                   load_config, set_seed)

try:
    from vae_model import VAE
except ImportError:
    from model import VAE


def unpack(b):
    if isinstance(b, dict):
        return b["image"], b.get("gt_mask")
    if isinstance(b, (list, tuple)):
        return b[0], None
    return b, None


def get_loader(dm, cls):
    norm = getattr(dm, "normal_loaders", None) or getattr(dm, "get_normal_loaders")
    path = getattr(dm, "pathology_loaders", None) or getattr(dm, "get_pathology_loaders")
    return norm()[2] if cls == "Normal" else path()[cls]


def main() -> None:
    a = argparse.ArgumentParser(description="Inpainting panel")
    a.add_argument("--config", default="config.yaml")
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--class-name", default="Normal")
    a.add_argument("--n", type=int, default=5, help="columns")
    a.add_argument("--skip", type=int, default=0)
    a.add_argument("--out", default="inpaint_panel")
    a.add_argument("--dpi", type=int, default=400)
    args = a.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"].get("device", "cuda"))
    ip = cfg["training"].get("inpaint", {}) or {}
    if not ip.get("enabled", False):
        raise SystemExit("This config has inpaint.enabled false; nothing to show.")
    k, n_pass = ip.get("patch", 16), ip.get("test_n", 3)

    model = VAE(cfg).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, device)["model_state"])
    model.eval()

    loader = get_loader(build_datamodule(cfg), args.class_name)
    xs = []
    for b in loader:
        xs.append(unpack(b)[0])
        if sum(t.shape[0] for t in xs) >= args.skip + args.n:
            break
    X = torch.cat(xs)[args.skip:args.skip + args.n].to(device)

    with torch.no_grad():
        masks = disjoint_mask_sets(X.shape[-1], k, n_pass, device)
        masked = X * masks[0].expand_as(X)            # one pass, for display
        recon = inpaint_reconstruct(model, X, k, n_pass).clamp(0, 1)
        resid = (X - recon).abs()

    g = lambda t: t.detach().cpu().numpy()[:, 0]
    xs_, ms_, rs_, ds_ = g(X), g(masked), g(recon), g(resid)
    vmax_r = float(np.percentile(ds_, 99))
    print(f"residual row scale: 0 to {vmax_r:.4f} (99th percentile)")
    print(f"masking: {k} px patches, {n_pass} disjoint passes, "
          f"{1/n_pass:.0%} of the frame hidden per pass")

    rows = [(xs_, "gray", "$X$"),
            (ms_, "gray", f"masked\n({k} px, 1 of {n_pass})"),
            (rs_, "gray", "inpainted\n$\\hat{X}$"),
            (ds_, "inferno", "$|X-\\hat{X}|$")]
    nr, nc = 4, args.n
    fig, axes = plt.subplots(nr, nc, figsize=(1.5 * nc, 1.5 * nr),
                             gridspec_kw=dict(wspace=0.02, hspace=0.02))
    axes = np.atleast_2d(axes)
    for r, (M, cm, lab) in enumerate(rows):
        for c in range(nc):
            ax = axes[r, c]
            ax.imshow(M[c], cmap=cm, vmin=0, vmax=(vmax_r if r == 3 else 1),
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("0.75")
        axes[r, 0].set_ylabel(lab, fontsize=9.5, rotation=90, va="center", labelpad=8)
    fig.subplots_adjust(left=0.07, right=0.998, top=0.998, bottom=0.004)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
        print(f"written {args.out}.{ext}")
    plt.close(fig)

    print("\nCaption material:")
    print(f"  Reconstruction by inpainting on {args.class_name} images. Row 1, input;")
    print(f"  row 2, one of {n_pass} disjoint masks at {k} px, shown for illustration;")
    print(f"  row 3, the reconstruction assembled from all {n_pass} passes, each region")
    print(f"  taken from the pass in which it was hidden, so no pixel shown was")
    print(f"  visible to the model that produced it; row 4, the residual, on a shared")
    print(f"  scale of 0 to {vmax_r:.4f}.")


if __name__ == "__main__":
    main()