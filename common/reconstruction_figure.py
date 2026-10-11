"""
reconstruction_figure.py

Qualitative panels comparing reconstructions from two or more trained
models, with three optional row types that make a small measured difference
visible without exaggerating it.

    # the plain panel: input over one row per model
    python ../common/reconstruction_figure.py --config config.yaml \
        --checkpoints <s1>/best.pt <s7>/best.pt --labels "VAE-pix" "VAE-RaDINO"

    # the version that actually shows the difference
    python ../common/reconstruction_figure.py --config config.yaml \
        --checkpoints <s1>/best.pt <s7>/best.pt --labels "VAE-pix" "VAE-RaDINO" \
        --zoom --residuals --difference --n 4

WHY THE EXTRA ROWS EXIST

On this data the two models differ by 22.4 % in LPIPS and by roughly 1 % in
SSIM and PSNR. A whole 128 x 128 chest printed a few centimetres wide cannot
show a difference of that kind, because the difference lives in
high-frequency texture that the page resolution discards. The honest way to
make it visible is to look where it is, not to alter what is drawn.

  --zoom        crops a square region and draws it at full column width,
                so rib and vascular texture is resolved on the page.
  --residuals   adds |x - x_hat| under each reconstruction. The models
                differ more in their residuals than in their outputs, and
                the residual is what the detector actually scores.
  --difference  adds the signed difference between the first two models on
                a diverging map, which shows exactly where the perceptual
                term changed the output and by how much.

EVERY ROW IS DRAWN HONESTLY

Reconstruction rows share vmin = 0 and vmax = 1, the range the data occupy,
so no row is stretched to look sharper than another. Residual rows share one
scale across all models, computed from their joint 99th percentile, so a
brighter residual means a larger residual rather than a different colour
limit. The difference row is symmetric about zero. All scales used are
printed to stdout and belong in the caption.
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

from utils import (build_datamodule, get_device, load_checkpoint,  # noqa: E402
                   load_config, set_seed)

try:
    from vae_model import VAE
except ImportError:
    from model import VAE


def unpack(b):
    if isinstance(b, dict):
        return b["image"]
    if isinstance(b, (list, tuple)):
        return b[0]
    return b


def get_loader(dm, cls):
    norm = getattr(dm, "normal_loaders", None) or getattr(dm, "get_normal_loaders")
    path = getattr(dm, "pathology_loaders", None) or getattr(dm, "get_pathology_loaders")
    return norm()[2] if cls == "Normal" else path()[cls]


@torch.no_grad()
def reconstruct(ckpt, cfg, x, device, label=""):
    """Build each model from the config stored IN ITS OWN CHECKPOINT.

    Runs made at different times may use different architectures: z_dim 128
    with five encoder blocks in one, z_dim 256 with four in another. Building
    every row from the current config.yaml then fails on shape mismatch, and
    forcing the load with strict=False would be worse than failing, because
    the unmatched layers would stay randomly initialised and the row would
    show output from a partly untrained network rather than from the model
    being compared.

    train.py saves the full config alongside the weights, so the correct
    architecture is always recoverable. The DATA config still comes from the
    current config.yaml, because every row must see identical inputs; if the
    checkpoint was trained at a different resolution that is reported rather
    than silently resized, since it would make the comparison invalid.
    """
    ck = load_checkpoint(ckpt, device)
    mcfg = ck.get("config")
    if mcfg is None:
        print(f"  {label or ckpt}: no config in checkpoint, using config.yaml")
        mcfg = cfg
    else:
        a, b = cfg["data"]["image_size"], mcfg["data"]["image_size"]
        if a != b:
            raise SystemExit(
                f"{label or ckpt} was trained at {b} px but the current config "
                f"is {a} px. Rows would not be comparable; use a checkpoint "
                f"trained at the same resolution.")
        z_now = cfg["model"].get("latent", {}).get("z_dim")
        z_ck = mcfg["model"].get("latent", {}).get("z_dim")
        if z_ck != z_now:
            print(f"  {label}: z_dim {z_ck} (config.yaml says {z_now}); "
                  f"built from the checkpoint's own config")
    model = VAE(mcfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  {label or Path(ckpt).parent.parent.name}: "
          f"z_dim {mcfg['model'].get('latent', {}).get('z_dim')}, {n:,} params")
    xh, _ = model.reconstruct(x)
    del model
    torch.cuda.empty_cache()
    return xh.clamp(0, 1).cpu().numpy()[:, 0]


def main() -> None:
    a = argparse.ArgumentParser(description="Reconstruction comparison panel")
    a.add_argument("--config", default="config.yaml")
    a.add_argument("--checkpoints", nargs="+", required=True)
    a.add_argument("--labels", nargs="+", required=True)
    a.add_argument("--class-name", default="Normal")
    a.add_argument("--n", type=int, default=6)
    a.add_argument("--skip", type=int, default=0,
                   help="Drop the first k cases, to page through candidates "
                        "reproducibly")
    a.add_argument("--spread", action="store_true",
                   help="Pick cases at evenly spaced error quantiles of the "
                        "first model instead of taking the first n")
    a.add_argument("--zoom", action="store_true",
                   help="Crop a square region and draw it at full width")
    a.add_argument("--zoom-box", nargs=4, type=int, default=[30, 34, 56, 56],
                   metavar=("ROW", "COL", "H", "W"),
                   help="Crop in pixels at the model's input resolution")
    a.add_argument("--residuals", action="store_true")
    a.add_argument("--difference", action="store_true")
    a.add_argument("--input-label", default="$X$")
    a.add_argument("--out", default="recon_panel")
    a.add_argument("--dpi", type=int, default=400)
    args = a.parse_args()

    if len(args.checkpoints) != len(args.labels):
        raise SystemExit(f"{len(args.checkpoints)} checkpoints, {len(args.labels)} labels")
    if args.difference and len(args.checkpoints) < 2:
        raise SystemExit("--difference needs at least two checkpoints")

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"].get("device", "cuda"))
    loader = get_loader(build_datamodule(cfg), args.class_name)

    need = args.skip + (args.n * 12 if args.spread else args.n)
    xs = []
    for b in loader:
        xs.append(unpack(b))
        if sum(t.shape[0] for t in xs) >= need:
            break
    X = torch.cat(xs)[args.skip:need].to(device)

    print("models:")
    recs = [reconstruct(ck, cfg, X, device, lbl)
            for ck, lbl in zip(args.checkpoints, args.labels)]
    xnp = X.cpu().numpy()[:, 0]

    if args.spread:
        err = np.abs(xnp - recs[0]).mean((1, 2))
        order = np.argsort(err)
        idx = order[np.linspace(0, len(order) - 1, args.n).astype(int)]
        note = (f"Columns span evenly spaced reconstruction-error quantiles of "
                f"{args.labels[0]}, easiest on the left.")
    else:
        idx = np.arange(args.n)
        note = "Columns are consecutive cases from the held-out split."

    xsel = xnp[idx]
    rsel = [r[idx] for r in recs]

    if args.zoom:
        r0, c0, h, w = args.zoom_box
        crop = lambda A: A[:, r0:r0 + h, c0:c0 + w]
        xsel = crop(xsel); rsel = [crop(r) for r in rsel]
        note += (f" Panels show a {h} x {w} crop at (row {r0}, col {c0}) of the "
                 f"{cfg['data']['image_size']} x {cfg['data']['image_size']} input.")

    # ---- assemble rows ----
    panel, labels, kinds = [xsel], [args.input_label], ["img"]
    for lbl, r in zip(args.labels, rsel):
        panel.append(r); labels.append(lbl); kinds.append("img")
    if args.residuals:
        res = [np.abs(xsel - r) for r in rsel]
        # One scale for every residual row, so brighter really means larger.
        vmax_r = float(np.percentile(np.concatenate(res), 99))
        for lbl, r in zip(args.labels, res):
            panel.append(r); labels.append(f"|$X-\\hat{{X}}$|\n{lbl}"); kinds.append("res")
        print(f"residual rows share vmax = {vmax_r:.4f} (joint 99th percentile)")
    else:
        vmax_r = 1.0
    if args.difference:
        d = rsel[1] - rsel[0]
        vmax_d = float(np.percentile(np.abs(d), 99))
        panel.append(d)
        labels.append(f"{args.labels[1]}\n$-$ {args.labels[0]}")
        kinds.append("diff")
        print(f"difference row symmetric about zero, +/- {vmax_d:.4f}")
    else:
        vmax_d = 1.0

    nr, nc = len(panel), args.n
    fig, axes = plt.subplots(nr, nc, figsize=(1.5 * nc, 1.5 * nr),
                             gridspec_kw=dict(wspace=0.02, hspace=0.02))
    axes = np.atleast_2d(axes)
    for r in range(nr):
        for c in range(nc):
            ax = axes[r, c]
            if kinds[r] == "img":
                ax.imshow(panel[r][c], cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
            elif kinds[r] == "res":
                ax.imshow(panel[r][c], cmap="inferno", vmin=0, vmax=vmax_r,
                          interpolation="nearest")
            else:
                ax.imshow(panel[r][c], cmap="coolwarm", vmin=-vmax_d, vmax=vmax_d,
                          interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("0.75")
        axes[r, 0].set_ylabel(labels[r], fontsize=9.5, rotation=90,
                              va="center", labelpad=8)
    fig.subplots_adjust(left=0.06, right=0.998, top=0.998, bottom=0.004)

    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=args.dpi, bbox_inches="tight",
                    pad_inches=0.01)
        print(f"written {args.out}.{ext}")
    plt.close(fig)

    print("\nCaption material (scales must be stated, not omitted):")
    print(f"  Held-out {args.class_name.lower()} chest radiographs. Row 1 is the "
          f"input; rows 2\u2013{1+len(args.labels)} are reconstructions decoded from")
    print(f"  the posterior mean, all displayed over [0, 1].")
    if args.residuals:
        print(f"  Residual rows share a single scale (0 to {vmax_r:.4f}).")
    if args.difference:
        print(f"  The difference row is symmetric about zero (\u00b1{vmax_d:.4f}).")
    print(f"  {note}")


if __name__ == "__main__":
    main()