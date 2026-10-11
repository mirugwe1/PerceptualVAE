"""
loc_visualize.py  --  figures for a stage-10 run.

    python ../common/loc_visualize.py --config config.yaml --run-dir <run_dir>

  maps_<variant>.png      Example cases. Rows: input, reconstruction, then one
                          row per scoring map. Radiologist boxes overlaid in
                          cyan on every map row, so you can judge by eye
                          whether a map lights up inside the box or merely
                          along the ribs and borders.
  localisation.png        Pixel AUPRC, pixel AUROC and IoU per map and variant
  detection.png           Image AUROC per map and class

DISPLAY SCALING

Each map row uses ONE colour scale across all its columns, set from that
map's 1st and 99th percentiles over the cases shown. So brightness compares
fairly across cases within a row. Rows are on different scales, which is
unavoidable since the maps measure different things; compare where each
map is bright, not how bright two different maps are.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loc_maps as LM                                              # noqa: E402
from loc_evaluate import (Context, apply_variant, loaders, resolve,  # noqa: E402
                          unpack)
from utils import build_datamodule, get_device, load_checkpoint, load_config  # noqa: E402

PRETTY = {"pixel": "|x - x̂|", "normative": "normative", "prior_sub": "prior sub.",
          "ssim": "SSIM", "feat_multiscale": "feature resid.", "kl_grad": "KL grad",
          "elbo_grad": "ELBO grad", "sample_var": "sample var.",
          "ensemble_var": "ensemble var.", "mars_spatial": "spatial MaRS",
          "fusion": "fusion", "uniform": "uniform (control)",
          "proj_grad": "projection", "proj_ssim": "projection SSIM",
          "restore": "restoration", "restore_ssim": "restoration SSIM",
          "latent_trunc": "latent trunc."}


def build_cfg(config_path, device):
    stage = load_config(config_path)
    src = resolve(stage["loc"]["source_checkpoint"])
    cfg = copy.deepcopy(load_checkpoint(src, device)["config"])
    cfg["loc"] = stage["loc"]
    cfg.setdefault("paths", {}).update(stage.get("paths", {}))
    return cfg


class _Quiet:
    def info(self, *a, **k): pass
    def warning(self, m, *a, **k): print("warning:", m)


def fig_maps(cfg, device, run_dir, variant, n, skip, methods):
    C = Context(cfg, device, _Quiet())
    dm = build_datamodule(cfg)
    (_, val_l, _), path_l = loaders(dm)
    cls = next((c for c in path_l if "opacity" in c.lower() and "not" not in c.lower()),
               next(iter(path_l)))

    # per-pixel healthy statistics, needed by normative and prior_sub
    s1 = s2 = None; k = 0
    for b in val_l:
        x = unpack(b)[0].to(device)
        r = C.maps(x, {"pixel"})["pixel"]
        s1 = r.sum(0) if s1 is None else s1 + r.sum(0)
        s2 = (r ** 2).sum(0) if s2 is None else s2 + (r ** 2).sum(0)
        k += x.shape[0]
    C.pix_mu = s1 / k
    C.pix_sd = (s2 / k - C.pix_mu ** 2).clamp(min=0).sqrt()

    # first n cases that actually carry a box, after skipping `skip` of them
    X, G, seen = [], [], 0
    for b in path_l[cls]:
        x, gt = unpack(b)
        if gt is None:
            continue
        gt = gt[:, 0] if gt.ndim == 4 else gt
        for i in range(x.shape[0]):
            if gt[i].any():
                if seen >= skip:
                    X.append(x[i:i + 1]); G.append(gt[i])
                seen += 1
            if len(X) >= n:
                break
        if len(X) >= n:
            break
    if not X:
        print("no annotated cases found; skipping map figure")
        return None
    x = torch.cat(X).to(device)
    G = torch.stack(G).numpy()

    want = [m for m in methods if m in C.methods] if methods else list(C.methods)
    mp = C.maps(x, set(want))
    fusion = [m for m in C.L.get("fusion", []) if m in mp]
    if fusion:
        z = [(mp[m] - mp[m].mean()) / (mp[m].std() + 1e-8) for m in fusion]
        mp["fusion"] = torch.stack(z).mean(0)
        want = want + ["fusion"]
    lung = C.lung(x) if C.lung is not None and "lung" in variant else None
    with torch.no_grad():
        xh = LM.recon(C.model, x)

    rows = [("input", x[:, 0].cpu().numpy(), "gray", (0, 1)),
            ("recon. x̂", xh[:, 0].cpu().numpy(), "gray", (0, 1))]
    for m in want:
        if m not in mp:
            continue
        t = apply_variant(mp[m], lung, variant, float(C.L.get("smooth_sigma", 2.0)))
        if t is None:
            continue
        a = t[:, 0].cpu().numpy()
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        rows.append((PRETTY.get(m, m), a, "inferno", (lo, hi)))

    nr, nc = len(rows), len(X)
    fig, axes = plt.subplots(nr, nc, figsize=(1.55 * nc, 1.55 * nr),
                             gridspec_kw=dict(wspace=.03, hspace=.03))
    axes = np.atleast_2d(axes)
    for r, (lab, A, cm, (lo, hi)) in enumerate(rows):
        for c in range(nc):
            ax = axes[r, c]
            ax.imshow(A[c], cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            if r >= 2:
                ax.contour(G[c].astype(float), levels=[0.5], colors="cyan", linewidths=0.9)
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(lab, fontsize=8.5, rotation=90, va="center", labelpad=6)
    fig.suptitle(f"RSNA {cls}, variant: {variant}", fontsize=9, y=1.0)
    fig.subplots_adjust(left=.08, right=.998, top=.985, bottom=.004)
    out = run_dir / f"maps_{variant.replace('+', '_')}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    return out


def read_csv(p):
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def fig_localisation(run_dir):
    rows = read_csv(run_dir / "localisation.csv")
    if not rows:
        print("no localisation.csv; skipping localisation chart")
        return None
    methods = list(dict.fromkeys(r["method"] for r in rows))
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    cols = {"raw": "#718096", "smooth": "#2b6cb0", "lung": "#2f855a", "lung+smooth": "#c05621"}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.6))
    w = 0.8 / len(variants)
    x = np.arange(len(methods))
    for ax, key, title in zip(axes, ("pixel_auprc", "pixel_auroc", "iou"),
                              ("pixel AUPRC", "pixel AUROC", "IoU")):
        for i, v in enumerate(variants):
            vals = [next((float(r[key]) for r in rows
                          if r["method"] == m and r["variant"] == v), np.nan)
                    for m in methods]
            ax.bar(x + (i - (len(variants) - 1) / 2) * w, vals, w,
                   color=cols.get(v, "0.5"), label=v)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY.get(m, m) for m in methods], rotation=40,
                           ha="right", fontsize=7.5)
        ax.set_title(title, fontsize=9.5); ax.grid(axis="y", alpha=.25)
        vv = [float(r[key]) for r in rows]
        ax.set_ylim(min(vv) * 0.97, max(vv) * 1.02)
    axes[0].legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    out = run_dir / "localisation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    return out


def fig_detection(run_dir, pool="mean"):
    rows = [r for r in read_csv(run_dir / "detection.csv") if r["pool"] == pool]
    if not rows:
        print("no detection.csv; skipping detection chart")
        return None
    classes = list(dict.fromkeys(r["cls"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, axes = plt.subplots(1, len(classes), figsize=(5.2 * len(classes), 3.6),
                             squeeze=False)
    for ax, c in zip(axes[0], classes):
        for i, v in enumerate(("raw", "lung+smooth")):
            vals = [next((float(r["auroc"]) for r in rows if r["cls"] == c and
                          r["method"] == m and r["variant"] == v), np.nan) for m in methods]
            ax.bar(np.arange(len(methods)) + (i - 0.5) * 0.38, vals, 0.38,
                   color=("#2b6cb0", "#c05621")[i], label=v)
        ax.axhline(0.5, color="crimson", ls="--", lw=1)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels([PRETTY.get(m, m) for m in methods], rotation=40,
                           ha="right", fontsize=7.5)
        ax.set_ylim(0.3, 1.0); ax.set_ylabel("image AUROC"); ax.grid(axis="y", alpha=.25)
        ax.set_title(f"{c}  (pool: {pool})", fontsize=9.5)
    axes[0, 0].legend(fontsize=7.5)
    fig.tight_layout()
    out = run_dir / "detection.png"
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 10 figures")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variants", nargs="+", default=["raw", "lung+smooth"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--skip", type=int, default=0,
                    help="Skip the first k annotated cases, to page through examples")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Subset of maps to draw; default all computed")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    device = get_device("cuda")
    for f in (fig_localisation(run_dir), fig_detection(run_dir)):
        if f:
            print(f"written {f}")
    cfg = build_cfg(args.config, device)
    for v in args.variants:
        f = fig_maps(cfg, device, run_dir, v, args.n, args.skip, args.methods)
        if f:
            print(f"written {f}")


if __name__ == "__main__":
    main()
