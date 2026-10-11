"""
mars_visualize.py  --  three diagnostic figures for stage 9.

    python ../common/mars_visualize.py --config config.yaml --run-dir <run_dir>

  mars_scores.png       Score distributions, normal against each pathology
                        class, for L2 residual, MaRS and raw-feature
                        Mahalanobis side by side. The overlap is the AUROC
                        made visible, and the three panels show directly
                        whether whitening separated what L2 could not.

  mars_spectrum.png     Eigenvalues of the residual covariance, log scale.
                        This is the MECHANISM: MaRS helps only when the
                        residual is anisotropic, a few directions carrying
                        most of the variance. A flat spectrum predicts that
                        L2 and Mahalanobis will agree.

  mars_maps.png         Spatial mode only. Input, reconstruction, pixel
                        residual and MaRS score map for the same cases. The
                        comparison that matters is columns three and four:
                        whether the Mahalanobis map suppresses the rib edges
                        that dominate the pixel residual.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mars import SpatialGaussian, loaders, spatial_embed, unpack   # noqa: E402
from mars_setup import setup                                       # noqa: E402
from utils import build_datamodule, set_seed                       # noqa: E402


def fig_spectrum(run_dir: Path):
    z = np.load(run_dir / "mars_fit.npz")
    if "g_eigvals" not in z.files:
        return None
    ev = np.clip(z["g_eigvals"], 1e-12, None)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ax[0].semilogy(np.arange(1, len(ev) + 1), ev, lw=1.4, color="#2b6cb0")
    ax[0].set_xlabel("eigenvalue rank"); ax[0].set_ylabel("variance (log)")
    ax[0].set_title(f"Residual covariance spectrum\ncondition number {ev[0]/ev[-1]:.1e}",
                    fontsize=9)
    ax[0].grid(alpha=.25)
    cum = np.cumsum(ev) / ev.sum()
    ax[1].plot(np.arange(1, len(ev) + 1), cum, lw=1.6, color="#c05621")
    for f in (0.5, 0.9):
        k = int(np.searchsorted(cum, f)) + 1
        ax[1].axhline(f, color="0.6", ls=":", lw=1)
        ax[1].text(len(ev) * 0.55, f + 0.015, f"{f:.0%} of variance in {k} directions",
                   fontsize=7.5)
    ax[1].set_xscale("log"); ax[1].set_xlabel("number of directions (log)")
    ax[1].set_ylabel("cumulative variance share")
    ax[1].set_title("Anisotropy: steep = Mahalanobis can help\nflat = it reduces to L2",
                    fontsize=9)
    ax[1].grid(alpha=.25)
    fig.tight_layout()
    out = run_dir / "mars_spectrum.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


def fig_scores(run_dir: Path):
    """Reads per-image scores if mars_evaluate saved them; otherwise AUROCs."""
    E = json.loads((run_dir / "eval_results.json").read_text())
    classes = [c for c in E if isinstance(E[c], dict) and "by_score" in E[c]]
    keys = [k for k in ("l2_residual", "mars", "maha_features",
                        "spatial/mean") if k in E[classes[0]]["by_score"]]
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(keys) * len(classes) / 2, 3.4))
    w = 0.8 / len(keys)
    cols = {"l2_residual": "#718096", "mars": "#2b6cb0",
            "maha_features": "#2f855a", "spatial/mean": "#c05621"}
    lab = {"l2_residual": "L2 residual", "mars": "MaRS (residual Maha)",
           "maha_features": "Maha, raw features", "spatial/mean": "spatial MaRS"}
    x = np.arange(len(classes))
    for i, k in enumerate(keys):
        v = [E[c]["by_score"][k]["auroc"] for c in classes]
        lo = [v[j] - E[c]["by_score"][k]["auroc_ci95"][0] for j, c in enumerate(classes)]
        hi = [E[c]["by_score"][k]["auroc_ci95"][1] - v[j] for j, c in enumerate(classes)]
        ax.bar(x + (i - (len(keys) - 1) / 2) * w, v, w, yerr=[lo, hi], capsize=2,
               color=cols[k], label=lab[k], error_kw=dict(lw=.7))
    ax.axhline(0.5, color="crimson", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([c.replace("_", " ")[:22] for c in classes],
                                          fontsize=8)
    ax.set_ylabel("AUROC"); ax.set_ylim(0.4, 1.0); ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    ax.set_title("Same residual, three ways of scoring it", fontsize=9)
    fig.tight_layout()
    out = run_dir / "mars_scores.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


def fig_maps(config, run_dir: Path, cls: str, n: int):
    z = np.load(run_dir / "mars_fit.npz")
    if "sp_mu" not in z.files:
        return None
    cfg, model, extractor, device, _ = setup(config)
    set_seed(cfg["experiment"]["seed"])
    sg = SpatialGaussian.load(z, device)
    dm = build_datamodule(cfg)
    (_, _, test_l), path_l = loaders(dm)
    loader = test_l if cls == "Normal" else path_l[cls]
    xs, gts = [], []
    for b in loader:
        x, gt = unpack(b); xs.append(x)
        if gt is not None: gts.append(gt)
        if sum(t.shape[0] for t in xs) >= n: break
    X = torch.cat(xs)[:n].to(device)
    GT = torch.cat(gts)[:n] if gts else None
    with torch.no_grad():
        xh, _ = model.reconstruct(X); xh = xh.clamp(0, 1)
        sx, _ = spatial_embed(extractor, X, sg.grid)
        sr, _ = spatial_embed(extractor, xh, sg.grid)
        mmap = sg.score_map(sx - sr, X.shape[-1])
    g = lambda t: t.detach().cpu().numpy()[:, 0]
    Xn, Rn, Pn, Mn = g(X), g(xh), g((X - xh).abs()), g(mmap)
    rows = [(Xn, "gray", "$X$", (0, 1)), (Rn, "gray", "$\\hat{X}$", (0, 1)),
            (Pn, "inferno", "$|X-\\hat{X}|$", (0, float(np.percentile(Pn, 99)))),
            (Mn, "inferno", "MaRS map", (float(np.percentile(Mn, 1)),
                                         float(np.percentile(Mn, 99))))]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.5 * n, 1.5 * len(rows)),
                             gridspec_kw=dict(wspace=.02, hspace=.02))
    axes = np.atleast_2d(axes)
    for r, (M, cm, lab, (lo, hi)) in enumerate(rows):
        for c in range(n):
            a = axes[r, c]
            a.imshow(M[c], cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            if GT is not None and r >= 2:
                a.contour(GT[c, 0].numpy() if GT.ndim == 4 else GT[c].numpy(),
                          levels=[0.5], colors="cyan", linewidths=0.8)
            a.set_xticks([]); a.set_yticks([])
        axes[r, 0].set_ylabel(lab, fontsize=9.5, rotation=90, va="center", labelpad=8)
    fig.subplots_adjust(left=.07, right=.998, top=.998, bottom=.004)
    out = run_dir / f"mars_maps_{cls.replace(' ', '_')}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="MaRS figures")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--class-name", default=None)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    rd = Path(args.run_dir)
    for f in (fig_spectrum(rd), fig_scores(rd)):
        if f: print(f"written {f}")
    cls = args.class_name
    if cls is None:
        E = json.loads((rd / "eval_results.json").read_text())
        cls = next((c for c in E if isinstance(E[c], dict) and "localisation" in E[c]),
                   "Normal")
    f = fig_maps(args.config, rd, cls, args.n)
    if f: print(f"written {f}")


if __name__ == "__main__":
    main()
