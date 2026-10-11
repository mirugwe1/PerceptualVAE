"""
padim_baseline.py

A non-generative reference point: PaDiM (Defard et al., 2021) built on the
same frozen backbones, the same splits and the same metric code as the VAE
variants, so the only thing that differs is the method.

    python ../common/padim_baseline.py --config config.yaml \
        --backbone resnet50_imagenet --out padim_resnet50.json

WHY THIS BASELINE AND WHY NOW

Nothing in the VAE results tells a reader whether 0.719 image-level AUROC or
0.128 pixel AUPRC is good for these benchmarks. Both numbers are reported
against chance, but chance is a floor, not a competitive reference, and a
reviewer will reasonably ask whether a different family of method does
better on the same data. PaDiM answers that in one evaluation pass.

It is also the right contrast. The VAE's claim is that reconstruction error
localises pathology; PaDiM makes no reconstruction at all, and instead asks
whether a patch's FEATURES are unusual for that position. If PaDiM localises
substantially better from the same backbone, the limitation is the residual
rather than the features available to it, which is a sharper conclusion than
either result alone.

METHOD

For each image, features are taken from several layers of a frozen backbone,
bilinearly resampled to a common grid and concatenated, giving an embedding
of dimension D at each of P grid positions. A random but fixed subset of d of
those D channels is retained, which is the dimension reduction used in the
original work and which matters here because a full covariance over D would
be both singular and large.

A Gaussian is then fitted INDEPENDENTLY AT EACH POSITION over the training
normals, N(mu_p, Sigma_p), and the anomaly score at that position is the
Mahalanobis distance

    M(f_p) = sqrt( (f_p - mu_p)^T  Sigma_p^-1  (f_p - mu_p) )

Position-wise rather than global is the point: a radiograph is spatially
registered enough that what is normal at the apex is not what is normal at
the diaphragm, and a single global Gaussian would discard that.

TWO IMPLEMENTATION NOTES

Covariance is accumulated in STREAMING form, holding only the running sum
and the running sum of outer products, because materialising every
embedding would need several gigabytes at this dataset size. The estimator
is the textbook one, Sigma = (S2 - n mu mu^T)/(n-1), with a ridge term
eps*I added so that every Sigma_p is invertible even where the features are
nearly collinear.

Estimating a d x d covariance needs n > d per position; with d = 100 and
several thousand training normals this is comfortable, but the check is made
explicitly rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import detection_metrics, pixel_localisation          # noqa: E402
from perceptual import build_extractor                             # noqa: E402
from utils import build_datamodule, get_device, get_logger, load_config, set_seed  # noqa: E402


# --------------------------------------------------------------------------- #

def unpack(batch):
    """(image, mask, gt_mask) whatever shape the batch arrives in.

    The two datamodules disagree: RSNA yields a dict with image, mask and
    gt_mask, the COVID loader yields a bare tensor or an (image, mask) tuple.
    """
    if isinstance(batch, dict):
        x = batch["image"]
        m = batch.get("mask")
        return x, (m if m is not None else torch.ones_like(x)), batch.get("gt_mask")
    if isinstance(batch, (list, tuple)):
        x = batch[0]
        m = (batch[1] if len(batch) > 1 and torch.is_tensor(batch[1])
             and batch[1].shape == x.shape else torch.ones_like(x))
        return x, m, None
    return batch, torch.ones_like(batch), None


def loaders(dm):
    n = getattr(dm, "normal_loaders", None) or getattr(dm, "get_normal_loaders")
    p = getattr(dm, "pathology_loaders", None) or getattr(dm, "get_pathology_loaders")
    return n(), p()


@torch.no_grad()
def embed(extractor, x, idx, grid):
    """Concatenated, channel-subsampled embedding at a common grid.

    Every layer is resampled to the FIRST layer's resolution rather than to a
    fixed size, so the grid follows the backbone instead of a magic number.
    """
    feats = extractor(x)
    g = grid or feats[0].shape[-1]
    z = torch.cat([f if f.shape[-1] == g else
                   Fn.interpolate(f, size=(g, g), mode="bilinear", align_corners=False)
                   for f in feats], dim=1)
    return z[:, idx], g


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="PaDiM baseline")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--backbone", default="resnet50_imagenet")
    ap.add_argument("--layers", nargs="*", default=None,
                    help="Defaults to the backbone's own first three stages")
    ap.add_argument("--dim", type=int, default=100,
                    help="Channels retained after random subsampling")
    ap.add_argument("--eps", type=float, default=0.01, help="Ridge added to each covariance")
    ap.add_argument("--out", default="padim_baseline.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"].get("device", "cuda"))
    logger = get_logger("padim", Path(args.out).with_suffix(".log"))

    layers = args.layers or (["layer1", "layer2", "layer3"]
                             if "resnet" in args.backbone else
                             ["relu1_2", "relu2_2", "relu3_4"])
    extractor = build_extractor(args.backbone, layers=layers).to(device).eval()
    for p in extractor.parameters():
        p.requires_grad_(False)

    dm = build_datamodule(cfg)
    (train_loader, _, test_loader), path_loaders = loaders(dm)
    use_mask = cfg["scoring"].get("use_lung_mask", False)

    # ---- pass 1: streaming mean and second moment, per position ----
    idx = None; S1 = S2 = None; n = 0; grid = None
    for b in train_loader:
        x, _, _ = unpack(b)
        x = x.to(device, non_blocking=True)
        if idx is None:
            probe = extractor(x[:1])
            D = sum(f.shape[1] for f in probe)
            d = min(args.dim, D)
            idx = torch.randperm(D, generator=torch.Generator().manual_seed(
                cfg["experiment"]["seed"]))[:d].to(device)
            logger.info(f"backbone {args.backbone}, layers {layers}: "
                        f"{D} channels -> {d} retained")
        z, grid = embed(extractor, x, idx, grid)          # (B, d, g, g)
        B, d, g, _ = z.shape
        z = z.permute(0, 2, 3, 1).reshape(B, g * g, d)    # (B, P, d)
        if S1 is None:
            S1 = torch.zeros(g * g, d, device=device, dtype=torch.float64)
            S2 = torch.zeros(g * g, d, d, device=device, dtype=torch.float64)
        zf = z.double()
        S1 += zf.sum(0)
        S2 += torch.einsum("bpi,bpj->pij", zf, zf)
        n += B

    P, d = S1.shape
    if n <= d:
        raise SystemExit(f"Only {n} training images for a {d}-dimensional covariance. "
                         f"Reduce --dim below {n}.")
    mu = S1 / n
    cov = (S2 - n * torch.einsum("pi,pj->pij", mu, mu)) / (n - 1)
    cov += args.eps * torch.eye(d, device=device, dtype=torch.float64)
    prec = torch.linalg.inv(cov)
    logger.info(f"fitted {P} position-wise Gaussians over {n} training normals "
                f"at a {grid}x{grid} grid, d = {d}")

    # ---- pass 2: Mahalanobis maps ----
    size = cfg["data"]["image_size"]

    @torch.no_grad()
    def score(loader, want_maps=False):
        pooled_mean, pooled_max, maps, gts = [], [], [], []
        for b in loader:
            x, m, gt = unpack(b)
            x = x.to(device, non_blocking=True)
            z, _ = embed(extractor, x, idx, grid)
            B = z.shape[0]
            z = z.permute(0, 2, 3, 1).reshape(B, P, d).double()
            delta = z - mu
            md = torch.sqrt(torch.clamp(
                torch.einsum("bpi,pij,bpj->bp", delta, prec, delta), min=0))
            amap = Fn.interpolate(md.reshape(B, 1, grid, grid).float(),
                                  size=(size, size), mode="bilinear",
                                  align_corners=False)
            mm = m.to(device) if use_mask else torch.ones_like(amap)
            pooled_mean.append(((amap * mm).sum((1, 2, 3)) /
                                mm.sum((1, 2, 3)).clamp(min=1)).cpu())
            pooled_max.append((amap * mm).amax((1, 2, 3)).cpu())
            if want_maps and gt is not None:
                maps.append((amap * mm)[:, 0].cpu())
                gts.append(gt[:, 0].cpu() if gt.ndim == 4 else gt.cpu())
        out = {"mean": torch.cat(pooled_mean).numpy(),
               "max": torch.cat(pooled_max).numpy()}
        if want_maps and maps:
            return out, torch.cat(maps).numpy(), torch.cat(gts).numpy()
        return out, None, None

    normal, _, _ = score(test_loader)
    results = {"method": "PaDiM", "backbone": args.backbone, "layers": layers,
               "dim": int(d), "grid": int(grid), "n_train": int(n),
               "n_normal": int(len(normal["mean"]))}

    want_pixel = cfg["data"].get("module") == "rsna"
    for cls, loader in path_loaders.items():
        pos, maps, gts = score(loader, want_maps=want_pixel)
        block = {"n_samples": int(len(pos["mean"])), "by_score": {}}
        for k in ("mean", "max"):
            block["by_score"][f"padim/{k}"] = detection_metrics(normal[k], pos[k])
        if maps is not None and gts is not None and gts.sum() > 0:
            block["localisation"] = pixel_localisation(maps, gts)
        results[cls] = block
        a = block["by_score"]["padim/mean"]
        line = (f"{cls:30s} AUROC {a['auroc']:.4f} "
                f"[{a['auroc_ci95'][0]:.4f}, {a['auroc_ci95'][1]:.4f}]  "
                f"AUPRC {a['auprc']:.4f} (chance {a['auprc_baseline']:.3f})")
        if "localisation" in block:
            L = block["localisation"]
            line += (f"  | pixel AUPRC {L['pixel_auprc']:.4f} "
                     f"(chance {L['pixel_baseline']:.4f}, "
                     f"lift {L['pixel_auprc'] - L['pixel_baseline']:+.4f})")
        logger.info(line)

    Path(args.out).write_text(json.dumps(results, indent=2))
    logger.info(f"written to {args.out}")


if __name__ == "__main__":
    main()