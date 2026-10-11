"""
loc_evaluate.py  --  stage 10: improved scoring for a trained VAE.

    python ../common/loc_evaluate.py --config config.yaml

Computes every Tier 1 map in loc_maps.py from one trained VAE and reports

    detection      image AUROC per pathology class      (COVID and RSNA)
    localisation   pixel AUPRC, pixel AUROC, IoU         (RSNA, which has boxes)

for each map under four post-processing variants (raw, smooth, lung,
lung+smooth) and two image-level poolings (mean, top 5 %).

HOW THE METHODS ARE KEPT COMPARABLE WITH EACH OTHER

Localisation is always evaluated on the FULL FRAME. The lung variant does
not restrict which pixels are scored; it sets out-of-lung pixels to the
lowest value in the map. So every method and variant sees identical pixels
and identical labels, and their numbers compare directly.

Detection pools the map into one score per image. For the lung variant the
pooling runs over lung pixels only, which is the natural image-level use of
a lung mask.

Every map is divided by its own healthy-validation standard deviation before
it is stored. AUROC, AUPRC and the oracle IoU are all invariant to a single
positive rescaling applied to every image, so this changes no reported
number; it keeps tiny gradient magnitudes representable in float16 and lets
the fusion map average methods on a common scale.

FOUR PASSES

    A  validation normals   per-pixel mean and sd of the pixel residual
    B  validation normals   each method's global mean and sd
    C  test normals         detection negatives
    D  pathology classes    detection positives, and stored maps for
                            localisation where boxes exist
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loc_maps as LM                                            # noqa: E402
from metrics import pixel_localisation                          # noqa: E402
from utils import (build_datamodule, get_device, get_logger,    # noqa: E402
                   load_checkpoint, load_config, set_seed)

try:
    from vae_model import VAE
except ImportError:
    from model import VAE

from sklearn.metrics import roc_auc_score                       # noqa: E402

VARIANTS = ("raw", "smooth", "lung", "lung+smooth")
POOLS = ("mean", "top5")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def resolve(spec):
    p = os.path.expanduser(spec)
    if os.path.isfile(p):
        return p
    hits = sorted(glob.glob(p), key=os.path.getmtime)
    if not hits:
        raise SystemExit(f"No checkpoint matches {spec!r}")
    return hits[-1]


def load_vae(path, device):
    """Builds the model from the config saved INSIDE the checkpoint, so the
    architecture is always the one that was trained."""
    ck = load_checkpoint(path, device)
    cfg = ck.get("config")
    if cfg is None:
        raise SystemExit(f"{path} carries no saved config; cannot rebuild it safely")
    m = VAE(cfg).to(device)
    m.load_state_dict(ck["model_state"])
    return m.eval(), cfg


def unpack(b):
    if isinstance(b, dict):
        return b["image"], b.get("gt_mask")
    if isinstance(b, (list, tuple)):
        return b[0], None
    return b, None


def loaders(dm):
    n = getattr(dm, "normal_loaders", None) or getattr(dm, "get_normal_loaders")
    p = getattr(dm, "pathology_loaders", None) or getattr(dm, "get_pathology_loaders")
    return n(), p()


# --------------------------------------------------------------------------- #
# Map computation for one batch
# --------------------------------------------------------------------------- #

class Context:
    def __init__(self, cfg, device, logger):
        L = cfg["loc"]
        self.L, self.device, self.log = L, device, logger
        self.model, self.mcfg = load_vae(resolve(L["source_checkpoint"]), device)
        self.beta = float(self.mcfg.get("loss", {}).get("beta", 1.0))
        self.methods = list(L["methods"])
        self.pix_mu = self.pix_sd = None
        self.gmu, self.gsd = {}, {}

        if "feat_multiscale" in self.methods:
            from perceptual import build_extractor
            self.fext = build_extractor(L.get("backbone", "resnet50_imagenet"),
                                        layers=L.get("feat_layers")
                                        or ["layer1", "layer2", "layer3"]).to(device).eval()
            for p in self.fext.parameters():
                p.requires_grad_(False)

        self.ens = []
        if "ensemble_var" in self.methods:
            specs = L.get("ensemble_checkpoints") or []
            if len(specs) < 2:
                logger.warning("ensemble_var needs at least 2 checkpoints; dropping it")
                self.methods.remove("ensemble_var")
            else:
                self.ens = [load_vae(resolve(s), device)[0] for s in specs]
                logger.info(f"ensemble: {len(self.ens)} models")

        self.sg = None
        if "mars_spatial" in self.methods:
            fit = L.get("mars_fit")
            if not fit or not os.path.isfile(os.path.expanduser(fit)):
                logger.warning("mars_spatial needs loc.mars_fit; dropping it")
                self.methods.remove("mars_spatial")
            else:
                from mars import SpatialGaussian
                from perceptual import build_extractor
                fit = os.path.expanduser(fit)
                summ = json.loads((Path(fit).parent / "fit_summary.json").read_text())
                self.sg = SpatialGaussian.load(np.load(fit), device)
                self.mext = build_extractor(summ["backbone"],
                                            layers=summ["layers"]).to(device).eval()
                for p in self.mext.parameters():
                    p.requires_grad_(False)

        self.lung = None
        if any("lung" in v for v in L.get("variants", VARIANTS)):
            try:
                self.lung = LM.LungMasker(device, int(L.get("lung_dilate", 3)))
                logger.info("lung masking: TorchXRayVision PSPNet")
            except Exception as e:                           # noqa: BLE001
                logger.warning(f"lung masker unavailable ({e}); lung variants skipped")

    def maps(self, x, need):
        """Returns {method: (B,1,H,W) float32}. `need` limits the work in
        pass A, which only wants the pixel residual."""
        out = {}
        with torch.no_grad():
            xh = LM.recon(self.model, x)
        r = (x - xh).abs()
        if "pixel" in need:
            out["pixel"] = r
        if "uniform" in need:
            # A BLANK map. Under the lung variant it scores every lung pixel
            # equally and pushes the rest to the floor, so its localisation is
            # what the lung mask achieves ON ITS OWN. Every other method's
            # lung-variant result should be read against this: the map adds
            # only what it gains above it.
            out["uniform"] = torch.zeros_like(x)
        if "normative" in need and self.pix_mu is not None:
            out["normative"] = (r - self.pix_mu) / (self.pix_sd + 1e-4)
        if "prior_sub" in need and self.pix_mu is not None:
            out["prior_sub"] = (r - self.pix_mu).clamp(min=0)
        if "ssim" in need:
            out["ssim"] = LM.ssim_map(x, xh)
        if "feat_multiscale" in need:
            out["feat_multiscale"] = LM.feat_multiscale(self.fext, x, xh)
        if "kl_grad" in need:
            out["kl_grad"] = LM.kl_grad(self.model, x)
        if "elbo_grad" in need:
            out["elbo_grad"] = LM.elbo_grad(self.model, x, self.beta)
        if "sample_var" in need:
            out["sample_var"] = LM.sample_var(self.model, x, int(self.L.get("sample_k", 8)))
        if "ensemble_var" in need and self.ens:
            out["ensemble_var"] = LM.ensemble_var(self.ens, x)
        # ---- Tier 2: restored reconstructions, each scored two ways ----
        T2 = self.L.get("tier2", {}) or {}
        if need & {"proj_grad", "proj_ssim"}:
            xp = LM.project(self.model, x, T2.get("proj_steps", 30),
                            T2.get("proj_lr", 5e-3), T2.get("proj_lambda", 0.1),
                            self.beta)
            if "proj_grad" in need:
                out["proj_grad"] = (x - xp).abs()
            if "proj_ssim" in need:
                out["proj_ssim"] = LM.ssim_map(x, xp)
        if need & {"restore", "restore_ssim"} and self.pix_mu is not None:
            xr = LM.restore(self.model, x, self.pix_mu, self.pix_sd,
                            T2.get("restore_iters", 3), T2.get("restore_z", 2.0))
            if "restore" in need:
                out["restore"] = (x - xr).abs()
            if "restore_ssim" in need:
                out["restore_ssim"] = LM.ssim_map(x, xr)
        if need & {"latent_trunc"}:
            out["latent_trunc"] = (x - LM.truncate(self.model, x,
                                   T2.get("trunc_scale", 0.5))).abs()
        if "mars_spatial" in need and self.sg is not None:
            from mars import spatial_embed
            with torch.no_grad():
                sx, _ = spatial_embed(self.mext, x, self.sg.grid)
                sr, _ = spatial_embed(self.mext, xh, self.sg.grid)
                out["mars_spatial"] = self.sg.score_map(sx - sr, x.shape[-1])
        return {k: v.float().detach() for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Variants and pooling
# --------------------------------------------------------------------------- #

def apply_variant(m, lung, variant, sigma):
    """m (B,1,H,W). Lung pixels kept; others pushed below the map minimum."""
    if "smooth" in variant:
        m = LM.gaussian_blur(m, sigma)
    if "lung" in variant:
        if lung is None:
            return None
        floor = m.amin((1, 2, 3), keepdim=True) - 1.0
        m = torch.where(lung, m, floor)
    return m


def pool(m, lung, variant, how):
    """Image score. Under the lung variant, pooled over lung pixels only."""
    B = m.shape[0]
    v = m.reshape(B, -1)
    if "lung" in variant and lung is not None:
        msk = lung.reshape(B, -1).clone()
        # An image where the segmenter finds NO lung would otherwise pool over
        # an empty set: -inf under top-5 %, which is what broke the first run.
        # Those images fall back to the whole frame. Dense consolidation can
        # genuinely read as "no lung", so this is expected occasionally on
        # pathology and is counted rather than hidden.
        empty = msk.sum(1) == 0
        if empty.any():
            msk[empty] = True
            pool.empty += int(empty.sum())
        big = torch.full_like(v, float("-inf"))
        v_in = torch.where(msk, v, big)
        if how == "mean":
            n = msk.sum(1).clamp(min=1)
            return (torch.where(msk, v, torch.zeros_like(v)).sum(1) / n)
        k = (msk.sum(1).float() * 0.05).clamp(min=1).long()
        s = torch.sort(v_in, dim=1, descending=True).values
        return torch.stack([s[i, :k[i]].mean() for i in range(B)])
    if how == "mean":
        return v.mean(1)
    k = max(1, int(v.shape[1] * 0.05))
    return v.topk(k, dim=1).values.mean(1)


pool.empty = 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 10: improved VAE scoring")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    stage = load_config(args.config)
    device = get_device(stage["experiment"].get("device", "cuda"))

    # Data and seed come from the SOURCE checkpoint, so the split matches the
    # one the VAE was trained on and no test image was seen in training.
    src = resolve(stage["loc"]["source_checkpoint"])
    base = load_checkpoint(src, device)["config"]
    cfg = copy.deepcopy(base)
    cfg["loc"] = stage["loc"]
    cfg.setdefault("paths", {}).update(stage.get("paths", {}))
    cfg["experiment"]["name"] = stage["experiment"]["name"]
    set_seed(cfg["experiment"]["seed"])

    run_dir = (Path(cfg["paths"]["output_root"]).expanduser() / cfg["experiment"]["name"] /
               f"{time.strftime('%Y%m%d_%H%M%S')}_seed{cfg['experiment']['seed']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("stage10", run_dir / "stage10.log")
    log.info(f"source VAE : {src}")
    log.info(f"dataset    : {cfg['data'].get('module')}   seed {cfg['experiment']['seed']}")

    C = Context(cfg, device, log)
    variants = [v for v in C.L.get("variants", VARIANTS)
                if "lung" not in v or C.lung is not None]
    sigma = float(C.L.get("smooth_sigma", 2.0))
    fusion = [m for m in C.L.get("fusion", []) if m in C.methods]
    log.info(f"methods    : {C.methods}")
    log.info(f"variants   : {variants}   fusion of {fusion or 'none'}")

    dm = build_datamodule(cfg)
    (_, val_l, test_l), path_l = loaders(dm)

    # ---- pass A: per-pixel healthy residual statistics ----
    s1 = s2 = None; n = 0
    for b in val_l:
        x = unpack(b)[0].to(device)
        r = C.maps(x, {"pixel"})["pixel"]
        s1 = r.sum(0) if s1 is None else s1 + r.sum(0)
        s2 = (r ** 2).sum(0) if s2 is None else s2 + (r ** 2).sum(0)
        n += x.shape[0]
    C.pix_mu = s1 / n
    C.pix_sd = (s2 / n - C.pix_mu ** 2).clamp(min=0).sqrt()
    log.info(f"pass A: per-pixel residual stats over {n} validation normals")

    # ---- pass B: global scale per method, from healthy validation ----
    acc = {m: [0.0, 0.0, 0] for m in C.methods}
    for b in val_l:
        x = unpack(b)[0].to(device)
        for m, v in C.maps(x, set(C.methods)).items():
            acc[m][0] += float(v.sum()); acc[m][1] += float((v ** 2).sum())
            acc[m][2] += v.numel()
    for m, (a, q, c) in acc.items():
        if c:
            mu = a / c
            C.gmu[m] = mu
            C.gsd[m] = max((q / c - mu ** 2), 1e-24) ** 0.5
    log.info("pass B: method scales from healthy validation")

    def norm(m, v):
        return (v - C.gmu[m]) / C.gsd[m]

    def batch_scores(x, store_maps):
        """Image scores for every (method, variant, pool); optionally maps."""
        mp = C.maps(x, set(C.methods))
        mp = {m: norm(m, v) for m, v in mp.items() if m in C.gmu}
        if fusion:
            mp["fusion"] = torch.stack([mp[m] for m in fusion]).mean(0)
        lung = C.lung(x) if C.lung is not None else None
        sc = {}
        for m, v in mp.items():
            for var in variants:
                t = apply_variant(v, lung, var, sigma)
                if t is None:
                    continue
                for how in POOLS:
                    sc[(m, var, how)] = pool(t, lung, var, how).cpu().numpy()
        stored = ({m: v.half().cpu() for m, v in mp.items()}, lung.cpu()
                  if lung is not None else None) if store_maps else None
        return sc, stored

    def run(loader, store_maps=False):
        S, M, LU, G = {}, {}, [], []
        for b in loader:
            x, gt = unpack(b)
            x = x.to(device)
            sc, st = batch_scores(x, store_maps and gt is not None)
            for k, v in sc.items():
                S.setdefault(k, []).append(v)
            if st is not None:
                for m, v in st[0].items():
                    M.setdefault(m, []).append(v)
                if st[1] is not None:
                    LU.append(st[1])
                G.append((gt[:, 0] if gt.ndim == 4 else gt).bool())
        S = {k: np.concatenate(v) for k, v in S.items()}
        M = {k: torch.cat(v) for k, v in M.items()}
        return S, M, (torch.cat(LU) if LU else None), (torch.cat(G) if G else None)

    # ---- pass C: test normals ----
    S_norm, _, _, _ = run(test_l)
    log.info(f"pass C: {len(next(iter(S_norm.values())))} test normals")

    # ---- pass D: pathology classes ----
    det, loc = [], []
    for cls, loader in path_l.items():
        S_p, M_p, LU, G = run(loader, store_maps=True)
        npos = len(next(iter(S_p.values())))
        if pool.empty:
            log.info(f"  lung masker found no lung in some images; {pool.empty} "
                     f"image-scorings fell back to the whole frame so far")
        log.info(f"pass D: {cls}, {npos} images"
                 + (f", {int(G.any((1,2)).sum())} with boxes" if G is not None else ""))
        for (m, var, how), pos in S_p.items():
            neg = S_norm[(m, var, how)]
            sc_all = np.r_[neg, pos]
            bad = ~np.isfinite(sc_all)
            if bad.any():
                log.warning(f"  {m}/{var}/{how}: {int(bad.sum())} non-finite scores, "
                            f"set to the finite minimum")
                sc_all[bad] = np.nanmin(np.where(bad, np.nan, sc_all))
                neg, pos = sc_all[:len(neg)], sc_all[len(neg):]
            y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
            det.append(dict(cls=cls, method=m, variant=var, pool=how,
                            auroc=float(roc_auc_score(y, np.r_[neg, pos]))))

        if G is not None and G.any():
            keep = G.reshape(G.shape[0], -1).any(1)
            mx = C.L.get("max_images")
            idx = torch.nonzero(keep).squeeze(1)
            if mx:
                idx = idx[: int(mx)]
            gts = G[idx].numpy().astype(np.uint8)
            lung = LU[idx].to(device) if LU is not None else None
            for m, V in M_p.items():
                for var in variants:
                    V_ = V[idx].float().to(device)
                    t = apply_variant(V_, lung, var, sigma)
                    if t is None:
                        continue
                    r = pixel_localisation(t[:, 0].cpu().numpy(), gts)
                    loc.append(dict(cls=cls, method=m, variant=var,
                                    pixel_auprc=float(r["pixel_auprc"]),
                                    pixel_auroc=float(r["pixel_auroc"]),
                                    iou=float(r["mean_iou"])))
                    log.info(f"  loc  {m:16s} {var:12s} AUPRC {r['pixel_auprc']:.4f}  "
                             f"AUROC {r['pixel_auroc']:.4f}  IoU {r['mean_iou']:.4f}")
                    del V_, t
                    torch.cuda.empty_cache()

    # ---- reporting ----
    out = {"source_checkpoint": src, "seed": cfg["experiment"]["seed"],
           "dataset": cfg["data"].get("module"), "detection": det, "localisation": loc}
    (run_dir / "stage10_results.json").write_text(json.dumps(out, indent=2))
    for name, rows in (("detection", det), ("localisation", loc)):
        if rows:
            with open(run_dir / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
                w.writerows(rows)

    log.info("")
    log.info("TOP DETECTION (image AUROC), per class")
    for cls in sorted({r["cls"] for r in det}):
        best = sorted([r for r in det if r["cls"] == cls], key=lambda r: -r["auroc"])[:8]
        log.info(f"  {cls}")
        for r in best:
            log.info(f"    {r['auroc']:.4f}  {r['method']:16s} {r['variant']:12s} {r['pool']}")
    if loc:
        log.info("")
        log.info("TOP LOCALISATION (ranked by pixel AUPRC)")
        for r in sorted(loc, key=lambda r: -r["pixel_auprc"])[:12]:
            log.info(f"    AUPRC {r['pixel_auprc']:.4f}  AUROC {r['pixel_auroc']:.4f}  "
                     f"IoU {r['iou']:.4f}  {r['method']:16s} {r['variant']}")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
