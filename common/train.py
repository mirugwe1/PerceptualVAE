"""
train.py

Trains the anomaly-detection VAE.

TWO CHOICES WORTH FLAGGING.

MODEL SELECTION IS BY VALIDATION ELBO, NOT BY DETECTION SCORE. It is
tempting to checkpoint on validation AUROC, but the only labelled data
available for that is the pathology set, which is the test set. Selecting on
it leaks and inflates every number afterwards. Validation here is held-out
normal images only, and the monitored quantity is the objective itself.

EVERY TERM'S SHARE OF THE OBJECTIVE IS LOGGED EVERY EPOCH. A term that
contributes nothing should be visible in epoch 1, not inferred after twelve
hours from a result that looks like the ablation. With the mean-reduced
objective all terms are order 1, so the shares should be sane by default,
but they are logged anyway because the failure is silent when it happens.

INPAINTING (optional). When training.inpaint.enabled is true, the model sees
a masked input and is scored against the ORIGINAL image. Validation and test
use masks and latent noise drawn from a fixed seed (training.inpaint.val_seed),
so validation loss is comparable across epochs and best.pt is not chosen on
mask luck.

Usage:
    python train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perceptual import PerceptualLoss, build_extractor  # noqa: E402
from utils import (                                      # noqa: E402
    EarlyStopping, build_datamodule, count_parameters, get, get_device,
    get_logger, load_config, save_checkpoint, set_seed,
)
from losses import VAEObjective                          # noqa: E402
from vae_model import VAE                                # noqa: E402
from inpaint import training_mask, apply_mask            # noqa: E402
from corruption import corrupt


KEYS = ["total", "recon", "kl", "kl_per_dim", "perceptual",
        "beta_t", "share_kl", "share_perc"]


def run_epoch(model, objective, loader, device, epoch, optimizer=None,
              scaler=None, clip=None, amp=False, ip=None, train_mean=None,
              fixed_seed=None, corr=None):
    """One pass over `loader`.

    ip          : inpainting config dict (empty/None disables it).
    train_mean  : fill tensor for masked pixels (None means zero fill).
    fixed_seed  : if set, RNG state (masks + latent noise) is seeded with this
                  value for the pass and restored afterwards, so evaluation
                  numbers are reproducible across epochs.
    """
    ip = ip or {}
    use_ip = bool(ip.get("enabled", False))
    train = optimizer is not None
    model.train(train)

    acc = {k: 0.0 for k in KEYS}
    n = 0

    if fixed_seed is not None:
        devs = [device] if device.type == "cuda" else []
        rng_ctx = torch.random.fork_rng(devices=devs)
    else:
        rng_ctx = nullcontext()

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with rng_ctx, grad_ctx:
        if fixed_seed is not None:
            torch.manual_seed(fixed_seed)

        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            bsz = x.size(0)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
                if use_ip:
                    m = training_mask(x.shape[0], x.shape[-1], ip["patch"],
                                      ip["test_n"], x.device,
                                      ip.get("scheme", "disjoint"))
                    out = model(apply_mask(x, m, train_mean), sample=True)
                else:
                    out = model(corrupt(x, corr), sample=True)   # corrupted input, clean target
                # The TARGET is the original x in both branches.
                losses = objective(x, out, epoch)

            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(losses["total"]).backward()
                    if clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total"].backward()
                    if clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()

            for k in KEYS:
                acc[k] += float(losses[k]) * bsz
            n += bsz

    return {k: v / max(n, 1) for k, v in acc.items()}


@torch.no_grad()
def compute_train_mean(train_loader, device):
    """Per-pixel mean over the training set, accumulated in float64."""
    s, n = None, 0
    for b in train_loader:
        xb = (b["image"] if isinstance(b, dict)
              else (b[0] if isinstance(b, (list, tuple)) else b))
        xb = xb.to(device).double()
        s = xb.sum(0) if s is None else s + xb.sum(0)
        n += xb.size(0)
    return (s / max(n, 1)).float()


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["experiment"]["seed"], cfg["experiment"].get("deterministic", False))
    device = get_device(cfg["experiment"].get("device", "cuda"))

    run_dir = Path(cfg["paths"]["output_root"]) / cfg["experiment"]["name"] / \
        time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("train", run_dir / "train.log")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")

    with open(run_dir / "config.resolved.yaml", "w") as f:
        import yaml
        yaml.safe_dump(cfg, f, sort_keys=False)

    dm = build_datamodule(cfg)
    logger.info(f"Class counts: {dm.validate_layout()}")
    train_loader, val_loader, test_loader = dm.normal_loaders()
    logger.info(f"Normal split: train {len(train_loader.dataset)}, "
                f"val {len(val_loader.dataset)}, test {len(test_loader.dataset)}")

    model = VAE(cfg).to(device)
    logger.info(f"VAE parameters: {count_parameters(model):,} "
                f"(z_dim={model.z_dim}, latent={model.latent_kind}, "
                f"bottleneck={model.bottleneck_size}x{model.bottleneck_size}, "
                f"predict_sigma={model.predict_sigma})")

    # ---- perceptual loss ---------------------------------------------------
    perceptual_fn = None
    pc = get(cfg, "loss.perceptual", {}) or {}
    if pc.get("enabled", False):
        extractor = build_extractor(
            pc["backbone"], layers=pc.get("layers"), weights=pc.get("weights"),
            checkpoint=pc.get("checkpoint"), normalization=pc.get("normalization"),
        ).to(device)
        if hasattr(extractor, "load_report"):
            logger.info(f"Backbone load report: {extractor.load_report}")
            if not extractor.load_report["normalization_from_config"]:
                logger.warning(
                    "loss.perceptual.normalization is not set: falling back to "
                    "ImageNet statistics. If the fine-tuning run used different "
                    "ones, every feature map is shifted and this variant is not "
                    "measuring what it claims.")
        perceptual_fn = PerceptualLoss(
            extractor, layer_weights=pc.get("layer_weights"),
            distance=pc.get("distance", "l2"),
            normalize_layers=pc.get("normalize_layers", True),
        ).to(device)
        frozen = sum(p.numel() for p in extractor.parameters())
        logger.info(f"Perceptual backbone {pc['backbone']}: {frozen:,} frozen params, "
                    f"weight {pc.get('weight')}, trainable {count_parameters(extractor)} "
                    f"(must be 0)")

    # ---- inpainting: config and fill value, computed once ------------------
    tc = cfg["training"]
    ip = tc.get("inpaint", {}) or {}
    train_mean = None
    corr = tc.get("corruption", {}) or {}
    if corr.get("enabled"):
        logger.info(f"corruption: {corr}")
    val_seed = None
    if ip.get("enabled", False):
        if ip.get("fill") == "dataset_mean":
            train_mean = compute_train_mean(train_loader, device)
        val_seed = int(ip.get("val_seed", 0))
        logger.info(
            f"inpaint: {ip['patch']} px patches, {ip['test_n']} groups, "
            f"scheme {ip.get('scheme', 'disjoint')}, "
            f"fill {'training mean' if train_mean is not None else 'zero'}, "
            f"fixed val/test seed {val_seed}")

    objective = VAEObjective(cfg, perceptual_fn)

    # ---- optimisation ------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["learning_rate"],
        weight_decay=tc.get("weight_decay", 1e-4), betas=(0.9, 0.999),
    )
    max_epochs = tc["max_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=tc.get("min_learning_rate", 1e-6))

    amp = bool(tc.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    stopper = EarlyStopping(
        patience=tc["early_stopping"]["patience"],
        min_delta=tc["early_stopping"]["min_delta"],
    )
    writer = SummaryWriter(str(run_dir / "tb"))
    clip = tc.get("clip_grad_norm")
    best = float("inf")
    warned_perc = False
    warned_kl = False
    epoch = 0
    va = None

    for epoch in tqdm(range(1, max_epochs + 1), desc="epochs"):
        tr = run_epoch(model, objective, train_loader, device, epoch,
                       optimizer, scaler, clip, amp,
                       ip=ip, train_mean=train_mean, corr=corr)
        va = run_epoch(model, objective, val_loader, device, epoch, amp=amp,
                       ip=ip, train_mean=train_mean, fixed_seed=val_seed, corr=corr)
        scheduler.step()

        for k, v in tr.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in va.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            f"epoch {epoch:04d} | total {tr['total']:.5f} recon {tr['recon']:.5f} "
            f"kl {tr['kl']:.5f} (per-dim {tr['kl_per_dim']:.4f}, beta_t {tr['beta_t']:.3f}, "
            f"share {tr['share_kl']:.1%}) perc {tr['perceptual']:.5f} "
            f"(share {tr['share_perc']:.1%}) | val {va['total']:.5f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if epoch >= max(2, objective.warmup_epochs):
            if (not warned_perc and objective.w_perc > 0
                    and tr["share_perc"] < 0.01):
                logger.warning(
                    f"perceptual share is {tr['share_perc']:.2%}: this run is "
                    f"effectively the no-perceptual baseline. Raise loss.perceptual.weight.")
                warned_perc = True
            if (not warned_kl and objective.free_bits > 0
                    and tr["kl_per_dim"] <= objective.free_bits * 1.01):
                logger.warning(
                    f"mean KL per dimension ({tr['kl_per_dim']:.4f}) is pinned at the "
                    f"free-bits floor ({objective.free_bits}): the latent is carrying "
                    f"little information. Lower beta or free_bits.")
                warned_kl = True

        state = {"epoch": epoch, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "val_total": va["total"], "config": cfg}
        if va["total"] < best:
            best = va["total"]
            save_checkpoint(state, run_dir / "checkpoints", "best.pt")
        if epoch % tc.get("checkpoint_every", 50) == 0:
            save_checkpoint(state, run_dir / "checkpoints", f"epoch_{epoch:04d}.pt")

        if stopper.step(va["total"]):
            logger.info(f"Early stopping at epoch {epoch} (best val {best:.5f})")
            break

    save_checkpoint({"epoch": epoch, "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(),
                     "val_total": va["total"] if va else None, "config": cfg},
                    run_dir / "checkpoints", "last.pt")

    te = run_epoch(model, objective, test_loader, device, epoch, amp=amp,
                   ip=ip, train_mean=train_mean, fixed_seed=val_seed, corr=corr)
    logger.info(f"Held-out normal test objective: {te['total']:.5f}")
    with open(run_dir / "train_summary.json", "w") as f:
        json.dump({"best_val_total": best, "final_test": te, "epochs_run": epoch},
                  f, indent=2)
    writer.close()
    logger.info(f"Done. Checkpoints in {run_dir / 'checkpoints'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the CXR anomaly-detection VAE")
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    main(args.config)