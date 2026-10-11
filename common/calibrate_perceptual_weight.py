"""
calibrate_perceptual_weight.py

Measures the raw magnitude of a backbone's perceptual term against the
reconstruction term, and reports the weight that would give it a chosen share
of the objective.

WHY THIS EXISTS

The perceptual weight was originally set to 0.5 on the reasoning that both
terms of the objective are dimensionless, so a weight near unity should give
the perceptual term a substantial but not dominant share. That reasoning was
wrong, and the training logs showed why: the raw perceptual term settled near
0.410 while the mean squared pixel error on a well-reconstructed radiograph
sat near 0.0039, a factor of 105 before any weight was applied. Multiplying
the larger quantity by 0.5 put it at 98 percent of the objective, left the
Kullback-Leibler term at 0.1 percent, and the information bottleneck stopped
being enforced. Detection fell by 0.08 to 0.10 AUROC. Reducing the weight to
0.001 recovered all of it.

Dimensionless is not the same as commensurate. Every backbone has its own
activation scale, so a weight calibrated for VGG-19 says nothing about ResNet
and less than nothing about a vision transformer, whose token magnitudes are
unrelated to a convolutional feature map's. Reusing 0.001 across backbones
would repeat the original error with a different number.

Run this once per backbone before training. It costs a few batches.

    python calibrate_perceptual_weight.py --config ../stage4_resnet/config.yaml
    python calibrate_perceptual_weight.py --config ../stage5_raddino/config.yaml --target 0.15

WHAT IT MEASURES

The raw perceptual distance is evaluated between real images and a stand-in
for an early-training reconstruction, since that is the regime in which the
weight has to be sane. Two stand-ins are offered:

  blur   (default) a Gaussian-blurred copy of the input. A tight-bottleneck
         VAE produces prototype-like output, so a blurred image is a
         reasonable proxy for what the decoder emits and gives a stable,
         reproducible number.
  model  an actual reconstruction from a checkpoint, if one exists. More
         faithful, but only available after a run.

The reconstruction term is measured the same way, so the ratio between them
is what matters and the choice of stand-in largely cancels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perceptual import PerceptualLoss, build_extractor   # noqa: E402
from utils import build_datamodule, get, get_device, load_config  # noqa: E402


def gaussian_blur(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """A stand-in for an early-training reconstruction."""
    k = int(2 * round(3 * sigma) + 1)
    coords = torch.arange(k, dtype=x.dtype, device=x.device) - (k - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, 1, -1)
    c = x.size(1)
    x = F.conv2d(F.pad(x, (k // 2,) * 2 + (0, 0), mode="reflect"),
                 g.expand(c, 1, 1, k), groups=c)
    x = F.conv2d(F.pad(x, (0, 0) + (k // 2,) * 2, mode="reflect"),
                 g.transpose(-1, -2).expand(c, 1, k, 1), groups=c)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate the perceptual weight")
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--target", type=float, default=0.12,
                    help="Desired share of the objective, 0.10 to 0.15 is sane")
    ap.add_argument("--source", choices=["blur", "model"], default="blur")
    ap.add_argument("--checkpoint", default=None, help="Required for --source model")
    ap.add_argument("--sigma", type=float, default=2.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["experiment"].get("device", "cuda"))
    pc = get(cfg, "loss.perceptual", {}) or {}
    if not pc.get("backbone"):
        raise SystemExit("loss.perceptual.backbone is not set in this config")

    norm = pc.get("normalization")
    if norm is not None:
        norm = (tuple(norm[0]), tuple(norm[1]))
    extractor = build_extractor(
        pc["backbone"], layers=pc.get("layers"), weights=pc.get("weights"),
        checkpoint=pc.get("checkpoint"), normalization=norm,
        input_size=pc.get("input_size", 224),
    ).to(device)
    loss_fn = PerceptualLoss(
        extractor, layer_weights=pc.get("layer_weights"),
        distance=pc.get("distance", "l2"),
        normalize_layers=pc.get("normalize_layers", True),
    ).to(device)

    print(f"backbone      : {pc['backbone']}")
    if hasattr(extractor, "extra_repr"):
        print(f"configuration : {extractor.extra_repr()}")
    if hasattr(extractor, "load_report"):
        print(f"load report   : {extractor.load_report}")
    frozen = sum(p.numel() for p in extractor.parameters())
    print(f"frozen params : {frozen:,}")

    dm = build_datamodule(cfg)
    train_loader, _, _ = dm.normal_loaders()

    model = None
    if args.source == "model":
        if not args.checkpoint:
            raise SystemExit("--source model requires --checkpoint")
        from model import VAE
        from utils import load_checkpoint
        model = VAE(cfg).to(device)
        model.load_state_dict(load_checkpoint(args.checkpoint, device)["model_state"])
        model.eval()

    perc_vals, recon_vals, n = [], [], 0
    with torch.no_grad():
        for batch in train_loader:
            x = batch["image"].to(device)
            if model is not None:
                xh, _ = model.reconstruct(x, n_samples=1)
            else:
                xh = gaussian_blur(x, args.sigma)
            perc_vals.append(float(loss_fn(xh, x)))
            recon_vals.append(float(F.mse_loss(xh, x)))
            n += 1
            if n >= args.batches:
                break

    perc = sum(perc_vals) / len(perc_vals)
    recon = sum(recon_vals) / len(recon_vals)
    kl = float(get(cfg, "loss.beta", 0.004)) * 0.0275   # KL/n_pixels, typical at convergence
    base = recon + kl

    print(f"\nmeasured over {n} batches, source = {args.source}")
    print(f"  raw perceptual term     {perc:.6f}")
    print(f"  reconstruction term     {recon:.6f}")
    print(f"  beta x KL (typical)     {kl:.6f}")
    print(f"  scale mismatch          {perc / max(recon, 1e-12):.0f}x before any weight\n")

    print(f"  {'weight w':>11s} {'w x perc':>11s} {'share':>8s}")
    for w in (0.5, 0.1, 0.05, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002):
        wp = w * perc
        print(f"  {w:11.4f} {wp:11.6f} {wp / (base + wp):8.1%}")

    t = args.target
    w_star = t * base / (perc * (1 - t))
    print(f"\n  RECOMMENDED for a {t:.0%} share:  loss.perceptual.weight: {w_star:.5f}")
    print(f"  (for 10%: {0.10*base/(perc*0.90):.5f}   for 15%: {0.15*base/(perc*0.85):.5f})")
    print("\n  This is a starting point, not a final answer. Read share_perc in the")
    print("  epoch-1 log and adjust if it is far from the target: the raw term")
    print("  changes as the model improves, and the estimate above assumes an")
    print("  early-training reconstruction.")


if __name__ == "__main__":
    main()