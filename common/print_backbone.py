"""
print_backbone.py

Architecture printout for the frozen perceptual backbone named in a stage
config: the extracted layers with their output shapes, how much of the loss
each one carries, the module tree, and a torchinfo table.

    python ../common/print_backbone.py --config config.yaml
    python ../common/print_backbone.py --backbone rad_dino          # no config needed

THE PART THAT MATTERS IS THE FEATURE-VOLUME COLUMN

A perceptual loss is a sum over the extracted layers, so how much each layer
contributes depends on how many numbers it produces, not on how informative it
is. For a VGG-19 at 128 pixels the first extracted layer emits 64 channels at
full resolution while the last emits 512 at an eighth of it, so the shallow
layer carries roughly sixteen times the element count of the deep one. Left
alone, an edge detector would dominate a loss meant to capture structure at
several scales.

That is what loss.perceptual.normalize_layers corrects: each layer's error is
divided by the magnitude of its own target features before summing, so the
layers contribute comparably regardless of size. This printout shows the raw
imbalance being corrected, which is worth having in an appendix because it is
a design decision a reader would otherwise have to take on trust.

It also reports the backbone's input size against the size it was pretrained
at, and the shapes are traced at the resolution your pipeline actually uses,
not at the backbone's nominal one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perceptual import REGISTRY, build_extractor      # noqa: E402
from utils import get, load_config                    # noqa: E402

# Resolution each backbone was pretrained at, for the mismatch note.
PRETRAIN_SIZE = {
    "vgg19_imagenet": 224, "vgg19_cxr": 128,
    "densenet_imagenet": 224, "densenet_cxr": 224,
    "resnet_cxr": 512, "resnet50_imagenet": 224,
    "resnet50_radimagenet": 224, "rad_dino": 518,
}


def render_png(text: str, path: str, font_size: int = 7) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = text.split("\n")
    width = max((len(l) for l in lines), default=80)
    fig = plt.figure(figsize=(min(max(width * font_size * 0.0098, 6), 26),
                              min(max(len(lines) * font_size * 0.0208, 3), 60)))
    fig.text(0.01, 0.99, text, family="DejaVu Sans Mono", fontsize=font_size,
             va="top", ha="left", linespacing=1.32)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Print a perceptual backbone")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--backbone", default=None,
                    help="Override the config, e.g. rad_dino. Uses that backbone's defaults.")
    ap.add_argument("--image-size", type=int, default=None,
                    help="Trace at this resolution. Defaults to data.image_size.")
    ap.add_argument("--out", default="backbone_summary.txt")
    ap.add_argument("--out-png", default="backbone_summary.png")
    args = ap.parse_args()

    if args.backbone:
        name, pc, size = args.backbone, {}, args.image_size or 128
        if name not in REGISTRY:
            raise SystemExit(f"Unknown backbone {name!r}. Available: {sorted(REGISTRY)}")
    else:
        cfg = load_config(args.config)
        pc = get(cfg, "loss.perceptual", {}) or {}
        name = pc.get("backbone")
        size = args.image_size or cfg["data"]["image_size"]
        if not pc.get("enabled", False):
            print(f"NOTE: loss.perceptual.enabled is false in {args.config}; "
                  f"printing {name} anyway.\n")
        if not name:
            raise SystemExit("No backbone named in loss.perceptual.backbone")

    norm = pc.get("normalization")
    if norm is not None:
        norm = (tuple(norm[0]), tuple(norm[1]))
    ex = build_extractor(name, layers=pc.get("layers"), weights=pc.get("weights"),
                         checkpoint=pc.get("checkpoint"), normalization=norm,
                         input_size=pc.get("input_size", 224)).eval()

    x = torch.zeros(2, 1, size, size)
    with torch.no_grad():
        prepared = ex.prepare(x)
        feats = ex(x)

    frozen = sum(p.numel() for p in ex.parameters())
    trainable = sum(p.numel() for p in ex.parameters() if p.requires_grad)

    L = []
    L.append("=" * 104)
    L.append(f"PERCEPTUAL BACKBONE -- {name}")
    L.append("=" * 104)
    L.append(f"Pipeline image size        : {size} x {size}, 1 channel")
    L.append(f"Delivered to the backbone  : {tuple(prepared.shape)}")
    resize = getattr(ex, "input_size", None)
    if resize and resize != size:
        L.append(f"  resized                  : {size} -> {resize} "
                 f"({'up' if resize > size else 'down'}sample x{max(resize,size)/min(resize,size):.2f}); "
                 f"upsampling adds no information")
    else:
        L.append(f"  resized                  : no, run at the pipeline resolution")
    if prepared.shape[1] != 1:
        L.append(f"  channels                 : 1 repeated to {prepared.shape[1]}")
    pre = PRETRAIN_SIZE.get(name)
    if pre:
        eff = resize or size
        L.append(f"Pretrained at              : {pre} x {pre}"
                 + ("" if eff == pre else
                    f"   MISMATCH: receives {eff}, so features are computed at a scale "
                    f"the network was not calibrated for"))
    L.append(f"Input normalization        : {getattr(ex, 'normalization', 'none')}")
    L.append(f"Frozen parameters          : {frozen:,}")
    L.append(f"Trainable parameters       : {trainable:,}   "
             f"{'(correct: the backbone must not learn)' if trainable == 0 else '(SHOULD BE ZERO)'}")
    L.append(f"Training mode              : {ex.training}   "
             f"{'(correct: eval, so normalization statistics cannot drift)' if not ex.training else '(SHOULD BE False)'}")
    if hasattr(ex, "load_report"):
        L.append(f"Checkpoint load report     : {ex.load_report}")
    L.append("")

    L.append("-" * 104)
    L.append("EXTRACTED LAYERS -- what the perceptual loss actually compares")
    L.append("-" * 104)
    total_elems = sum(f[0].numel() for f in feats)
    names = getattr(ex, "names", [f"layer {i}" for i in range(len(feats))])
    L.append(f"  {'layer':16s} {'output shape (per image)':28s} {'elements':>12s} "
             f"{'share of volume':>16s}")
    for nm, f in zip(names, feats):
        per = f[0]
        L.append(f"  {str(nm):16s} {str(tuple(per.shape)):28s} {per.numel():12,} "
                 f"{per.numel()/total_elems:15.1%}")
    L.append(f"  {'TOTAL':16s} {'':28s} {total_elems:12,}")
    L.append("")
    biggest = max(feats, key=lambda f: f[0].numel())[0].numel()
    smallest = min(feats, key=lambda f: f[0].numel())[0].numel()
    L.append(f"  The largest extracted layer carries {biggest/smallest:.1f} times the element")
    L.append(f"  count of the smallest. Without per-layer normalization it would dominate")
    L.append(f"  the loss by size rather than by relevance, which is what")
    L.append(f"  loss.perceptual.normalize_layers corrects.")
    L.append("")

    L.append("-" * 104)
    L.append("LAYER-BY-LAYER SUMMARY (torchinfo)")
    L.append("-" * 104)
    table = ""
    try:
        from torchinfo import summary
        result = summary(ex, input_data=x,
                         col_names=["input_size", "output_size", "num_params"],
                         depth=3, verbose=0)
        table = str(result)
        L.append(table)
    except ImportError:
        L.append("  torchinfo is not installed; the sections above do not depend on it.")
        L.append("  pip install torchinfo")
    except Exception as exc:                          # noqa: BLE001
        L.append(f"  torchinfo failed: {type(exc).__name__}: {exc}")
    L.append("")

    L.append("-" * 104)
    L.append("MODULE TREE (PyTorch native repr)")
    L.append("-" * 104)
    L.append(str(ex))

    text = "\n".join(L)
    Path(args.out).write_text(text)
    print(text)
    print(f"\nwritten to {args.out}")

    header = (f"Perceptual backbone: {name}\n"
              f"pipeline input {size}x{size}, delivered {tuple(prepared.shape)}, "
              f"{frozen:,} frozen parameters\n\n")
    render_png(header + "\n".join(L[L.index("EXTRACTED LAYERS -- what the perceptual loss actually compares") - 1:]),
               args.out_png)
    print(f"rendered to {args.out_png}")


if __name__ == "__main__":
    main()