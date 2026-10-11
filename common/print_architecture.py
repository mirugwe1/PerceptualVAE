"""
print_architecture.py

A publication-ready architecture printout for the VAE in this project: a
layer-by-layer table with shapes and per-layer parameter counts, a
per-component breakdown, and PyTorch's native module tree as a cross-check.
Written to a text file and rendered to PNG, both suitable for a thesis
appendix or supplementary methods.

    python ../common/print_architecture.py --config config.yaml

FOUR THINGS THIS HANDLES THAT A PLAIN torchinfo CALL DOES NOT

  THE MODULE NAME. common/model.py had to be renamed to vae_model.py because
  torchxrayvision imports a package called "model", and with common/ first on
  sys.path the local file shadowed it. This tries both names.

  A DICT-RETURNING FORWARD. VAE.forward returns a dict of mean, log_sigma, mu,
  logvar and z. torchinfo reports the output shape of the container rather
  than of the reconstruction, which is uninformative in a table. A thin
  wrapper exposes the reconstruction as the traced output, so the printed
  output shape is the image the model actually produces.

  MISSING torchinfo. If it is not installed the script still produces the
  module tree, the component breakdown and the shape trace, which is most of
  what an appendix needs, and says what is missing rather than failing.

  RENDERING WITHOUT AN EXTRA DEPENDENCY. The PNG is drawn with matplotlib,
  which the project already uses, rather than a separate text-to-image module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import count_parameters, load_config      # noqa: E402
try:
    from vae_model import VAE                        # after the rename
except ImportError:                                  # pragma: no cover
    from model import VAE                            # noqa: F401


class ReconOnly(nn.Module):
    """Exposes the reconstruction as the forward output.

    VAE.forward returns a dict, and a traced dict gives a table whose output
    column reads as the container rather than the image. Wrapping it keeps the
    parameter counts identical while making the shapes readable.
    """

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        return self.vae(x)["mean"]


def component_breakdown(model) -> list[tuple[str, int, str]]:
    """Parameter count per named component, with the largest first."""
    rows = []
    for name, module in model.named_children():
        for sub, m in module.named_children():
            n = sum(p.numel() for p in m.parameters() if p.requires_grad)
            if n:
                rows.append((f"{name}.{sub}", n, m.__class__.__name__))
    return sorted(rows, key=lambda r: -r[1])


@torch.no_grad()
def trace_shapes(model, x) -> list[str]:
    """Shape at each stage, taken from the model directly rather than a tracer,
    so it is correct even when torchinfo is unavailable."""
    out = [f"  input                      {tuple(x.shape)}"]
    h = model.encoder.blocks(x)
    out.append(f"  after encoder blocks       {tuple(h.shape)}")
    mu, logvar = model.encoder(x)
    out.append(f"  mu                         {tuple(mu.shape)}")
    out.append(f"  logvar                     {tuple(logvar.shape)}")
    z = model.reparameterize(mu, logvar)
    out.append(f"  z (sampled)                {tuple(z.shape)}")
    mean, log_sigma = model.decoder(z)
    out.append(f"  reconstruction             {tuple(mean.shape)}")
    out.append(f"  log_sigma                  "
               f"{tuple(log_sigma.shape) if log_sigma is not None else 'not predicted'}")
    return out


def render_png(text: str, path: str, font_size: int = 7) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = text.split("\n")
    width = max((len(l) for l in lines), default=80)
    fig_w = min(max(width * font_size * 0.0098, 6), 26)
    fig_h = min(max(len(lines) * font_size * 0.0208, 3), 60)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.text(0.01, 0.99, text, family="DejaVu Sans Mono", fontsize=font_size,
             va="top", ha="left", linespacing=1.32)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(config_path: str, out_path: str, out_png: str, batch: int) -> None:
    cfg = load_config(config_path)
    model = VAE(cfg).eval()

    size = cfg["data"]["image_size"]
    ch = cfg["data"].get("channels", 1)
    input_size = (batch, ch, size, size)
    x = torch.zeros(input_size)

    enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    dec = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    total = count_parameters(model)

    L = []
    L.append("=" * 104)
    L.append("VARIATIONAL AUTOENCODER -- ARCHITECTURE SUMMARY")
    L.append("=" * 104)
    L.append(f"Config                     : {config_path}")
    L.append(f"Experiment                 : {cfg['experiment']['name']}")
    L.append(f"Input traced               : {input_size}")
    L.append(f"Latent                     : {model.latent_kind}, z_dim {model.z_dim}")
    L.append(f"Pre-bottleneck grid        : {model.bottleneck_size} x {model.bottleneck_size} "
             f"x {model.encoder.out_channels}")
    pre = model.bottleneck_size ** 2 * model.encoder.out_channels
    L.append(f"Bottleneck compression     : {pre:,} values -> {model.z_dim} "
             f"({pre / model.z_dim:.0f}:1); against the raw input "
             f"{size*size*ch:,} -> {model.z_dim} ({size*size*ch/model.z_dim:.0f}:1)")
    L.append(f"Per-pixel sigma head       : {'yes' if model.predict_sigma else 'no'}")
    L.append(f"Output activation          : {getattr(model, 'output_activation', 'sigmoid')}")
    L.append("")
    L.append(f"Trainable parameters       : {total:,}")
    L.append(f"  encoder                  : {enc:,}  ({enc/total:.1%})")
    L.append(f"  decoder                  : {dec:,}  ({dec/total:.1%})")
    L.append("")

    L.append("-" * 104)
    L.append("SHAPE TRACE")
    L.append("-" * 104)
    L.extend(trace_shapes(model, x))
    L.append("")

    L.append("-" * 104)
    L.append("PARAMETERS BY COMPONENT (largest first)")
    L.append("-" * 104)
    L.append(f"  {'component':32s} {'type':22s} {'params':>12s}  {'share':>7s}")
    for name, n, kind in component_breakdown(model):
        L.append(f"  {name:32s} {kind:22s} {n:12,}  {n/total:6.1%}")
    L.append("")

    L.append("-" * 104)
    L.append("LAYER-BY-LAYER SUMMARY (torchinfo)")
    L.append("-" * 104)
    try:
        from torchinfo import summary
        result = summary(ReconOnly(model), input_size=input_size,
                         col_names=["input_size", "output_size", "num_params"],
                         depth=4, verbose=0)
        table = str(result)
        L.append(table)
    except ImportError:
        table = ""
        L.append("  torchinfo is not installed, so the layer table is omitted.")
        L.append("  pip install torchinfo    (the sections above do not depend on it)")
    except Exception as exc:                          # noqa: BLE001
        table = ""
        L.append(f"  torchinfo failed: {type(exc).__name__}: {exc}")
        L.append("  The shape trace and component breakdown above are unaffected.")
    L.append("")

    L.append("-" * 104)
    L.append("MODULE TREE (PyTorch native repr)")
    L.append("-" * 104)
    L.append(str(model))

    text = "\n".join(L)
    Path(out_path).write_text(text)
    print(text)
    print(f"\nwritten to {out_path}")

    header = (f"Variational autoencoder, {cfg['experiment']['name']}\n"
              f"input {input_size}, latent z_dim {model.z_dim}, "
              f"{total:,} trainable parameters\n\n")
    render_png(header + (table if table else "\n".join(L[:60])), out_png)
    print(f"rendered to {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Print the VAE architecture")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--out", type=str, default="architecture_summary.txt")
    ap.add_argument("--out-png", type=str, default="architecture_summary.png")
    ap.add_argument("--batch", type=int, default=2,
                    help="Batch size used only for tracing shapes")
    args = ap.parse_args()
    main(args.config, args.out, args.out_png, args.batch)