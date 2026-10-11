"""
fidelity_metrics.py

Reconstruction-quality metrics beyond SSIM and LPIPS, chosen from the
chest-radiograph column of Kastryulin et al.'s benchmark of image-quality
metrics against radiologist judgement.

    from fidelity_metrics import fidelity_metrics
    m = fidelity_metrics(recon, target)      # both (N, 1, H, W) in [0, 1]

WHY BOTHER, GIVEN THIS MEASURES RECONSTRUCTION AND NOT DETECTION

It does not change any detection number. What it does is harden the half of
the argument that rests on fidelity. The central claim is that reconstruction
improved while detection did not, and that claim is only as strong as the
fidelity measurement behind it. Two metrics can disagree by chance; seven
agreeing is evidence.

It also removes an awkward fact. On chest radiographs, ranked by agreement
with radiologist judgement, SSIM places tenth of fourteen and PSNR twelfth.
Those are the two this project has leaned on hardest. LPIPS places third,
which is convenient except that it is approximately what the perceptual
variants optimise, so improving it is weak evidence taken alone. Adding
metrics that are neither weak on this modality nor the training objective is
what makes the fidelity claim defensible.

METRICS AND THEIR RANK ON CHEST X-RAY (PLCC / SRCC)

    HaarPSI   0.83 / 0.61     best on this modality, cheap, grayscale-native
    VSI       0.83 / 0.62     joint best, visual-saliency weighted
    GMSD      0.82 / 0.61     gradient-magnitude similarity, very cheap
    MS-SSIM   0.80 / 0.58     multi-scale SSIM, a fairer SSIM
    FSIM      0.79 / 0.56     phase-congruency based
    DISTS     0.77 / 0.54     deep, texture-tolerant, unlike LPIPS
    SSIM      0.70 / 0.50     retained for continuity with earlier tables
    PSNR      0.66 / 0.48     retained for the same reason

DISTS is worth including specifically because it tolerates texture
resampling where LPIPS penalises it, so the two disagree in an informative
way on blurred reconstructions.

TWO PRACTICAL TRAPS

  MS-SSIM AND IMAGE SIZE. Five scales need 161 pixels, four need 81, three
  need 41. The largest count that fits is chosen automatically and reported as
  ms_ssim_scales. Values at different scale counts are NOT comparable, so a
  64-pixel run and a 128-pixel run cannot share an MS-SSIM column without
  saying which used how many scales.

  CHANNELS. VSI and FSIM expect three channels. Grayscale is repeated rather
  than converted, which is the same convention the perceptual extractors use.
"""

from __future__ import annotations

import torch

# Wang et al.'s five-scale weights. MS-SSIM halves the image at each scale and
# needs an 11-pixel window to survive, so scale n requires 10 * 2^(n-1) + 1
# pixels: 161 for five, 81 for four, 41 for three, 21 for two. The largest
# count that fits is chosen at call time and the weights renormalised.
#
# NOTE THAT MS-SSIM AT DIFFERENT SCALE COUNTS IS NOT COMPARABLE. A 64-pixel run
# uses three scales and a 128-pixel run four, so those two numbers measure
# different things. The count is returned as ms_ssim_scales; report it, and do
# not put MS-SSIM in a table that mixes resolutions without saying so.
_MSSSIM_W = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]


def _msssim_scales(h: int) -> int:
    """Largest number of scales an h-pixel image supports, up to five."""
    n = 0
    while n < 5 and 10 * 2 ** n + 1 <= h:
        n += 1
    return n


def _rgb(t: torch.Tensor) -> torch.Tensor:
    return t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t


@torch.no_grad()
def fidelity_metrics(recon: torch.Tensor, target: torch.Tensor,
                     data_range: float = 1.0, include: list[str] | None = None,
                     verbose: bool = False) -> dict:
    """Reconstruction quality over a batch. Both tensors (N, C, H, W) in [0, 1].

    Every metric is attempted independently and a failure records None with
    the reason rather than aborting the evaluation, because a single metric
    refusing a small image should not cost the whole run.
    """
    try:
        import piq
    except ImportError:
        return {"error": "piq is not installed. pip install piq"}

    x = recon.clamp(0, 1).float()
    y = target.clamp(0, 1).float()
    h = min(x.shape[-2], x.shape[-1])

    metrics = {
        # name: (callable, needs_three_channels)
        "haarpsi":  (lambda a, b: piq.haarpsi(a, b, data_range=data_range), False),
        "gmsd":     (lambda a, b: piq.gmsd(a, b, data_range=data_range), False),
        "vsi":      (lambda a, b: piq.vsi(_rgb(a), _rgb(b), data_range=data_range), True),
        "fsim":     (lambda a, b: piq.fsim(_rgb(a), _rgb(b), data_range=data_range), True),
        "ms_ssim":  (lambda a, b: piq.multi_scale_ssim(
                         a, b, data_range=data_range, scale_weights=torch.tensor(
                             [w / sum(_MSSSIM_W[:_msssim_scales(h)])
                              for w in _MSSSIM_W[:_msssim_scales(h)]],
                             device=a.device)), False),
        "dists":    (lambda a, b: piq.DISTS().to(a.device)(_rgb(a), _rgb(b)), True),
        "ssim":     (lambda a, b: piq.ssim(a, b, data_range=data_range), False),
        "psnr":     (lambda a, b: piq.psnr(a, b, data_range=data_range), False),
    }
    if include:
        metrics = {k: v for k, v in metrics.items() if k in include}

    out: dict[str, object] = {}
    for name, (fn, _) in metrics.items():
        # Two scales is the minimum for MS-SSIM to mean anything at all.
        if name == "ms_ssim" and _msssim_scales(h) < 2:
            out[name] = None
            out[f"{name}_note"] = f"needs >= 21 px for 2 scales, got {h}"
            continue
        try:
            out[name] = float(fn(x, y).mean())
            if name == "ms_ssim":
                out["ms_ssim_scales"] = _msssim_scales(h)
        except Exception as exc:                        # noqa: BLE001
            out[name] = None
            out[f"{name}_note"] = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  {name}: {type(exc).__name__}: {exc}")

    # Direction, so a table can be read without looking metrics up. Every
    # entry here except GMSD is higher-is-better; GMSD is a distance.
    out["_higher_is_better"] = ["haarpsi", "vsi", "fsim", "ms_ssim", "ssim", "psnr"]
    out["_lower_is_better"] = ["gmsd", "dists"]
    return out


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description="Extended fidelity on a checkpoint")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-images", type=int, default=990)
    args = ap.parse_args()

    from utils import build_datamodule, get_device, load_checkpoint, load_config, set_seed
    try:
        from vae_model import VAE
    except ImportError:
        from model import VAE

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"].get("device", "cuda"))
    model = VAE(cfg).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, device)["model_state"])
    model.eval()

    _, _, test = build_datamodule(cfg).normal_loaders()
    xs, rs, n = [], [], 0
    with torch.no_grad():
        for b in test:
            x = b["image"].to(device)
            r, _ = model.reconstruct(x, n_samples=1)
            xs.append(x.cpu()); rs.append(r.cpu()); n += x.size(0)
            if n >= args.max_images:
                break
    x = torch.cat(xs)[:args.max_images]
    r = torch.cat(rs)[:args.max_images]

    m = fidelity_metrics(r, x, verbose=True)
    print(f"\n{cfg['experiment']['name']}, {len(x)} held-out normal images\n")
    hi = set(m.get("_higher_is_better", []))
    for k, v in m.items():
        if k.startswith("_") or k.endswith("_note"):
            continue
        arrow = "higher better" if k in hi else "lower better"
        print(f"  {k:10s} {v:10.4f}   {arrow}" if v is not None
              else f"  {k:10s} {'n/a':>10s}   {m.get(k + '_note', '')}")