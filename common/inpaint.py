"""
inpaint.py  --  reconstruction-by-inpainting masking, after RIAD
(Zavrtanik, Kristan and Skocaj, Pattern Recognition 112:107706, 2021).

THE MECHANISM, AND THE ONE WAY IT IS USUALLY BROKEN

A reconstruction detector fails when the model reproduces the anomaly as
readily as the anatomy. Masking removes the region from the input entirely,
so the decoder must predict it from surrounding context learned on healthy
images only. Pathology it has never seen cannot be predicted, and the
residual there grows.

That argument holds ONLY IF THE MASK IS ALSO APPLIED AT TEST TIME. Masking
during training and feeding the whole image at test leaves the lesion
visible to the encoder, the model reproduces it, and the result is identical
to ordinary reconstruction. This is the most common way the method is
implemented into uselessness, so the test path here is built around
disjoint_mask_sets() rather than treated as an afterthought.

HOW THE MASKS ARE ARRANGED

The frame is divided into a grid of k x k patches. A mask SET is a partition
of that grid into n groups, so across the n forward passes every pixel is
masked exactly once and inpainted exactly once. The reconstructed image is
assembled by taking each region from the pass in which it was hidden, which
is what makes the final residual a comparison against content the model
never saw.

THE SIZE TRADE-OFF IS THE REAL HYPERPARAMETER

Larger k makes anomalies harder to reconstruct, which raises their score,
but it makes healthy tissue harder to reconstruct too, which raises the
noise floor. RIAD sweeps k in {2,4,8,16}; at 128 px, k=16 with n=3 is a
reasonable starting point and k=8 a conservative one. Report the value used:
it matters more than any other choice here.
"""

from __future__ import annotations

import torch


def patch_grid(size: int, k: int) -> int:
    if size % k:
        raise ValueError(f"image size {size} is not divisible by patch size {k}")
    return size // k


def random_mask(batch: int, size: int, k: int, frac: float, device,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """Random subset of grid cells per image, value 0 = hidden.

    Kept for comparison, but prefer training_mask(scheme="disjoint"): training
    on random occlusions and testing on disjoint ones is a train/test mismatch
    that costs accuracy for no benefit.
    """
    g = patch_grid(size, k)
    n_cells = g * g
    n_hide = max(1, int(round(n_cells * frac)))
    keep = torch.ones(batch, n_cells, device=device)
    for b in range(batch):
        idx = torch.randperm(n_cells, generator=generator, device=device)[:n_hide]
        keep[b, idx] = 0.0
    return keep.view(batch, 1, g, g).repeat_interleave(k, 2).repeat_interleave(k, 3)


def training_mask(batch: int, size: int, k: int, n: int, device,
                  scheme: str = "disjoint", frac: float = 0.25) -> torch.Tensor:
    """Per-image training mask, matched to the test-time scheme by default.

    "disjoint" draws one of the n test groups at random per image, so the
    occlusion statistics seen during training are exactly those seen at
    evaluation. "random" falls back to an arbitrary subset.
    """
    if scheme != "disjoint":
        return random_mask(batch, size, k, frac, device)
    sets = disjoint_mask_sets(size, k, n, device)
    pick = torch.randint(0, n, (batch,), device=device)
    return torch.cat([sets[int(i)] for i in pick], dim=0)


def apply_mask(x: torch.Tensor, m: torch.Tensor, fill) -> torch.Tensor:
    """Replace hidden pixels with `fill` rather than zero.

    Zero is pure black, far outside the distribution of a chest radiograph,
    so a zero-filled input pushes the encoder into a regime it never meets on
    clean images. Filling with the training mean keeps the input plausible
    and the latent closer to its unmasked value, which is what lets separate
    passes agree with one another.
    """
    if fill is None:
        return x * m
    f = fill if torch.is_tensor(fill) else torch.full_like(x, float(fill))
    return x * m + f.expand_as(x) * (1.0 - m)


def disjoint_mask_sets(size: int, k: int, n: int, device) -> list[torch.Tensor]:
    """Test-time masks: n masks that together hide every pixel exactly once.

    Cells are assigned to groups by a FIXED stride rather than at random, so
    the evaluation is deterministic and two models are compared on identical
    occlusions. Each returned mask is (1,1,size,size) in {0,1}.
    """
    g = patch_grid(size, k)
    rows = torch.arange(g, device=device).view(g, 1)
    cols = torch.arange(g, device=device).view(1, g)
    # A DIAGONAL assignment, not a flat stride over the row-major index. With
    # a flat stride and n dividing the grid width, as 4 divides 8, the group
    # depends only on the column and the mask becomes full-height vertical
    # bars. The model can then interpolate horizontally across them and never
    # needs a model of anatomy, which defeats the method. Offsetting by the
    # row index breaks that aliasing, so every hidden cell has unmasked
    # neighbours above, below and on both sides.
    group = (rows + 2 * cols) % n
    masks = []
    for i in range(n):
        keep = (group != i).float()
        masks.append(keep.view(1, 1, g, g).repeat_interleave(k, 2).repeat_interleave(k, 3))
    return masks


@torch.no_grad()
def inpaint_reconstruct(model, x, k: int, n: int, fill=None):
    """Assemble a full reconstruction from n disjoint-mask passes.

    Every output region is taken from the pass in which that region was
    HIDDEN, so no pixel of the returned image was ever visible to the model
    that produced it. This is the quantity the residual must be computed
    against; reconstructing the unmasked image would defeat the method.
    """
    size = x.shape[-1]
    out = torch.zeros_like(x)
    for m in disjoint_mask_sets(size, k, n, x.device):
        m = m.expand_as(x)
        xh, _ = model.reconstruct(apply_mask(x, m, fill))
        out = out + xh * (1.0 - m)          # keep only the hidden regions
    return out


# --------------------------------------------------------------------------- #
# Evaluation wrapper
# --------------------------------------------------------------------------- #

class InpaintingModel(torch.nn.Module):
    """Wraps a trained VAE so that `reconstruct` inpaints.

    WHY A WRAPPER RATHER THAN PATCHING EACH CALL SITE

    The evaluation path reconstructs in at least four places: fidelity(),
    collect_maps(), score_loader() in scoring.py, and the figure scripts.
    Patching each one is four chances to miss a branch, and a missed branch
    fails SILENTLY: the numbers look plausible but describe ordinary
    reconstruction rather than inpainting, which is the exact failure this
    method is prone to. Overriding reconstruct() once means every caller,
    including modules not edited at all, gets the inpainted output.

    Everything else is forwarded, so the wrapper is transparent to code that
    reads z_dim, latent_kind, encoder or decoder.
    """

    def __init__(self, model, patch: int, test_n: int, fill=None):
        super().__init__()
        self.model = model
        self.patch = int(patch)
        self.test_n = int(test_n)
        self.register_buffer("fill", fill if torch.is_tensor(fill)
                             else torch.tensor(float("nan")), persistent=False)
        self._fill = fill

    @torch.no_grad()
    def reconstruct(self, x, n_samples: int = 1):
        """Assembled from test_n disjoint passes.

        n_samples is accepted and ignored: the passes already average over
        the mask partition, and stochastic sampling on top would make
        separate passes disagree, which is what produced the tiled output in
        the first attempt.

        Returns (mean, None). log_sigma is dropped because a per-pixel
        variance from one pass does not describe the assembled image, and
        pixel_residual() treats None as unnormalised, which is correct here.
        """
        return inpaint_reconstruct(self.model, x, self.patch, self.test_n,
                                   self._fill), None

    def forward(self, *a, **k):
        return self.model(*a, **k)

    def __getattr__(self, name):
        # Only reached for attributes nn.Module does not already resolve.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def wrap_for_eval(model, cfg, train_loader=None, device=None, logger=None):
    """Returns the model unchanged unless training.inpaint.enabled is set.

    The fill value must match training. It is recomputed from the training
    split rather than stored, so a checkpoint evaluated on a different split
    still gets the mean of the images that split actually contains.
    """
    ip = (cfg.get("training", {}) or {}).get("inpaint", {}) or {}
    if not ip.get("enabled", False):
        return model
    fill = None
    if ip.get("fill") == "dataset_mean":
        if train_loader is None:
            raise ValueError("inpaint.fill is dataset_mean but no train_loader "
                             "was given; the fill would differ from training.")
        s, n = None, 0
        for b in train_loader:
            xb = (b["image"] if isinstance(b, dict)
                  else (b[0] if isinstance(b, (list, tuple)) else b))
            s = xb.sum(0) if s is None else s + xb.sum(0)
            n += xb.size(0)
        fill = (s / n).to(device) if device is not None else s / n
    if logger is not None:
        logger.info(f"inpaint scoring: {ip['patch']} px patches, "
                    f"{ip['test_n']} disjoint passes, "
                    f"fill {'training mean' if fill is not None else 'zero'}")
    return InpaintingModel(model, ip["patch"], ip["test_n"], fill).to(device).eval()