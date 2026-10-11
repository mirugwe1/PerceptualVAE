"""
model.py

Convolutional VAE for unsupervised anomaly detection. 128x128 grayscale in,
128x128 out, with a global 1x1x128 latent.

    ENCODER                                        params
      ConvBlock 1   128x128x1   -> 64x64x32          576
      ConvBlock 2   64x64x32    -> 32x32x64       32,896
      ConvBlock 3   32x32x64    -> 16x16x128     131,328
      ConvBlock 4   16x16x128   -> 8x8x256       524,800
      ConvBlock 5   8x8x256     -> 4x4x256     1,049,088
      Conv mu       4x4x256     -> 1x1x128       524,288
      Conv logvar   4x4x256     -> 1x1x128       524,288

    BOTTLENECK
      z in R^128, sampled as mu + exp(0.5 * logvar) * eps

    DECODER
      ConvT launch  1x1x128     -> 4x4x256       524,288
      DeconvBlock 1 4x4x256     -> 8x8x256     1,049,088
      DeconvBlock 2 8x8x256     -> 16x16x128     524,544
      DeconvBlock 3 16x16x128   -> 32x32x64      131,200
      DeconvBlock 4 32x32x64    -> 64x64x32       32,832
      ConvT head    64x64x32    -> 128x128x1         513

Each ConvBlock is Conv2d(k4, s2, p1) then InstanceNorm2d then LeakyReLU.
Each DeconvBlock is the transpose of that. Total 5,049,729 parameters.

-------------------------------------------------------------------------
WHY THIS SHAPE
-------------------------------------------------------------------------

FIVE STAGES TO 4x4, NOT FOUR TO 8x8. The collapse convolution appears three
times (mu, logvar, and the decoder launch) and its cost scales with the square
of the remaining grid. Stopping at 8x8 needs a k8 kernel and puts 82 percent
of the whole model into those three layers; stopping at 4x4 needs k4 and puts
31 percent there. The 4x4 route also destroys less in one step: 4,096 values
into 128 is 32:1, against 128:1 from 8x8.

z = 128, NOT 256. The bottleneck is the mechanism the detector depends on. An
autoencoder that can copy its input reconstructs pathology as happily as
health and its residual separates nothing. Measured on this project, the
variant with the WORST reconstructions detected best and the one with the best
reconstructions detected worst, so the tighter latent is the safer default.
At z=256 the per-dimension KL previously settled near 2.35 nats against a 0.5
free-bits floor, so halving still leaves ample capacity.

CAPACITY. 5,049,729 parameters against roughly 7,000 training normals, about
713 per image. The previous 7,694,498-parameter model showed a
validation-training gap of +0.27 on a LARGER training set, and an overfitting
VAE reconstructs pathology about as well as health.

SEPARATE mu AND logvar CONVOLUTIONS. Each reads the full 4x4x256 tensor
independently. A shared trunk followed by a linear split would cap the rank of
the combined map at the trunk width, which is a real loss of expressiveness
for no saving worth having.

INSTANCENORM, NOT BATCHNORM. Like GroupNorm and unlike BatchNorm, it
normalises within a sample, so a sample's anomaly score never depends on which
other images happened to share its batch. That is a correctness property for a
per-sample score, not merely a stability one.

BOUNDED OUTPUT BY DEFAULT. The data lies in [0, 1] and the rest of the
pipeline assumes it: SSIM uses data_range 1.0, LPIPS expects [0, 1], the
perceptual backbones normalise from it, and visualize.py draws with vmin 0 and
vmax 1. An unbounded output forces a clamping decision inside each of those,
and every such decision silently changes what is being measured. Set
model.output_activation: identity for the unbounded form.

TRANSPOSED CONVOLUTIONS IN THE DECODER. This mirrors the reference design.
They can leave periodic artifacts, and those artifacts land in the residual
map the anomaly score reads, so it is worth looking for a grid pattern in the
qualitative figures. If one appears, model.decoder_upsample: resize switches
to nearest-neighbour upsampling followed by a 3x3 convolution, which cannot
produce it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #

class ConvBlock(nn.Module):
    """Conv2d(k4, s2, p1) -> InstanceNorm2d -> LeakyReLU. Halves the grid."""

    def __init__(self, cin: int, cout: int, slope: float = 0.2,
                 norm: bool = True, act: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=not norm)
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(cout, affine=True))
        if act:
            layers.append(nn.LeakyReLU(slope, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """The transpose of ConvBlock. Doubles the grid.

    `mode="resize"` swaps the transposed convolution for nearest-neighbour
    upsampling followed by a 3x3 convolution, which cannot produce the
    periodic artifacts a transposed convolution can.
    """

    def __init__(self, cin: int, cout: int, slope: float = 0.2,
                 norm: bool = True, act: bool = True, mode: str = "transpose"):
        super().__init__()
        if mode == "transpose":
            up: nn.Module = nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2,
                                               padding=1, bias=not norm)
        elif mode == "resize":
            up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=not norm),
            )
        else:
            raise ValueError(f"decoder_upsample must be 'transpose' or 'resize', got {mode!r}")

        layers: list[nn.Module] = [up]
        if norm:
            layers.append(nn.InstanceNorm2d(cout, affine=True))
        if act:
            layers.append(nn.LeakyReLU(slope, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# --------------------------------------------------------------------------- #
# Encoder / decoder
# --------------------------------------------------------------------------- #

class Encoder(nn.Module):
    def __init__(self, in_ch, channels, z_dim, grid, slope, spatial_channels=None):
        super().__init__()
        blocks, cin = [], in_ch
        for cout in channels:
            blocks.append(ConvBlock(cin, cout, slope))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = cin
        self.grid = grid

        # Two independent heads, each reading the whole pre-collapse tensor.
        # kernel == stride == the remaining grid, so 4x4 -> 1x1 in one learned
        # step rather than by averaging, which would discard position.
        # GLOBAL: kernel and stride equal the remaining grid, collapsing it to a
        # single point, so all spatial information is discarded and the decoder
        # must invent placement from a flat vector.
        #
        # SPATIAL: a 1x1 convolution, so the grid is preserved and each latent
        # position describes its own patch. Same parameter budget per position,
        # far more total capacity, and crucially the decoder is told WHERE
        # things are rather than having to reconstruct arrangement from scratch.
        self.spatial = spatial_channels is not None
        k = 1 if self.spatial else grid
        st = 1 if self.spatial else grid
        out_ch = spatial_channels if self.spatial else z_dim
        self.to_mu = nn.Conv2d(cin, out_ch, kernel_size=k, stride=st, bias=False)
        self.to_logvar = nn.Conv2d(cin, out_ch, kernel_size=k, stride=st, bias=False)

    def forward(self, x):
        h = self.blocks(x)
        mu = self.to_mu(h).flatten(1)
        logvar = self.to_logvar(h).flatten(1)
        # exp(0.5 * 8) is a standard deviation of about 55. Beyond that is a
        # diverging run, not a wide posterior.
        return mu, logvar.clamp(-8.0, 8.0)


class Decoder(nn.Module):
    def __init__(self, out_ch, channels, z_dim, grid, slope, output_activation,
                 upsample_mode, predict_sigma, spatial_channels=None):
        super().__init__()
        self.z_dim = z_dim
        self.spatial = spatial_channels is not None
        self.spatial_channels = spatial_channels
        self.grid = grid
        rev = list(reversed(channels))          # [256, 256, 128, 64, 32]

        # Mirror of the encoder collapse. Spatial keeps the grid, so a 1x1
        # convolution is all that is needed to widen back to the decoder trunk.
        launch_conv = (nn.Conv2d(spatial_channels, rev[0], kernel_size=1, bias=False)
                       if self.spatial else
                       nn.ConvTranspose2d(z_dim, rev[0], kernel_size=grid,
                                          stride=grid, bias=False))
        self.launch = nn.Sequential(
            launch_conv,
            nn.InstanceNorm2d(rev[0], affine=True),
            nn.LeakyReLU(slope, inplace=True),
        )
        self.blocks = nn.Sequential(*[
            DeconvBlock(a, b, slope, mode=upsample_mode) for a, b in zip(rev[:-1], rev[1:])
        ])
        # Final upsample to full resolution. No normalization, no activation:
        # the output head must be free to reach the extremes of the range.
        self.mean_head = DeconvBlock(rev[-1], out_ch, mode=upsample_mode,
                                     norm=False, act=False)
        self.sigma_head = (DeconvBlock(rev[-1], out_ch, mode=upsample_mode,
                                       norm=False, act=False)
                           if predict_sigma else None)
        self.output_activation = output_activation

    def forward(self, z):
        # z arrives flat in both modes so that every downstream tool, the
        # dashboards and diagnostics included, can treat the latent as a vector
        # of length z_dim without knowing which mode is in use.
        if self.spatial:
            h = self.launch(z.view(-1, self.spatial_channels, self.grid, self.grid))
        else:
            h = self.launch(z.view(-1, self.z_dim, 1, 1))
        h = self.blocks(h)
        mean = self.mean_head(h)
        if self.output_activation == "sigmoid":
            mean = torch.sigmoid(mean)
        log_sigma = self.sigma_head(h) if self.sigma_head is not None else None
        return mean, log_sigma


# --------------------------------------------------------------------------- #
# VAE
# --------------------------------------------------------------------------- #

class VAE(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m, d = cfg["model"], cfg["data"]
        self.image_size = d["image_size"]
        self.in_channels = d.get("channels", 1)

        self.channels = list(m.get("channels", [32, 64, 128, 256, 256]))
        slope = m.get("leaky_slope", 0.2)
        self.output_activation = m.get("output_activation", "sigmoid")
        if self.output_activation not in ("sigmoid", "identity"):
            raise ValueError(
                f"model.output_activation must be 'sigmoid' or 'identity', "
                f"got {self.output_activation!r}"
            )
        upsample_mode = m.get("decoder_upsample", "transpose")

        lat = m.get("latent", {})
        self.latent_kind = lat.get("kind", "global")
        if self.latent_kind not in ("global", "spatial"):
            raise ValueError(f"latent.kind must be global or spatial, "
                             f"got {self.latent_kind!r}")
        self.spatial_channels = (int(lat["channels"]) if self.latent_kind == "spatial"
                                 else None)

        self.predict_sigma = m.get("predict_sigma", False)
        self.sigma_min = m.get("sigma_min", 0.01)
        self.sigma_max = m.get("sigma_max", 1.0)

        # Grid remaining after the stride-2 stack; the collapse kernel equals it.
        self.bottleneck_size = self.image_size // (2 ** len(self.channels))
        # z_dim is the total number of latent values either way, so compression
        # ratios, KL-per-dimension and every diagnostic stay comparable between
        # the two modes.
        if self.latent_kind == "spatial":
            self.z_dim = self.spatial_channels * self.bottleneck_size ** 2
        else:
            self.z_dim = lat.get("z_dim", 128)
        if self.bottleneck_size < 1:
            raise ValueError(
                f"image_size {self.image_size} with {len(self.channels)} stride-2 "
                f"stages collapses below one pixel. Shorten model.channels."
            )
        if self.image_size % (2 ** len(self.channels)) != 0:
            raise ValueError(
                f"image_size {self.image_size} is not divisible by "
                f"2^{len(self.channels)}; the decoder could not mirror the encoder."
            )

        self.encoder = Encoder(self.in_channels, self.channels, self.z_dim,
                               self.bottleneck_size, slope, self.spatial_channels)
        self.decoder = Decoder(self.in_channels, self.channels, self.z_dim,
                               self.bottleneck_size, slope,
                               self.output_activation, upsample_mode,
                               self.predict_sigma, self.spatial_channels)

    # ------------------------------------------------------------------ #

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def clamp_sigma(self, log_sigma):
        """Bound sigma away from zero. Without a floor the Gaussian likelihood
        can drive it toward zero on easy pixels and take the loss to minus
        infinity; the floor also stops the uncertainty head explaining away
        real anomalies by declaring them unpredictable."""
        return log_sigma.clamp(math.log(self.sigma_min), math.log(self.sigma_max))

    def forward(self, x, sample: bool = True):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        mean, log_sigma = self.decoder(z)
        if mean.shape[-1] != x.shape[-1]:            # only if the maths drifts
            mean = F.interpolate(mean, size=x.shape[-2:], mode="bilinear",
                                 align_corners=False)
        if log_sigma is not None:
            log_sigma = self.clamp_sigma(log_sigma)
        return {"mean": mean, "log_sigma": log_sigma, "mu": mu, "logvar": logvar, "z": z}

    @torch.no_grad()
    def reconstruct(self, x, n_samples: int = 1):
        """Deterministic by default: decode from mu, the standard VAE
        evaluation convention. n_samples > 1 averages several posterior draws,
        which smooths the residual map and, measured on this pipeline, costs
        the fine poolings more than it gains the coarse ones."""
        self.eval()
        if n_samples <= 1:
            out = self.forward(x, sample=False)
            return out["mean"], out["log_sigma"]
        means, sigmas = [], []
        for _ in range(n_samples):
            out = self.forward(x, sample=True)
            means.append(out["mean"])
            if out["log_sigma"] is not None:
                sigmas.append(out["log_sigma"])
        return (torch.stack(means).mean(0),
                torch.stack(sigmas).mean(0) if sigmas else None)
