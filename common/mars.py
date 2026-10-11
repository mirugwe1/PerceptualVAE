"""
mars.py  --  Mahalanobis Residual Scoring, library.

After MaRS (arXiv 2606.22649, 2026), adapted to an image VAE, with a spatial
extension for localisation that the original paper does not include.

Shared by mars_fit.py, mars_evaluate.py and mars_visualize.py, so the
residual definition, the feature extraction and the Gaussian are defined in
exactly one place and the three scripts cannot disagree about any of them.

THE RESIDUAL

    r(x) = Phi(x) - Phi(x_hat),   x_hat = VAE(x)

Phi is a frozen backbone, its output features left UN-NORMALISED: MaRS
reports that normalising the feature vector suppresses the anisotropy of the
residual distribution, which is precisely what the Mahalanobis metric
exploits. The backbone still applies its own input normalisation.

TWO MODES

global   Each layer is average-pooled to a vector and the vectors are
         concatenated, giving one residual per image. One Gaussian over the
         normals, Ledoit-Wolf shrunk. Image-level detection only. This is
         MaRS as published.

spatial  Layers are resampled to a common grid and concatenated, giving a
         residual at every grid position. One Gaussian PER POSITION, over a
         fixed random subset of channels. Produces a score map, so it
         localises. This is PaDiM applied to the reconstruction residual
         rather than to raw features, and it is the novel part.

WHY THE SPATIAL MODE SHOULD BEAT PADIM, IF IT DOES

PaDiM models raw features, so between-patient variation in habitus and
positioning inflates every position's covariance. That variation is normal,
but it widens the distribution and buries pathology. The VAE's x_hat is THIS
patient's normal anatomy, so subtracting it in feature space removes the
patient-specific normal component and should leave a tighter residual
distribution. Whether it actually does is the experiment.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as Fn


def unpack(b):
    """(image, gt_mask or None) whatever shape the batch arrives in."""
    if isinstance(b, dict):
        return b["image"], b.get("gt_mask")
    if isinstance(b, (list, tuple)):
        return b[0], None
    return b, None


def loaders(dm):
    """Both naming conventions used across the project's datamodules."""
    n = getattr(dm, "normal_loaders", None) or getattr(dm, "get_normal_loaders")
    p = getattr(dm, "pathology_loaders", None) or getattr(dm, "get_pathology_loaders")
    return n(), p()


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

@torch.no_grad()
def global_embed(extractor, x):
    """Average-pooled, concatenated, UN-normalised. Shape (B, D)."""
    return torch.cat([f.mean(dim=(2, 3)) for f in extractor(x)], dim=1)


@torch.no_grad()
def spatial_embed(extractor, x, grid=None):
    """Resampled to the first layer's grid and concatenated. Shape (B, D, g, g).

    The grid follows the backbone rather than a fixed number, so changing
    layers changes the resolution in a predictable way.
    """
    feats = extractor(x)
    g = grid or feats[0].shape[-1]
    return torch.cat([f if f.shape[-1] == g else
                      Fn.interpolate(f, size=(g, g), mode="bilinear", align_corners=False)
                      for f in feats], dim=1), g


# --------------------------------------------------------------------------- #
# Global Gaussian
# --------------------------------------------------------------------------- #

class GlobalGaussian:
    """Mean and Ledoit-Wolf-shrunk precision over residual vectors.

    Shrinkage matters here. With D in the thousands and N in the low
    thousands, the sample covariance is ill-conditioned, and its inverse
    amplifies noise along the SMALLEST eigenvectors, which are exactly the
    directions Mahalanobis weights most heavily. Ledoit-Wolf chooses the
    shrinkage analytically, so there is nothing to tune.
    """

    def __init__(self, mu=None, prec=None, shrinkage=None, eigvals=None):
        self.mu, self.prec, self.shrinkage, self.eigvals = mu, prec, shrinkage, eigvals

    @classmethod
    def fit(cls, X: np.ndarray) -> "GlobalGaussian":
        from sklearn.covariance import LedoitWolf
        mu = X.mean(0)
        lw = LedoitWolf().fit(X - mu)
        # The eigenspectrum is kept for the visualisation: its spread is the
        # anisotropy MaRS exploits, so a flat spectrum predicts no gain over
        # a uniform L2 norm.
        ev = np.linalg.eigvalsh(np.cov((X - mu).T))[::-1]
        return cls(mu, np.linalg.inv(lw.covariance_), float(lw.shrinkage_), ev)

    def score(self, X: np.ndarray) -> np.ndarray:
        d = X - self.mu
        return np.einsum("ij,jk,ik->i", d, self.prec, d)

    def save(self, path, prefix):
        return {f"{prefix}_mu": self.mu, f"{prefix}_prec": self.prec,
                f"{prefix}_shrinkage": np.array(self.shrinkage),
                f"{prefix}_eigvals": self.eigvals}

    @classmethod
    def load(cls, z, prefix):
        return cls(z[f"{prefix}_mu"], z[f"{prefix}_prec"],
                   float(z[f"{prefix}_shrinkage"]), z[f"{prefix}_eigvals"])


# --------------------------------------------------------------------------- #
# Spatial Gaussians
# --------------------------------------------------------------------------- #

class SpatialGaussian:
    """One Gaussian per grid position over a fixed channel subset.

    Accumulated in STREAMING form, holding only the running sum and the
    running sum of outer products, because materialising every residual at
    every position would need several gigabytes. A ridge term rather than
    Ledoit-Wolf, since per-position shrinkage would need the residuals
    stored; with d = 100 and thousands of normals the ridge suffices.
    """

    def __init__(self, idx=None, mu=None, prec=None, grid=None):
        self.idx, self.mu, self.prec, self.grid = idx, mu, prec, grid
        self._S1 = self._S2 = None
        self._n = 0

    def begin(self, idx, grid, device):
        self.idx, self.grid = idx, grid
        P, d = grid * grid, len(idx)
        self._S1 = torch.zeros(P, d, device=device, dtype=torch.float64)
        self._S2 = torch.zeros(P, d, d, device=device, dtype=torch.float64)
        self._n = 0

    def update(self, R):
        """R: residual features (B, D, g, g)."""
        B = R.shape[0]
        z = R[:, self.idx].permute(0, 2, 3, 1).reshape(B, self.grid ** 2, -1).double()
        self._S1 += z.sum(0)
        self._S2 += torch.einsum("bpi,bpj->pij", z, z)
        self._n += B

    def finish(self, eps=0.01):
        n, d = self._n, self._S1.shape[1]
        if n <= d:
            raise ValueError(f"{n} normals for a {d}-dimensional covariance; reduce dim")
        mu = self._S1 / n
        cov = (self._S2 - n * torch.einsum("pi,pj->pij", mu, mu)) / (n - 1)
        cov += eps * torch.eye(d, device=cov.device, dtype=cov.dtype)
        self.mu, self.prec = mu, torch.linalg.inv(cov)
        self._S1 = self._S2 = None
        return n

    @torch.no_grad()
    def score_map(self, R, out_size):
        """Mahalanobis at every position, upsampled. Shape (B, 1, H, W)."""
        B = R.shape[0]
        z = R[:, self.idx].permute(0, 2, 3, 1).reshape(B, self.grid ** 2, -1).double()
        dz = z - self.mu
        m = torch.einsum("bpi,pij,bpj->bp", dz, self.prec, dz).clamp(min=0).sqrt()
        return Fn.interpolate(m.view(B, 1, self.grid, self.grid).float(),
                              size=(out_size, out_size), mode="bilinear",
                              align_corners=False)

    def save(self):
        return {"sp_idx": self.idx.cpu().numpy(), "sp_mu": self.mu.cpu().numpy(),
                "sp_prec": self.prec.cpu().numpy(), "sp_grid": np.array(self.grid)}

    @classmethod
    def load(cls, z, device):
        return cls(torch.as_tensor(z["sp_idx"], device=device),
                   torch.as_tensor(z["sp_mu"], device=device),
                   torch.as_tensor(z["sp_prec"], device=device),
                   int(z["sp_grid"]))
