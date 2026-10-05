"""The Gaussian scene container and its parameterisation.

A scene is a set of N anisotropic 3D Gaussians. Each one stores the *unconstrained* values the
optimiser works on (Kerbl et al. 2023, section 4):

===============  =======  ==================================================
field            shape    activated quantity
===============  =======  ==================================================
means            (N, 3)   centre ``mu``
log_scales       (N, 3)   standard deviations ``s = exp(log_scales)``
quats            (N, 4)   rotation, ``q / |q|`` with ``q = (w, x, y, z)``
opacity_logits   (N,)     opacity ``o = sigmoid(logit)``
sh_dc            (N, 3)   colour ``c = max(0, 0.5 + SH_C0 * sh_dc)``
===============  =======  ==================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from microsplat.settings import SH_C0

#: Number of nearest neighbours whose mean distance sets the initial scale (Kerbl et al. 2023,
#: section 5.1: "the mean of the distance to the closest three points").
INIT_NEIGHBOURS = 3
INIT_OPACITY = 0.1


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """Degree-0 spherical-harmonic coefficient that decodes to ``rgb``."""
    return (rgb - 0.5) / SH_C0


def sh_dc_to_rgb(sh_dc: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 0.5 + SH_C0 * sh_dc)


@dataclass
class GaussianCloud:
    """N Gaussians stored as float64 NumPy arrays of unconstrained parameters."""

    means: np.ndarray
    log_scales: np.ndarray
    quats: np.ndarray
    opacity_logits: np.ndarray
    sh_dc: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.means)
        expected = {
            "means": (count, 3),
            "log_scales": (count, 3),
            "quats": (count, 4),
            "opacity_logits": (count,),
            "sh_dc": (count, 3),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            setattr(self, name, value)

    def __len__(self) -> int:
        return len(self.means)

    @classmethod
    def from_activated(
        cls,
        means: np.ndarray,
        scales: np.ndarray,
        quats: np.ndarray,
        opacities: np.ndarray,
        colours: np.ndarray,
    ) -> GaussianCloud:
        """Build a cloud from human-readable values (scales, opacities in (0, 1), RGB)."""
        return cls(
            means=means,
            log_scales=np.log(scales),
            quats=quats,
            opacity_logits=logit(np.asarray(opacities, dtype=np.float64)),
            sh_dc=rgb_to_sh_dc(np.asarray(colours, dtype=np.float64)),
        )

    @property
    def scales(self) -> np.ndarray:
        return np.exp(self.log_scales)

    @property
    def opacities(self) -> np.ndarray:
        return sigmoid(self.opacity_logits)

    @property
    def colours(self) -> np.ndarray:
        return sh_dc_to_rgb(self.sh_dc)

    def permuted(self, order: np.ndarray) -> GaussianCloud:
        """The same scene with its Gaussians stored in another order."""
        return GaussianCloud(
            self.means[order],
            self.log_scales[order],
            self.quats[order],
            self.opacity_logits[order],
            self.sh_dc[order],
        )


def random_cloud(
    count: int, box_min: np.ndarray, box_max: np.ndarray, rng: np.random.Generator
) -> GaussianCloud:
    """Initial guess for an optimisation: grey isotropic Gaussians at uniform random positions.

    Kerbl et al. start from a structure-from-motion point cloud, or from uniformly random points
    inside the scene bounds when none is available; this is the second case. Each scale is the
    mean distance to the three nearest neighbours, so that neighbouring Gaussians overlap.
    """
    box_min = np.asarray(box_min, dtype=np.float64)
    box_max = np.asarray(box_max, dtype=np.float64)
    means = rng.uniform(box_min, box_max, size=(count, 3))
    squared = ((means[:, None, :] - means[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(squared, np.inf)
    nearest = np.sqrt(np.partition(squared, INIT_NEIGHBOURS - 1, axis=1)[:, :INIT_NEIGHBOURS])
    scale = nearest.mean(axis=1, keepdims=True)
    quats = np.zeros((count, 4))
    quats[:, 0] = 1.0
    return GaussianCloud(
        means=means,
        log_scales=np.log(np.repeat(scale, 3, axis=1)),
        quats=quats,
        opacity_logits=np.full(count, logit(np.float64(INIT_OPACITY))),
        sh_dc=np.zeros((count, 3)),
    )
