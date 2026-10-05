"""Rasterisation constants shared by the NumPy reference and the PyTorch twin.

The default values are the ones used by Kerbl et al., "3D Gaussian Splatting for Real-Time
Radiance Field Rendering", ACM TOG 42(4), 2023.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Zeroth real spherical harmonic, Y_0^0 = 1 / (2 sqrt(pi)). A degree-0 coefficient ``f`` encodes
#: the view-independent colour ``0.5 + SH_C0 * f``.
SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class RasterSettings:
    """Numerical constants of the compositing equation.

    Attributes:
        alpha_min: A Gaussian contributes to a pixel only where its alpha reaches this value
            (1/255, one 8-bit grey level). Set it to 0 to obtain a smooth function of the
            parameters, which is what the finite-difference gradient checks use.
        alpha_max: Upper clamp of the per-Gaussian alpha, so that the transmittance never
            reaches exactly zero.
        low_pass: Variance, in pixel^2, added to both diagonal terms of every projected
            covariance. It is the screen-space reconstruction filter of EWA splatting and
            keeps each footprint at least about one pixel wide.
        near: Gaussians whose centre is closer than this camera-space depth are discarded.
    """

    alpha_min: float = 1.0 / 255.0
    alpha_max: float = 0.99
    low_pass: float = 0.3
    near: float = 0.05
