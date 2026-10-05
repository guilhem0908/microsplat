"""NumPy reference rasteriser: the specification the PyTorch twin is tested against.

It favours being checkable over being fast: no tiles, no culling beyond the near plane, one
explicit loop over the depth-sorted Gaussians. Every step is one equation of

* Zwicker et al., "EWA Splatting", IEEE TVCG 8(3), 2002 (projection of a 3D Gaussian), and
* Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", ACM TOG 42(4),
  2023 (parameterisation and compositing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from microsplat.camera import Camera
from microsplat.gaussians import GaussianCloud
from microsplat.settings import RasterSettings


@dataclass
class Projection:
    """Screen-space footprint of every Gaussian for one camera."""

    means2d: np.ndarray  # (N, 2) image coordinates of the centres
    cov2d: np.ndarray  # (N, 2, 2) projected covariances, in pixel^2
    depth: np.ndarray  # (N,) camera-space z of the centres


@dataclass
class Render:
    """Output buffers of one rendered view."""

    color: np.ndarray  # (H, W, 3) composited colour including the background
    depth: np.ndarray  # (H, W) alpha-weighted depth, sum_i w_i z_i (0 where nothing is hit)
    alpha: np.ndarray  # (H, W) accumulated opacity, 1 - final transmittance


def quat_to_rotmat(quats: np.ndarray) -> np.ndarray:
    """Rotation matrices (N, 3, 3) of (N, 4) quaternions ``(w, x, y, z)``, normalised here."""
    q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.empty((len(q), 3, 3))
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - w * z)
    rot[:, 0, 2] = 2.0 * (x * z + w * y)
    rot[:, 1, 0] = 2.0 * (x * y + w * z)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - w * x)
    rot[:, 2, 0] = 2.0 * (x * z - w * y)
    rot[:, 2, 1] = 2.0 * (y * z + w * x)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def covariance_3d(log_scales: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """World-space covariances ``Sigma = R S S^T R^T`` (Kerbl et al. 2023, eq. 6).

    ``S = diag(exp(log_scales))`` and ``R`` is the rotation of the quaternion, so ``Sigma`` is
    symmetric positive definite for any value of the unconstrained parameters.
    """
    rot = quat_to_rotmat(quats)
    m = rot * np.exp(log_scales)[:, None, :]  # R S: scale the columns of R
    return m @ m.transpose(0, 2, 1)


def perspective_jacobian(cam_points: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Jacobian (N, 2, 3) of the pinhole projection at camera-space points ``t``.

    With ``u = fx t_x / t_z + cx`` and ``v = fy t_y / t_z + cy``::

        J = [[fx / t_z, 0,        -fx t_x / t_z^2],
             [0,        fy / t_z, -fy t_y / t_z^2]]

    This is the matrix of the local affine approximation of EWA splatting (Zwicker et al.
    2002), written for a pinhole camera in pixel units.
    """
    tx, ty, tz = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
    jac = np.zeros((len(cam_points), 2, 3))
    jac[:, 0, 0] = fx / tz
    jac[:, 0, 2] = -fx * tx / tz**2
    jac[:, 1, 1] = fy / tz
    jac[:, 1, 2] = -fy * ty / tz**2
    return jac


def project_gaussians(cloud: GaussianCloud, camera: Camera, low_pass: float) -> Projection:
    """Project every Gaussian to the image plane (Kerbl et al. 2023, eq. 5).

    A Gaussian stays a Gaussian under an affine map, so the perspective projection is replaced
    by its first-order expansion at the centre. With ``W`` the world-to-camera rotation and
    ``J`` the projection Jacobian::

        Sigma' = J W Sigma W^T J^T + low_pass * I

    Centres at or behind the image plane give non-finite values here; :func:`render` never
    reads them because it skips every Gaussian closer than the near plane.
    """
    cam_points = camera.world_to_camera(cloud.means)
    means2d, depth = camera.project(cloud.means)
    jac = perspective_jacobian(cam_points, camera.fx, camera.fy)
    jw = jac @ camera.rotation
    cov2d = jw @ covariance_3d(cloud.log_scales, cloud.quats) @ jw.transpose(0, 2, 1)
    cov2d = cov2d + low_pass * np.eye(2)
    return Projection(means2d=means2d, cov2d=cov2d, depth=depth)


def render(
    cloud: GaussianCloud,
    camera: Camera,
    background: np.ndarray,
    settings: RasterSettings | None = None,
) -> Render:
    """Render one view by front-to-back alpha compositing (Kerbl et al. 2023, eq. 3).

    Gaussians are visited by increasing depth of their centre. For the pixel at ``x``::

        alpha_i = min(alpha_max, o_i * exp(-0.5 (x - mu'_i)^T Sigma'_i^-1 (x - mu'_i)))
        alpha_i = 0                       where alpha_i < alpha_min
        C       = sum_i c_i alpha_i T_i + T_end * background,   T_i = prod_{j<i} (1 - alpha_j)
    """
    settings = settings or RasterSettings()
    with np.errstate(divide="ignore", invalid="ignore"):
        proj = project_gaussians(cloud, camera, settings.low_pass)
    opacities = cloud.opacities
    colours = cloud.colours
    px, py = camera.pixel_centres()

    color = np.zeros((camera.height, camera.width, 3))
    depth = np.zeros((camera.height, camera.width))
    transmittance = np.ones((camera.height, camera.width))

    for i in np.argsort(proj.depth, kind="stable"):
        if not proj.depth[i] > settings.near:
            continue
        (a, b), (_, c) = proj.cov2d[i]
        det = a * c - b * b
        dx = px - proj.means2d[i, 0]
        dy = py - proj.means2d[i, 1]
        # Mahalanobis distance with the inverse covariance [[c, -b], [-b, a]] / det.
        mahalanobis = (c * dx * dx - 2.0 * b * dx * dy + a * dy * dy) / det
        alpha = np.minimum(settings.alpha_max, opacities[i] * np.exp(-0.5 * mahalanobis))
        alpha[alpha < settings.alpha_min] = 0.0
        weight = alpha * transmittance
        color += weight[..., None] * colours[i]
        depth += weight * proj.depth[i]
        transmittance *= 1.0 - alpha

    color += transmittance[..., None] * np.asarray(background, dtype=np.float64)
    return Render(color=color, depth=depth, alpha=1.0 - transmittance)
