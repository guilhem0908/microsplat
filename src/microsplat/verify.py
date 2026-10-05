"""Numerical cross-checks, each returning the error it measured.

The test suite asserts bounds on these values and ``scripts/reproduce.py`` writes them to
``results/verification.json``, so the numbers quoted in the README are the tested ones.
"""

from __future__ import annotations

import numpy as np
import torch

from microsplat import reference, torch_renderer
from microsplat.camera import Camera, look_at
from microsplat.gaussians import GaussianCloud
from microsplat.settings import RasterSettings

FINITE_DIFFERENCE_STEP = 1e-6

# Bounds that the test suite asserts on the quantities measured below.
JACOBIAN_TOLERANCE = 1e-8
MONTE_CARLO_TOLERANCE = 0.01  # for the smallest Gaussian, where sampling noise dominates
PARITY_FLOAT64_TOLERANCE = 1e-10
PARITY_FLOAT32_TOLERANCE = 1e-5
GRADIENT_TOLERANCE = 1e-6

CHECK_BACKGROUND = (0.1, 0.2, 0.3)
PARITY_BUFFERS = ("color", "depth", "alpha")
#: Centre of the Gaussian used by the Monte-Carlo check, about 3.4 units from the camera.
MONTE_CARLO_MEAN = (0.2, -0.3, 0.1)


def check_camera(width: int, height: int) -> Camera:
    """An oblique camera three and a half units away from the origin."""
    return look_at(
        np.array([3.0, 1.0, 1.5]), np.zeros(3), width=width, height=height, fov_x_deg=50.0
    )


def random_scene(count: int, seed: int, max_scale: float = 0.2) -> GaussianCloud:
    """Random anisotropic Gaussians inside the cube [-1, 1]^3."""
    rng = np.random.default_rng(seed)
    return GaussianCloud.from_activated(
        means=rng.uniform(-1.0, 1.0, (count, 3)),
        scales=rng.uniform(0.25 * max_scale, max_scale, (count, 3)),
        quats=rng.normal(size=(count, 4)),
        opacities=rng.uniform(0.05, 0.95, count),
        colours=rng.uniform(0.0, 1.0, (count, 3)),
    )


def jacobian_error(seed: int = 0, points: int = 200) -> float:
    """Largest relative gap between the analytic projection Jacobian and central differences."""
    cam = check_camera(640, 480)
    rng = np.random.default_rng(seed)
    t = rng.uniform([-1.0, -1.0, 2.0], [1.0, 1.0, 6.0], size=(points, 3))
    analytic = reference.perspective_jacobian(t, cam.fx, cam.fy)

    def pinhole(p: np.ndarray) -> np.ndarray:
        return np.stack([cam.fx * p[:, 0] / p[:, 2], cam.fy * p[:, 1] / p[:, 2]], axis=1)

    numeric = np.empty_like(analytic)
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = FINITE_DIFFERENCE_STEP
        numeric[:, :, axis] = (pinhole(t + delta) - pinhole(t - delta)) / (
            2.0 * FINITE_DIFFERENCE_STEP
        )
    return float(np.abs(analytic - numeric).max() / np.abs(analytic).max())


def monte_carlo_covariance_error(scale: float, samples: int, seed: int = 0) -> float:
    """Relative Frobenius gap between the EWA covariance and a sampled one.

    One anisotropic Gaussian with largest standard deviation ``scale`` is sampled in 3D, every
    sample goes through the exact pinhole projection, and the sample covariance of the image
    points is compared with ``J W Sigma W^T J^T``. The viewing distance is about 3.4, so
    ``scale / 3.4`` is the relative size of the Gaussian.
    """
    cam = check_camera(640, 480)
    cloud = GaussianCloud.from_activated(
        means=np.array([MONTE_CARLO_MEAN]),
        scales=scale * np.array([[1.0, 0.4, 0.7]]),
        quats=np.array([[0.9, 0.2, -0.3, 0.1]]),
        opacities=np.array([0.5]),
        colours=np.full((1, 3), 0.5),
    )
    ewa = reference.project_gaussians(cloud, cam, low_pass=0.0).cov2d[0]
    cov3d = reference.covariance_3d(cloud.log_scales, cloud.quats)[0]
    points = np.random.default_rng(seed).multivariate_normal(cloud.means[0], cov3d, size=samples)
    uv, _ = cam.project(points)
    sampled = np.cov(uv.T)
    return float(np.linalg.norm(ewa - sampled) / np.linalg.norm(sampled))


def parity_errors(
    count: int,
    seed: int,
    dtype: torch.dtype,
    *,
    width: int = 100,
    height: int = 72,
    device: str = "cpu",
    tile_size: int = torch_renderer.DEFAULT_TILE_SIZE,
    chunk_elements: int = torch_renderer.DEFAULT_CHUNK_ELEMENTS,
    settings: RasterSettings | None = None,
) -> dict[str, float | bool]:
    """Largest absolute difference between the two renderers on one random scene.

    ``same_order`` tells whether both renderers sorted the Gaussians identically. In float32
    two centres can be closer in depth than the format resolves; they are then composited in
    the other order, which changes the colour of the pixels they share but not the
    accumulated opacity, because the product of the ``1 - alpha`` terms does not depend on
    the order.
    """
    cam = check_camera(width, height)
    cloud = random_scene(count, seed)
    settings = settings or RasterSettings()
    expected = reference.render(cloud, cam, np.array(CHECK_BACKGROUND), settings)
    params = torch_renderer.cloud_to_params(cloud, dtype=dtype, device=device)
    with torch.no_grad():
        actual = torch_renderer.render(
            params,
            cam,
            torch.tensor(CHECK_BACKGROUND, dtype=torch.float64),
            settings,
            tile_size=tile_size,
            chunk_elements=chunk_elements,
        )
        twin_depth = torch_renderer.project_gaussians(params, cam, settings)[2]
    twin_order = torch.argsort(twin_depth, stable=True).cpu().numpy()
    reference_order = np.argsort(cam.project(cloud.means)[1], kind="stable")
    errors: dict[str, float | bool] = {
        name: float(np.abs(getattr(actual, name).cpu().numpy() - getattr(expected, name)).max())
        for name in PARITY_BUFFERS
    }
    errors["same_order"] = bool(np.array_equal(twin_order, reference_order))
    return errors


def gradient_errors(alpha_min: float, seed: int = 0, count: int = 6) -> dict[str, float]:
    """Autograd versus central differences, per parameter group, in float64.

    The scalar being differentiated is a fixed random linear functional of the colour, depth
    and opacity buffers of a 12 x 10 render, so every output takes part. Each value returned is
    ``max |autograd - numeric| / max |autograd|`` over the entries of one parameter group.
    """
    cam = check_camera(12, 10)
    settings = RasterSettings(alpha_min=alpha_min)
    cloud = random_scene(count, seed, max_scale=0.5)
    background = torch.tensor(CHECK_BACKGROUND, dtype=torch.float64)
    weights = torch.rand(
        (cam.height, cam.width, 5),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(seed),
    )

    def functional(params: dict[str, torch.Tensor]) -> torch.Tensor:
        out = torch_renderer.render(params, cam, background, settings, tile_size=4)
        buffers = torch.cat([out.color, out.depth[..., None], out.alpha[..., None]], dim=-1)
        return (buffers * weights).sum()

    params = torch_renderer.cloud_to_params(cloud, dtype=torch.float64, requires_grad=True)
    functional(params).backward()

    errors = {}
    with torch.no_grad():
        for name, value in params.items():
            numeric = torch.zeros_like(value)
            flat, numeric_flat = value.view(-1), numeric.view(-1)
            for k in range(flat.numel()):
                original = flat[k].item()
                flat[k] = original + FINITE_DIFFERENCE_STEP
                upper = functional(params)
                flat[k] = original - FINITE_DIFFERENCE_STEP
                lower = functional(params)
                flat[k] = original
                numeric_flat[k] = (upper - lower) / (2.0 * FINITE_DIFFERENCE_STEP)
            errors[name] = float((value.grad - numeric).abs().max() / value.grad.abs().max())
    return errors
