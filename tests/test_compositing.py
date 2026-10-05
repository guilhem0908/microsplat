"""Invariants of the front-to-back compositing in the NumPy reference."""

from __future__ import annotations

import numpy as np

from microsplat import reference, verify
from microsplat.camera import look_at
from microsplat.gaussians import GaussianCloud
from microsplat.settings import RasterSettings

SMOOTH = RasterSettings(alpha_min=0.0)


def _isotropic(position, sigma: float, opacity: float, colour) -> GaussianCloud:
    return GaussianCloud.from_activated(
        means=np.array([position], dtype=float),
        scales=np.full((1, 3), sigma),
        quats=np.array([[1.0, 0.0, 0.0, 0.0]]),
        opacities=np.array([opacity]),
        colours=np.array([colour], dtype=float),
    )


def _concat(*clouds: GaussianCloud) -> GaussianCloud:
    fields = ("means", "log_scales", "quats", "opacity_logits", "sh_dc")
    return GaussianCloud(*(np.concatenate([getattr(c, f) for c in clouds]) for f in fields))


def test_single_isotropic_gaussian_renders_its_analytic_footprint():
    cam = look_at(np.array([4.0, 0.0, 0.0]), np.zeros(3), width=33, height=33, fov_x_deg=40.0)
    sigma, opacity = 0.2, 0.7
    colour, background = np.array([0.9, 0.5, 0.1]), np.array([0.0, 0.2, 0.4])
    settings = RasterSettings(alpha_min=0.0, low_pass=0.0)
    out = reference.render(_isotropic((0, 0, 0), sigma, opacity, colour), cam, background, settings)

    # On the optical axis the footprint is a circular Gaussian of std f * sigma / z pixels.
    sigma_px = cam.fx * sigma / 4.0
    u, v = cam.pixel_centres()
    radius_sq = (u - cam.cx) ** 2 + (v - cam.cy) ** 2
    alpha = opacity * np.exp(-0.5 * radius_sq / sigma_px**2)
    expected = alpha[..., None] * colour + (1.0 - alpha[..., None]) * background
    np.testing.assert_allclose(out.color, expected, atol=1e-12)
    np.testing.assert_allclose(out.alpha, alpha, atol=1e-12)
    np.testing.assert_allclose(out.depth, 4.0 * alpha, atol=1e-12)


def test_alpha_is_clamped_and_thresholded():
    cam = look_at(np.array([4.0, 0.0, 0.0]), np.zeros(3), width=33, height=33, fov_x_deg=40.0)
    cloud = _isotropic((0, 0, 0), 0.1, 0.999, (1.0, 1.0, 1.0))
    out = reference.render(cloud, cam, np.zeros(3))
    assert np.isclose(out.alpha.max(), 0.99)
    visible = out.alpha[out.alpha > 0.0]
    assert visible.min() >= 1.0 / 255.0
    assert (out.alpha == 0.0).any()


def test_opaque_front_gaussian_hides_what_is_behind_it():
    cam = look_at(np.array([4.0, 0.0, 0.0]), np.zeros(3), width=33, height=33, fov_x_deg=40.0)
    front = _isotropic((1.0, 0, 0), 0.5, 0.999, (1.0, 0.0, 0.0))
    back = _isotropic((-1.0, 0, 0), 0.5, 0.999, (0.0, 1.0, 0.0))
    out = reference.render(_concat(back, front), cam, np.zeros(3))
    centre = out.color[16, 16]
    # Both alphas saturate at 0.99: the back one is seen through 1 % transmittance.
    np.testing.assert_allclose(centre, [0.99, 0.01 * 0.99, 0.0], atol=1e-9)


def test_render_does_not_depend_on_the_storage_order(camera, background):
    cloud = verify.random_scene(40, seed=1)
    order = np.random.default_rng(2).permutation(len(cloud))
    first = reference.render(cloud, camera, background)
    second = reference.render(cloud.permuted(order), camera, background)
    np.testing.assert_allclose(second.color, first.color, atol=1e-12)
    np.testing.assert_allclose(second.depth, first.depth, atol=1e-12)


def test_weights_and_final_transmittance_sum_to_one(camera):
    # With every colour equal to the background colour, the image must be that colour.
    cloud = verify.random_scene(40, seed=3)
    grey = np.full(3, 0.6)
    cloud.sh_dc[:] = (grey - 0.5) / 0.28209479177387814
    out = reference.render(cloud, camera, grey)
    np.testing.assert_allclose(out.color, np.broadcast_to(grey, out.color.shape), atol=1e-12)


def test_accumulated_opacity_never_decreases_when_gaussians_are_added(camera, background):
    cloud = verify.random_scene(30, seed=4)
    previous = np.zeros((camera.height, camera.width))
    for count in (1, 5, 15, 30):
        subset = cloud.permuted(np.arange(count))
        alpha = reference.render(subset, camera, background).alpha
        assert (alpha >= previous - 1e-12).all()
        assert alpha.min() >= 0.0
        assert alpha.max() < 1.0
        previous = alpha


def test_gaussians_behind_the_camera_are_ignored(camera, background):
    cloud = verify.random_scene(10, seed=5)
    behind = _isotropic(camera.position - 2.0 * camera.rotation[2], 0.3, 0.9, (1.0, 1.0, 1.0))
    expected = reference.render(cloud, camera, background)
    actual = reference.render(_concat(cloud, behind), camera, background)
    np.testing.assert_array_equal(actual.color, expected.color)


def test_depth_of_a_nearly_opaque_surface_is_the_depth_of_its_gaussians(camera):
    # Many overlapping Gaussians in a plane facing the camera at distance 3.
    rng = np.random.default_rng(6)
    axis, right, down = camera.rotation[2], camera.rotation[0], camera.rotation[1]
    centre = camera.position + 3.0 * axis
    spread = rng.uniform(-0.6, 0.6, (200, 2))
    means = centre + spread[:, :1] * right + spread[:, 1:] * down
    cloud = GaussianCloud.from_activated(
        means=means,
        scales=np.full((200, 3), 0.15),
        quats=np.tile([1.0, 0.0, 0.0, 0.0], (200, 1)),
        opacities=np.full(200, 0.9),
        colours=np.full((200, 3), 0.5),
    )
    out = reference.render(cloud, camera, np.zeros(3), SMOOTH)
    middle = (camera.height // 2, camera.width // 2)
    assert out.alpha[middle] > 0.999
    assert np.isclose(out.depth[middle] / out.alpha[middle], 3.0, atol=1e-9)
