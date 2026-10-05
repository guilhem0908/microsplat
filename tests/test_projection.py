"""The projection of a 3D Gaussian: covariance construction, Jacobian and EWA footprint."""

from __future__ import annotations

import numpy as np

from microsplat import reference, verify
from microsplat.camera import look_at
from microsplat.gaussians import GaussianCloud

MONTE_CARLO_SAMPLES = 400_000


def test_quaternion_gives_an_orthonormal_right_handed_matrix():
    rot = reference.quat_to_rotmat(np.random.default_rng(0).normal(size=(50, 4)))
    identity = np.broadcast_to(np.eye(3), rot.shape)
    np.testing.assert_allclose(rot @ rot.transpose(0, 2, 1), identity, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0)


def test_quaternion_of_a_quarter_turn_about_z_maps_x_onto_y():
    half_angle = np.pi / 4.0
    quat = np.array([[np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)]])
    rot = reference.quat_to_rotmat(quat)[0]
    np.testing.assert_allclose(rot @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rot @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], atol=1e-12)


def test_covariance_eigenvalues_are_the_squared_scales():
    cloud = verify.random_scene(20, seed=1)
    cov = reference.covariance_3d(cloud.log_scales, cloud.quats)
    np.testing.assert_allclose(cov, cov.transpose(0, 2, 1), atol=1e-14)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(cov), np.sort(cloud.scales**2, axis=1), rtol=1e-10
    )


def test_covariance_ignores_the_quaternion_norm():
    cloud = verify.random_scene(10, seed=2)
    expected = reference.covariance_3d(cloud.log_scales, cloud.quats)
    scaled = reference.covariance_3d(cloud.log_scales, 3.7 * cloud.quats)
    np.testing.assert_allclose(scaled, expected, rtol=1e-12)


def test_perspective_jacobian_matches_central_differences():
    assert verify.jacobian_error(seed=3) < 1e-8


def test_projected_centre_and_depth_match_the_camera_model(camera):
    cloud = verify.random_scene(15, seed=4)
    proj = reference.project_gaussians(cloud, camera, low_pass=0.0)
    uv, depth = camera.project(cloud.means)
    np.testing.assert_allclose(proj.means2d, uv)
    np.testing.assert_allclose(proj.depth, depth)


def test_low_pass_adds_its_variance_to_the_diagonal_only(camera):
    cloud = verify.random_scene(15, seed=5)
    sharp = reference.project_gaussians(cloud, camera, low_pass=0.0).cov2d
    filtered = reference.project_gaussians(cloud, camera, low_pass=0.3).cov2d
    expected = np.broadcast_to(0.3 * np.eye(2), sharp.shape)
    np.testing.assert_allclose(filtered - sharp, expected, atol=1e-12)


def test_projected_covariance_matches_monte_carlo_for_a_small_gaussian():
    # Scale 0.02 at distance 3.5: the sampling noise, about sqrt(2 / samples) = 0.2 %,
    # dominates the linearisation error.
    assert verify.monte_carlo_covariance_error(0.02, MONTE_CARLO_SAMPLES, seed=6) < 0.01


def test_linearisation_error_grows_with_the_size_of_the_gaussian():
    small = verify.monte_carlo_covariance_error(0.02, MONTE_CARLO_SAMPLES, seed=7)
    large = verify.monte_carlo_covariance_error(0.6, MONTE_CARLO_SAMPLES, seed=7)
    # EWA is a first-order expansion: it is exact only in the limit of a small Gaussian.
    assert large > 0.03
    assert large > 5.0 * small


def test_on_axis_footprint_shrinks_with_the_square_of_the_distance():
    cloud = GaussianCloud.from_activated(
        means=np.zeros((1, 3)),
        scales=np.array([[0.05, 0.02, 0.035]]),
        quats=np.array([[0.9, 0.2, -0.3, 0.1]]),
        opacities=np.array([0.5]),
        colours=np.full((1, 3), 0.5),
    )
    kwargs = {"width": 64, "height": 64, "fov_x_deg": 40.0}
    near = look_at(np.array([3.0, 0.0, 0.0]), np.zeros(3), **kwargs)
    far = look_at(np.array([6.0, 0.0, 0.0]), np.zeros(3), **kwargs)
    cov_near = reference.project_gaussians(cloud, near, low_pass=0.0).cov2d[0]
    cov_far = reference.project_gaussians(cloud, far, low_pass=0.0).cov2d[0]
    np.testing.assert_allclose(cov_far, cov_near / 4.0, rtol=1e-10)
