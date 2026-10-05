"""Camera conventions: axes, pixel centres and ray/projection consistency."""

from __future__ import annotations

import numpy as np

from microsplat.camera import Camera, look_at, orbit_cameras


def _camera_looking_along_x() -> Camera:
    """Camera at the origin looking along world +x, with world z up."""
    return look_at(np.zeros(3), np.array([1.0, 0.0, 0.0]), width=64, height=48, fov_x_deg=60.0)


def test_rotation_is_a_proper_rotation():
    cam = look_at(np.array([2.0, -1.0, 0.7]), np.zeros(3), width=32, height=32, fov_x_deg=45.0)
    np.testing.assert_allclose(cam.rotation @ cam.rotation.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(cam.rotation), 1.0)


def test_target_projects_to_the_principal_point():
    eye, target = np.array([2.0, -1.0, 0.7]), np.array([0.1, 0.2, 0.3])
    cam = look_at(eye, target, width=64, height=48, fov_x_deg=45.0)
    uv, depth = cam.project(target[None])
    np.testing.assert_allclose(uv[0], [32.0, 24.0], atol=1e-10)
    assert np.isclose(depth[0], np.linalg.norm(target - eye))


def test_axes_follow_the_opencv_convention():
    cam = _camera_looking_along_x()
    centre, _ = cam.project(np.array([[5.0, 0.0, 0.0]]))
    # Looking along +x with z up, the right-hand side is -y.
    right, _ = cam.project(np.array([[5.0, -1.0, 0.0]]))
    above, _ = cam.project(np.array([[5.0, 0.0, 1.0]]))
    assert right[0, 0] > centre[0, 0]
    assert np.isclose(right[0, 1], centre[0, 1])
    assert above[0, 1] < centre[0, 1]
    assert np.isclose(above[0, 0], centre[0, 0])


def test_camera_position_inverts_the_extrinsics():
    eye = np.array([2.0, -1.0, 0.7])
    cam = look_at(eye, np.zeros(3), width=32, height=32, fov_x_deg=45.0)
    np.testing.assert_allclose(cam.position, eye, atol=1e-12)
    np.testing.assert_allclose(cam.world_to_camera(eye[None])[0], 0.0, atol=1e-12)


def test_pixel_rays_project_back_to_their_pixel_centres():
    cam = look_at(np.array([2.0, -1.0, 0.7]), np.zeros(3), width=16, height=12, fov_x_deg=45.0)
    points = cam.position + 3.0 * cam.pixel_rays()
    uv, _ = cam.project(points.reshape(-1, 3))
    u, v = cam.pixel_centres()
    np.testing.assert_allclose(uv[:, 0], u.ravel(), atol=1e-9)
    np.testing.assert_allclose(uv[:, 1], v.ravel(), atol=1e-9)


def test_orbit_cameras_are_equidistant_and_look_at_the_target():
    target = np.array([0.0, 0.0, 0.25])
    cams = orbit_cameras(
        6, radius=4.0, elevation_deg=30.0, target=target, width=32, height=32, fov_x_deg=40.0
    )
    for cam in cams:
        assert np.isclose(np.linalg.norm(cam.position - target), 4.0)
        np.testing.assert_allclose(cam.project(target[None])[0][0], [16.0, 16.0], atol=1e-9)
    heights = [cam.position[2] for cam in cams]
    np.testing.assert_allclose(heights, 0.25 + 4.0 * np.sin(np.deg2rad(30.0)))
