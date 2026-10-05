"""Ground-truth scenes: the ray tracer and the splat renderer share one camera model."""

from __future__ import annotations

import numpy as np
import pytest

from microsplat import reference, scenes
from microsplat.camera import look_at
from microsplat.gaussians import GaussianCloud

RESOLUTION = 48


def _top_camera():
    """Camera straight above the largest sphere, looking down."""
    centre = np.array(scenes.SPHERES[0][0])
    return look_at(
        centre + np.array([0.0, 0.0, 4.0]),
        centre,
        width=65,
        height=65,
        fov_x_deg=30.0,
        up=(0, 1, 0),
    )


def test_unknown_scene_is_rejected():
    with pytest.raises(ValueError, match="unknown scene"):
        scenes.make_scene("teapot")


def test_training_and_held_out_cameras_never_coincide():
    train = scenes.training_cameras(RESOLUTION)
    held_out = scenes.held_out_cameras(RESOLUTION)
    assert len(train) == 24
    assert len(held_out) == 8
    for cam in held_out:
        gaps = [np.linalg.norm(cam.position - other.position) for other in train]
        assert min(gaps) > 0.5


def test_ray_traced_depth_of_a_sphere_apex_is_distance_minus_radius():
    cam = _top_camera()
    view = scenes.make_scene("shapes").observe(cam)
    radius = scenes.SPHERES[0][1]
    assert np.isclose(view.depth[32, 32], 4.0 - radius, atol=1e-3)


def test_ray_traced_depth_is_undefined_on_the_background_only():
    scene = scenes.make_scene("shapes")
    view = scene.observe(scenes.training_cameras(RESOLUTION)[0])
    background = np.isnan(view.depth)
    assert 0.2 < background.mean() < 0.9
    # Pixels fully inside the background keep the background colour (edges are anti-aliased).
    np.testing.assert_allclose(view.image[0, 0], scene.background)
    assert np.nanmin(view.depth) > 2.0


def test_shading_is_bounded_and_shadows_darken_the_disc():
    view = scenes.make_scene("shapes").observe(scenes.training_cameras(96)[0])
    assert view.image.min() >= 0.0
    assert view.image.max() <= 1.0
    light = np.array(scenes.LIGHT_DIRECTION) / np.linalg.norm(scenes.LIGHT_DIRECTION)
    lit_checker = np.array(scenes.CHECKER_ALBEDOS[0]) * (scenes.AMBIENT + scenes.DIFFUSE * light[2])
    shadowed_checker = np.array(scenes.CHECKER_ALBEDOS[1]) * scenes.AMBIENT
    colours = view.image.reshape(-1, 3)
    assert np.abs(colours - shadowed_checker).sum(axis=1).min() < 0.02
    assert np.abs(colours - lit_checker).sum(axis=1).min() < 0.02


def test_ray_tracer_and_rasteriser_agree_on_where_a_sphere_is():
    # A small opaque Gaussian at the sphere centre must land on the centroid of the ray-traced
    # silhouette: the two image-formation codes use the same pixel convention.
    centre, radius, _ = scenes.SPHERES[0]
    cam = _top_camera()
    view = scenes.make_scene("shapes").observe(cam)
    silhouette = np.nan_to_num(view.depth, nan=np.inf) < 4.0 - 0.5 * radius
    u, v = cam.pixel_centres()
    traced = np.array([u[silhouette].mean(), v[silhouette].mean()])

    blob = GaussianCloud.from_activated(
        means=np.array([centre]),
        scales=np.full((1, 3), 0.1),
        quats=np.array([[1.0, 0.0, 0.0, 0.0]]),
        opacities=np.array([0.9]),
        colours=np.ones((1, 3)),
    )
    alpha = reference.render(blob, cam, np.zeros(3)).alpha
    splatted = np.array([(u * alpha).sum(), (v * alpha).sum()]) / alpha.sum()
    np.testing.assert_allclose(splatted, traced, atol=0.05)
    np.testing.assert_allclose(splatted, [32.5, 32.5], atol=1e-6)


def test_knot_gaussians_are_elongated_along_the_curve():
    cloud = scenes.make_scene("knot").hidden_cloud
    cov = reference.covariance_3d(cloud.log_scales, cloud.quats)
    _, vectors = np.linalg.eigh(cov)
    long_axis = vectors[:, :, -1]
    chord = np.roll(cloud.means, -1, axis=0) - np.roll(cloud.means, 1, axis=0)
    chord /= np.linalg.norm(chord, axis=1, keepdims=True)
    alignment = np.abs((long_axis * chord).sum(axis=1))
    assert alignment.min() > 0.99


def test_knot_view_is_the_reference_render_of_the_hidden_cloud():
    scene = scenes.make_scene("knot")
    cam = scenes.held_out_cameras(RESOLUTION)[0]
    view = scene.observe(cam)
    expected = reference.render(scene.hidden_cloud, cam, scene.background)
    np.testing.assert_array_equal(view.image, expected.color)
    solid = expected.alpha > 0.5
    np.testing.assert_allclose(view.depth[solid], expected.depth[solid] / expected.alpha[solid])
    assert np.isnan(view.depth[~solid]).all()


def test_scene_bounds_contain_the_geometry():
    shapes = scenes.make_scene("shapes")
    for centre, radius, _ in scenes.SPHERES:
        assert (np.array(centre) - radius >= shapes.box_min).all()
        assert (np.array(centre) + radius <= shapes.box_max).all()
    knot = scenes.make_scene("knot")
    assert (knot.hidden_cloud.means >= knot.box_min).all()
    assert (knot.hidden_cloud.means <= knot.box_max).all()
