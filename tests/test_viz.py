"""Figure helpers: geometry of the overlay and structure of the frames."""

from __future__ import annotations

import numpy as np
from PIL import Image

from microsplat import viz
from microsplat.camera import look_at
from microsplat.gaussians import GaussianCloud

MAX_README_WIDTH = 900


def _one_red_gaussian() -> tuple[GaussianCloud, object]:
    cam = look_at(np.array([4.0, 0.0, 0.0]), np.zeros(3), width=64, height=64, fov_x_deg=40.0)
    cloud = GaussianCloud.from_activated(
        means=np.zeros((1, 3)),
        scales=np.array([[0.1, 0.3, 0.1]]),  # elongated along world y, the image x axis
        quats=np.array([[1.0, 0.0, 0.0, 0.0]]),
        opacities=np.array([0.8]),
        colours=np.array([[1.0, 0.0, 0.0]]),
    )
    return cloud, cam


def test_ellipse_outline_is_centred_and_follows_the_long_axis():
    cloud, cam = _one_red_gaussian()
    overlay = np.asarray(viz.ellipse_overlay(cloud, cam, size=128)).astype(int)
    drawn = (overlay[..., 0] > 200) & (overlay[..., 1] < 50)
    rows, cols = np.nonzero(drawn)
    assert drawn.sum() > 20
    assert abs(rows.mean() - 64.0) < 1.5
    assert abs(cols.mean() - 64.0) < 1.5
    width, height = np.ptp(cols), np.ptp(rows)
    # Projected standard deviations are f * s / z, so the outline is about 3 times wider.
    assert 2.5 < width / height < 3.5
    expected_width = 2 * viz.ELLIPSE_SIGMAS * (2.0 * cam.fx) * 0.3 / 4.0
    assert abs(width - expected_width) < 4.0


def test_transparent_gaussians_are_left_out_of_the_overlay():
    cloud, cam = _one_red_gaussian()
    cloud.opacity_logits[:] = -6.0
    overlay = np.asarray(viz.ellipse_overlay(cloud, cam, size=64))
    assert (overlay == np.array(viz.CANVAS_COLOUR)).all()


def test_error_map_is_dark_where_the_images_agree():
    target = np.full((8, 8, 3), 0.4)
    image = target.copy()
    image[2, 3] += 0.25
    errors = viz.error_map(image, target)
    assert errors.shape == (8, 8, 3)
    assert errors[0, 0].sum() < 0.05
    assert errors[2, 3].sum() > 2.0


def test_frame_fits_the_readme_and_gif_keeps_every_frame(tmp_path):
    cloud, cam = _one_red_gaussian()
    target = np.zeros((64, 64, 3))
    frames = [
        viz.four_panel_frame(target, target + 0.1 * k, cloud, cam, f"frame {k}") for k in range(3)
    ]
    assert frames[0].width <= MAX_README_WIDTH
    assert frames[0].size == (
        4 * viz.PANEL_SIZE + 5 * viz.PANEL_GAP,
        viz.HEADER_HEIGHT + viz.PANEL_SIZE + viz.FOOTER_HEIGHT,
    )
    path = tmp_path / "frames.gif"
    viz.save_gif(frames, [50, 50, 200], path)
    with Image.open(path) as gif:
        assert gif.n_frames == 3
        assert gif.size == frames[0].size
