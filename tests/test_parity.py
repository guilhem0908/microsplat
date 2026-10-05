"""The PyTorch twin must reproduce the NumPy reference, whatever its tiling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from microsplat import reference, torch_renderer, verify
from microsplat.settings import RasterSettings

FLOAT64_TOLERANCE = verify.PARITY_FLOAT64_TOLERANCE
FLOAT32_TOLERANCE = verify.PARITY_FLOAT32_TOLERANCE


def _worst(errors: dict) -> float:
    return max(errors[buffer] for buffer in verify.PARITY_BUFFERS)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_float64_twin_matches_the_reference_to_rounding(seed):
    errors = verify.parity_errors(300, seed, torch.float64)
    assert errors["same_order"]
    assert _worst(errors) < FLOAT64_TOLERANCE


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_float32_twin_matches_the_reference_to_1e_5(seed):
    errors = verify.parity_errors(300, seed, torch.float32)
    assert errors["same_order"]
    assert errors["color"] < FLOAT32_TOLERANCE
    assert errors["alpha"] < FLOAT32_TOLERANCE
    # Depth values are about 3.5, so their absolute rounding error is larger.
    assert errors["depth"] < 4.0 * FLOAT32_TOLERANCE


def test_float32_depth_tie_changes_colour_but_not_opacity():
    # In this scene two of the 1000 centres are closer in depth than float32 resolves, so the
    # float32 twin composites them in the other order. The colour of the pixels they share
    # moves; the accumulated opacity cannot, since prod (1 - alpha) ignores the order.
    errors = verify.parity_errors(1000, 0, torch.float32, width=128, height=128)
    assert not errors["same_order"]
    assert errors["color"] > FLOAT32_TOLERANCE
    assert errors["alpha"] < FLOAT32_TOLERANCE
    # The same scene in float64 keeps the reference order and matches to rounding.
    exact = verify.parity_errors(1000, 0, torch.float64, width=128, height=128)
    assert exact["same_order"]
    assert _worst(exact) < FLOAT64_TOLERANCE


@pytest.mark.parametrize("tile_size", [4, 8, 16, 32])
def test_tile_size_does_not_change_the_image(tile_size):
    errors = verify.parity_errors(200, 3, torch.float64, tile_size=tile_size)
    assert _worst(errors) < FLOAT64_TOLERANCE


@pytest.mark.parametrize("chunk_elements", [1, 1 << 10, 1 << 16])
def test_chunk_budget_does_not_change_the_image(chunk_elements):
    errors = verify.parity_errors(200, 4, torch.float64, chunk_elements=chunk_elements)
    assert _worst(errors) < FLOAT64_TOLERANCE


def test_image_size_need_not_be_a_multiple_of_the_tile_size():
    errors = verify.parity_errors(100, 5, torch.float64, width=37, height=23, tile_size=16)
    assert _worst(errors) < FLOAT64_TOLERANCE


def test_parity_holds_without_the_alpha_threshold():
    settings = RasterSettings(alpha_min=0.0)
    errors = verify.parity_errors(100, 6, torch.float64, settings=settings)
    assert _worst(errors) < FLOAT64_TOLERANCE


def test_culling_is_conservative_for_faint_and_hidden_gaussians(background):
    # Opacities below alpha_min and centres behind the camera reach no tile at all,
    # yet the image still equals the reference, which never culls by footprint.
    cam = verify.check_camera(48, 32)
    cloud = verify.random_scene(60, seed=7)
    cloud.opacity_logits[:20] = -8.0  # opacity 3e-4 < 1/255
    cloud.means[20:30] = cam.position - 2.0 * cam.rotation[2]
    expected = reference.render(cloud, cam, background)
    params = torch_renderer.cloud_to_params(cloud, dtype=torch.float64)
    actual = torch_renderer.render(params, cam, torch.tensor(background))
    assert not actual.visible[:30].any()
    np.testing.assert_allclose(actual.color.numpy(), expected.color, atol=FLOAT64_TOLERANCE)


def test_empty_view_is_the_background(background):
    cam = verify.check_camera(48, 32)
    cloud = verify.random_scene(5, seed=8)
    cloud.means[:] = cam.position - 2.0 * cam.rotation[2]
    params = torch_renderer.cloud_to_params(cloud, dtype=torch.float64)
    out = torch_renderer.render(params, cam, torch.tensor(background))
    np.testing.assert_allclose(
        out.color.numpy(), np.broadcast_to(background, out.color.shape), atol=1e-15
    )
    assert float(out.alpha.abs().max()) == 0.0


def test_twin_reports_the_same_projection_as_the_reference(camera):
    cloud = verify.random_scene(50, seed=9)
    proj = reference.project_gaussians(cloud, camera, low_pass=0.3)
    params = torch_renderer.cloud_to_params(cloud, dtype=torch.float64)
    means2d, cov2d, depth, in_front = torch_renderer.project_gaussians(
        params, camera, RasterSettings()
    )
    np.testing.assert_allclose(means2d.numpy(), proj.means2d, atol=1e-10)
    np.testing.assert_allclose(depth.numpy(), proj.depth, atol=1e-12)
    np.testing.assert_allclose(cov2d[:, 0].numpy(), proj.cov2d[:, 0, 0], rtol=1e-10)
    np.testing.assert_allclose(cov2d[:, 1].numpy(), proj.cov2d[:, 0, 1], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cov2d[:, 2].numpy(), proj.cov2d[:, 1, 1], rtol=1e-10)
    assert bool(in_front.all())


@pytest.mark.gpu
def test_cuda_twin_matches_the_reference():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    errors = verify.parity_errors(300, 0, torch.float64, device="cuda")
    assert _worst(errors) < FLOAT64_TOLERANCE
