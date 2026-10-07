"""Autograd through the PyTorch renderer against finite differences."""

from __future__ import annotations

import torch

from microsplat import torch_renderer, verify
from microsplat.metrics import photometric_loss
from microsplat.settings import RasterSettings
from microsplat.torch_renderer import PARAM_NAMES


def test_autograd_matches_central_differences_on_the_smooth_renderer():
    # alpha_min = 0 removes the visibility threshold, so the render is differentiable.
    errors = verify.gradient_errors(alpha_min=0.0, seed=0)
    assert set(errors) == set(PARAM_NAMES)
    assert max(errors.values()) < verify.GRADIENT_TOLERANCE


def test_autograd_matches_central_differences_with_the_default_threshold():
    # The threshold makes the render piecewise smooth. Autograd differentiates the pieces; a
    # finite difference agrees unless a pixel crosses alpha_min inside the step, which does
    # not happen for this scene.
    errors = verify.gradient_errors(alpha_min=RasterSettings().alpha_min, seed=0)
    assert max(errors.values()) < verify.GRADIENT_TOLERANCE


def test_torch_gradcheck_accepts_the_renderer_in_float64():
    cam = verify.check_camera(8, 6)
    cloud = verify.random_scene(4, seed=1, max_scale=0.5)
    params = torch_renderer.cloud_to_params(cloud, dtype=torch.float64, requires_grad=True)
    background = torch.tensor(verify.CHECK_BACKGROUND, dtype=torch.float64)
    settings = RasterSettings(alpha_min=0.0)

    def image(*values: torch.Tensor) -> torch.Tensor:
        named = dict(zip(PARAM_NAMES, values, strict=True))
        return torch_renderer.render(named, cam, background, settings, tile_size=4).color

    inputs = tuple(params[name] for name in PARAM_NAMES)
    assert torch.autograd.gradcheck(image, inputs, eps=1e-6, atol=1e-7, rtol=1e-5)


def test_training_loss_has_finite_gradients_for_every_parameter():
    cam = verify.check_camera(32, 24)
    cloud = verify.random_scene(50, seed=2)
    params = torch_renderer.cloud_to_params(cloud, requires_grad=True)
    target = torch.rand((24, 32, 3), generator=torch.Generator().manual_seed(0))
    out = torch_renderer.render(params, cam, torch.zeros(3))
    photometric_loss(out.color, target).backward()
    for name in PARAM_NAMES:
        grad = params[name].grad
        assert torch.isfinite(grad).all(), name
        assert grad.abs().max() > 0.0, name


def test_a_view_in_which_nothing_is_visible_still_backpropagates():
    # Every Gaussian behind the camera: no tile holds a Gaussian, the image is the background.
    cam = verify.check_camera(32, 24)
    cloud = verify.random_scene(20, seed=3)
    cloud.means[:] = cam.position - 2.0 * cam.rotation[2]
    params = torch_renderer.cloud_to_params(cloud, requires_grad=True)
    background = torch.full((3,), 0.25)
    out = torch_renderer.render(params, cam, background)
    out.means2d.retain_grad()
    assert not out.visible.any()
    assert torch.equal(out.color, background.expand_as(out.color))
    (out.color.sum() + out.depth.sum() + out.alpha.sum()).backward()
    assert float(out.means2d.grad.abs().max()) == 0.0
    for name in PARAM_NAMES:
        assert params[name].grad is not None, name
        assert float(params[name].grad.abs().max()) == 0.0, name


def test_gaussians_behind_the_camera_receive_zero_gradient():
    cam = verify.check_camera(32, 24)
    cloud = verify.random_scene(20, seed=3)
    cloud.means[:5] = cam.position - 2.0 * cam.rotation[2]
    params = torch_renderer.cloud_to_params(cloud, dtype=torch.float64, requires_grad=True)
    out = torch_renderer.render(params, cam, torch.zeros(3, dtype=torch.float64))
    out.color.sum().backward()
    for name in PARAM_NAMES:
        assert torch.isfinite(params[name].grad).all(), name
        assert float(params[name].grad[:5].abs().max()) == 0.0, name
    assert float(params["means"].grad[5:].abs().max()) > 0.0
