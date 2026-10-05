"""Clone, split and prune keep the scene and the optimiser state consistent."""

from __future__ import annotations

import math

import numpy as np
import torch

from microsplat import verify
from microsplat.density import (
    SPLIT_CHILDREN,
    SPLIT_SCALE_DIVISOR,
    DensityConfig,
    DensityController,
)
from microsplat.optim import Adam
from microsplat.torch_renderer import PARAM_NAMES, cloud_to_params

EXTENT = 2.0
COUNT = 12


def _setup(config: DensityConfig, scales: list[float], opacities: list[float]):
    """A scene of COUNT Gaussians with chosen isotropic scales and opacities."""
    cloud = verify.random_scene(COUNT, seed=0)
    cloud.log_scales[:] = np.log(np.array(scales))[:, None]
    cloud.opacity_logits[:] = np.log(np.array(opacities) / (1.0 - np.array(opacities)))
    params = cloud_to_params(cloud, requires_grad=True)
    adam = Adam(params)
    for name in PARAM_NAMES:  # recognisable moments: row index everywhere
        rows = torch.arange(COUNT, dtype=torch.float32)
        adam.first[name] += rows.view(-1, *[1] * (params[name].dim() - 1))
    controller = DensityController(config, EXTENT, COUNT, torch.device("cpu"))
    return params, adam, controller


def _observe(controller: DensityController, gradient_norms: list[float]) -> None:
    grads = torch.zeros(COUNT, 2)
    grads[:, 0] = torch.tensor(gradient_norms)
    controller.observe(grads, torch.ones(COUNT, dtype=torch.bool))


def test_schedule_fires_on_the_interval_between_start_and_stop():
    controller = DensityController(
        DensityConfig(start=200, stop=500, interval=100), EXTENT, COUNT, torch.device("cpu")
    )
    due = [i for i in range(1, 801) if controller.is_due(i)]
    assert due == [200, 300, 400, 500]


def test_gradient_statistic_is_the_mean_norm_over_views_that_saw_the_gaussian():
    controller = DensityController(DensityConfig(), EXTENT, 3, torch.device("cpu"))
    controller.observe(
        torch.tensor([[3.0, 4.0], [0.0, 1.0], [9.0, 9.0]]), torch.tensor([True, True, False])
    )
    controller.observe(
        torch.tensor([[0.0, 1.0], [5.0, 5.0], [0.0, 2.0]]), torch.tensor([True, False, True])
    )
    np.testing.assert_allclose(controller.grad_sum.numpy(), [6.0, 1.0, 2.0])
    np.testing.assert_allclose(controller.seen.numpy(), [2.0, 1.0, 1.0])


def test_small_hot_gaussians_are_cloned_and_large_ones_split():
    config = DensityConfig(grad_threshold=1.0, split_scale=0.05, max_gaussians=100)
    small, large = 0.01, 0.5  # threshold is split_scale * EXTENT = 0.1
    scales = [small] * 6 + [large] * 6
    params, adam, controller = _setup(config, scales, [0.5] * COUNT)
    hot = [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    _observe(controller, hot)
    old = {name: p.detach().clone() for name, p in params.items()}

    new, step = controller.update(params, adam, torch.Generator().manual_seed(0))

    assert (step.cloned, step.split, step.pruned) == (2, 1, 0)
    # 12 - 1 split parent + 2 clones + 2 children.
    assert len(new["means"]) == COUNT - 1 + 2 + SPLIT_CHILDREN
    kept = [i for i in range(COUNT) if i != 6]
    torch.testing.assert_close(new["means"][: COUNT - 1], old["means"][kept])
    # Clones are exact copies of Gaussians 0 and 1.
    for name in PARAM_NAMES:
        torch.testing.assert_close(new[name][COUNT - 1 : COUNT + 1], old[name][:2])
    # Children: scales divided by 1.6, everything else inherited, positions near the parent.
    children = slice(COUNT + 1, COUNT + 1 + SPLIT_CHILDREN)
    expected_log_scale = math.log(large / SPLIT_SCALE_DIVISOR)
    torch.testing.assert_close(
        new["log_scales"][children], torch.full((SPLIT_CHILDREN, 3), expected_log_scale)
    )
    torch.testing.assert_close(new["sh_dc"][children], old["sh_dc"][[6, 6]])
    distance = (new["means"][children] - old["means"][6]).norm(dim=1)
    assert (distance > 0).all()
    assert (distance < 5 * large).all()
    assert all(p.requires_grad and p.is_leaf for p in new.values())


def test_optimiser_moments_follow_the_surviving_gaussians():
    config = DensityConfig(grad_threshold=1.0, split_scale=0.05, max_gaussians=100)
    params, adam, controller = _setup(config, [0.01] * 6 + [0.5] * 6, [0.5] * COUNT)
    _observe(controller, [2.0] + [0.0] * 5 + [2.0] + [0.0] * 5)
    new, _ = controller.update(params, adam, torch.Generator().manual_seed(0))

    kept = [i for i in range(COUNT) if i != 6]
    for name in PARAM_NAMES:
        assert adam.first[name].shape == new[name].shape
        rows = adam.first[name].reshape(len(new[name]), -1)[:, 0]
        np.testing.assert_allclose(rows[: COUNT - 1].numpy(), kept)
        assert not rows[COUNT - 1 :].any()


def test_transparent_gaussians_are_pruned_and_never_densified():
    config = DensityConfig(grad_threshold=1.0, prune_opacity=0.005, max_gaussians=100)
    opacities = [0.001, 0.004] + [0.5] * (COUNT - 2)
    params, adam, controller = _setup(config, [0.01] * COUNT, opacities)
    _observe(controller, [5.0] * 2 + [0.0] * (COUNT - 2))
    new, step = controller.update(params, adam, torch.Generator().manual_seed(0))
    assert (step.cloned, step.split, step.pruned) == (0, 0, 2)
    assert len(new["means"]) == COUNT - 2
    assert (torch.sigmoid(new["opacity_logits"]) >= 0.005).all()


def test_cap_keeps_the_largest_gradients_first():
    config = DensityConfig(grad_threshold=1.0, max_gaussians=COUNT + 2)
    params, adam, controller = _setup(config, [0.01] * COUNT, [0.5] * COUNT)
    norms = [float(i + 2) for i in range(COUNT)]  # all above the threshold
    _observe(controller, norms)
    old_means = params["means"].detach().clone()
    new, step = controller.update(params, adam, torch.Generator().manual_seed(0))
    assert step.cloned == 2
    assert len(new["means"]) == COUNT + 2
    torch.testing.assert_close(new["means"][COUNT:], old_means[[COUNT - 2, COUNT - 1]])


def test_statistics_restart_after_an_update():
    config = DensityConfig(grad_threshold=1.0, max_gaussians=100)
    params, adam, controller = _setup(config, [0.01] * COUNT, [0.5] * COUNT)
    _observe(controller, [2.0] * COUNT)
    new, _ = controller.update(params, adam, torch.Generator().manual_seed(0))
    assert controller.grad_sum.shape == (len(new["means"]),)
    assert not controller.grad_sum.any()
    assert not controller.seen.any()
