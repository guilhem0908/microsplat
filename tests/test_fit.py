"""End-to-end optimisation: gradients must actually pull a scene towards its target images."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from microsplat import scenes, verify
from microsplat.camera import look_at
from microsplat.density import DensityConfig
from microsplat.fit import FitConfig, evaluate, fit
from microsplat.optim import Adam
from microsplat.torch_renderer import cloud_to_params, render

QUICK = FitConfig(
    iterations=120,
    initial_gaussians=200,
    density=DensityConfig(start=40, stop=100, interval=20, max_gaussians=400),
    eval_every=40,
)


def _views(name: str, resolution: int):
    scene = scenes.make_scene(name)
    train = [scene.observe(c) for c in scenes.training_cameras(resolution)]
    held_out = [scene.observe(c) for c in scenes.held_out_cameras(resolution)[:2]]
    return scene, train, held_out


def test_descent_on_positions_recovers_a_perturbed_scene():
    # Three views from different sides make all three coordinates observable.
    eyes = ([3.0, 1.0, 1.5], [-1.0, 3.0, 1.0], [0.5, -3.2, 0.8])
    cams = [
        look_at(np.array(eye), np.zeros(3), width=32, height=24, fov_x_deg=50.0) for eye in eyes
    ]
    truth = verify.random_scene(8, seed=0, max_scale=0.3)
    background = torch.zeros(3, dtype=torch.float64)
    with torch.no_grad():
        exact = cloud_to_params(truth, dtype=torch.float64)
        targets = [render(exact, cam, background).color for cam in cams]

    generator = torch.Generator().manual_seed(1)
    offset = 0.05 * torch.randn((8, 3), dtype=torch.float64, generator=generator)
    params = cloud_to_params(truth, dtype=torch.float64)
    params["means"] = (params["means"] + offset).requires_grad_(True)
    moving = {"means": params["means"]}

    adam = Adam(moving)
    for step in range(300):
        view = step % len(cams)
        loss = ((render(params, cams[view], background).color - targets[view]) ** 2).mean()
        loss.backward()
        adam.step(moving, {"means": 2e-3 * 0.99**step})

    initial_error = offset.norm(dim=1).mean().item()
    final_error = (params["means"].detach() - exact["means"]).norm(dim=1).mean().item()
    assert final_error < 0.1 * initial_error


def test_short_fit_improves_held_out_psnr_and_respects_the_cap():
    scene, train, held_out = _views("knot", 32)
    result = fit(scene, train, held_out, QUICK)
    history = result.history
    assert [h["iteration"] for h in history] == [0, 40, 80, 120]
    assert history[-1]["psnr"] > history[0]["psnr"] + 6.0
    assert history[-1]["ssim"] > history[0]["ssim"]
    assert all(h["gaussians"] <= 400 for h in history)
    assert len(result.cloud) == history[-1]["gaussians"]
    assert result.seconds > 0.0


def test_fit_is_deterministic_for_a_given_seed():
    scene, train, held_out = _views("knot", 32)
    config = FitConfig(iterations=30, initial_gaussians=100, density=None, eval_every=30)
    torch.set_num_threads(1)
    first = fit(scene, train, held_out, config)
    second = fit(scene, train, held_out, config)
    np.testing.assert_array_equal(first.cloud.means, second.cloud.means)
    assert first.history[-1]["psnr"] == second.history[-1]["psnr"]


def test_callback_sees_every_iteration_including_the_initial_state():
    scene, train, held_out = _views("knot", 32)
    config = FitConfig(iterations=5, initial_gaussians=50, density=None, eval_every=5)
    seen = []
    fit(scene, train, held_out, config, on_iteration=lambda i, p: seen.append((i, len(p["means"]))))
    assert seen == [(i, 50) for i in range(6)]


def test_evaluate_scores_the_hidden_scene_as_exact():
    scene, _, held_out = _views("knot", 32)
    params = cloud_to_params(scene.hidden_cloud, dtype=torch.float64)
    metrics = evaluate(params, held_out, torch.tensor(scene.background))
    assert metrics["psnr"] > 100.0
    assert np.isclose(metrics["ssim"], 1.0)
    assert metrics["depth_rel_err_median"] < 1e-10


@pytest.mark.slow
def test_longer_fit_reaches_30_db_on_the_knot():
    scene = scenes.make_scene("knot")
    train = [scene.observe(c) for c in scenes.training_cameras(64)]
    held_out = [scene.observe(c) for c in scenes.held_out_cameras(64)]
    config = FitConfig(
        iterations=800,
        initial_gaussians=500,
        density=DensityConfig(start=100, stop=600, interval=50, max_gaussians=1200),
        eval_every=400,
    )
    result = fit(scene, train, held_out, config)
    assert result.history[-1]["psnr"] > 30.0
    assert result.history[-1]["ssim"] > 0.95
