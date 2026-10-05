"""Fit a Gaussian scene to posed images by gradient descent, and score it on held-out views.

The loop is the one of Kerbl et al. 2023 (section 5 and algorithm 1) at a small scale: one
random training view per iteration, the L1 + D-SSIM loss, Adam with per-group learning rates,
an exponentially decaying position learning rate, and optional adaptive density control.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from microsplat.density import DensityConfig, DensityController
from microsplat.gaussians import GaussianCloud, random_cloud
from microsplat.metrics import photometric_loss, psnr, ssim
from microsplat.optim import Adam
from microsplat.scenes import Scene, View
from microsplat.settings import RasterSettings
from microsplat.torch_renderer import cloud_to_params, params_to_cloud, render

#: Accumulated opacity above which a pixel counts as covered when depth is compared.
DEPTH_ALPHA_THRESHOLD = 0.5


@dataclass(frozen=True)
class FitConfig:
    """Hyper-parameters of one optimisation run.

    The learning rates are those of the reference implementation multiplied by a constant
    factor (see ``docs/design.md``): its schedule spans 30,000 iterations, this one 3,000.
    The position rate is multiplied by the scene extent and decays log-linearly to
    ``lr_means_final_ratio`` times its initial value.
    """

    iterations: int = 3000
    initial_gaussians: int = 1000
    lr_means: float = 8e-4
    lr_means_final_ratio: float = 0.02
    lr_log_scales: float = 0.01
    lr_quats: float = 0.002
    lr_opacity_logits: float = 0.05
    lr_sh_dc: float = 0.01
    density: DensityConfig | None = field(default_factory=DensityConfig)
    eval_every: int = 100
    seed: int = 0
    device: str = "cpu"


@dataclass
class FitResult:
    cloud: GaussianCloud
    history: list[dict[str, float]]  # one record per evaluation of the held-out views
    seconds: float  # wall-clock time of the optimisation, evaluations excluded


@torch.no_grad()
def evaluate(
    params: dict[str, torch.Tensor], views: list[View], background: torch.Tensor
) -> dict[str, float]:
    """Mean PSNR and SSIM over ``views`` and the median relative depth error.

    The rendered depth is the alpha-weighted depth divided by the accumulated opacity; it is
    compared with the true surface depth on pixels where the scene has a surface and the
    render is mostly opaque.
    """
    device, dtype = params["means"].device, params["means"].dtype
    psnrs, ssims, depth_errors = [], [], []
    for view in views:
        target = torch.as_tensor(view.image, dtype=dtype, device=device)
        out = render(params, view.camera, background)
        image = out.color.clamp(0.0, 1.0)
        psnrs.append(psnr(image, target).item())
        ssims.append(ssim(image, target).item())
        true_depth = torch.as_tensor(view.depth, dtype=dtype, device=device)
        covered = torch.isfinite(true_depth) & (out.alpha > DEPTH_ALPHA_THRESHOLD)
        estimate = out.depth[covered] / out.alpha[covered]
        depth_errors.append(((estimate - true_depth[covered]).abs() / true_depth[covered]).cpu())
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "depth_rel_err_median": float(torch.cat(depth_errors).median()),
    }


def fit(
    scene: Scene,
    train_views: list[View],
    test_views: list[View],
    config: FitConfig,
    on_iteration: Callable[[int, dict[str, torch.Tensor]], None] | None = None,
) -> FitResult:
    """Optimise a random initial cloud against ``train_views``.

    ``on_iteration(i, params)`` is called before the first update (``i = 0``) and after every
    update; the demo uses it to record frames.
    """
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator(device=config.device).manual_seed(config.seed)
    device = torch.device(config.device)
    settings = RasterSettings()

    initial = random_cloud(config.initial_gaussians, scene.box_min, scene.box_max, rng)
    params = cloud_to_params(initial, device=device, requires_grad=True)
    background = torch.as_tensor(scene.background, dtype=torch.float32, device=device)
    targets = [torch.as_tensor(v.image, dtype=torch.float32, device=device) for v in train_views]

    adam = Adam(params)
    controller = None
    if config.density is not None:
        controller = DensityController(config.density, scene.extent, len(initial), device)
    learning_rates = {
        "log_scales": config.lr_log_scales,
        "quats": config.lr_quats,
        "opacity_logits": config.lr_opacity_logits,
        "sh_dc": config.lr_sh_dc,
    }

    history: list[dict[str, float]] = []
    optimisation_seconds = 0.0

    def record(iteration: int) -> None:
        metrics = evaluate(params, test_views, background)
        history.append(
            {
                "iteration": iteration,
                "gaussians": len(params["means"]),
                "seconds": optimisation_seconds,
                **metrics,
            }
        )

    record(0)
    if on_iteration is not None:
        on_iteration(0, params)

    order = np.arange(len(train_views))
    for iteration in range(1, config.iterations + 1):
        tick = time.perf_counter()
        slot = (iteration - 1) % len(train_views)
        if slot == 0:
            order = rng.permutation(len(train_views))
        index = int(order[slot])

        out = render(params, train_views[index].camera, background, settings)
        out.means2d.retain_grad()
        loss = photometric_loss(out.color, targets[index])
        loss.backward()

        if controller is not None:
            controller.observe(out.means2d.grad, out.visible)
        progress = (iteration - 1) / max(config.iterations - 1, 1)
        learning_rates["means"] = (
            config.lr_means * scene.extent * config.lr_means_final_ratio**progress
        )
        adam.step(params, learning_rates)
        if controller is not None and controller.is_due(iteration):
            params, _ = controller.update(params, adam, generator)
        optimisation_seconds += time.perf_counter() - tick

        if iteration % config.eval_every == 0 or iteration == config.iterations:
            record(iteration)
        if on_iteration is not None:
            on_iteration(iteration, params)

    return FitResult(params_to_cloud(params), history, optimisation_seconds)
