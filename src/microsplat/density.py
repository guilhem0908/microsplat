"""Adaptive density control: clone, split and prune (Kerbl et al. 2023, section 5.2).

A Gaussian whose screen-space position receives a large average gradient sits in a region
that is not reconstructed well. If it is small it is *cloned* (under-reconstruction: more
primitives are needed), if it is large it is *split* into two smaller ones drawn from its own
density (over-reconstruction: one primitive covers too much). Nearly transparent Gaussians
are removed.

Not implemented: the periodic opacity reset of the paper, which matters for runs of tens of
thousands of iterations rather than for the few thousand used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from microsplat.optim import Adam
from microsplat.torch_renderer import quat_to_rotmat

#: Scale divisor applied to the two children of a split (phi = 1.6 in the paper).
SPLIT_SCALE_DIVISOR = 1.6
SPLIT_CHILDREN = 2


@dataclass(frozen=True)
class DensityConfig:
    """Schedule and thresholds of the density control.

    Attributes:
        start, stop, interval: Densify every ``interval`` iterations in ``[start, stop]``.
        grad_threshold: Mean norm of ``d loss / d (centre in pixels)`` above which a Gaussian
            is densified. The paper's 0.0002 is expressed for normalised coordinates and
            800-pixel images, so it does not transfer to the small images used here.
        split_scale: Largest standard deviation, as a fraction of the scene extent, that is
            still cloned; bigger Gaussians are split (0.01 in the paper).
        prune_opacity: Gaussians below this opacity are removed (0.005 in the paper).
        max_gaussians: Hard cap on the size of the scene; the largest gradients go first.
    """

    start: int = 200
    stop: int = 2000
    interval: int = 100
    grad_threshold: float = 4e-5
    split_scale: float = 0.01
    prune_opacity: float = 0.005
    max_gaussians: int = 3000


@dataclass(frozen=True)
class DensityStep:
    cloned: int
    split: int
    pruned: int


class DensityController:
    """Accumulates view-space gradients and periodically rewrites the Gaussian set."""

    def __init__(self, config: DensityConfig, extent: float, count: int, device: torch.device):
        self.config = config
        self.extent = extent
        self._reset(count, device)

    def _reset(self, count: int, device: torch.device) -> None:
        self.grad_sum = torch.zeros(count, device=device)
        self.seen = torch.zeros(count, device=device)

    def is_due(self, iteration: int) -> bool:
        cfg = self.config
        return cfg.start <= iteration <= cfg.stop and (iteration - cfg.start) % cfg.interval == 0

    @torch.no_grad()
    def observe(self, means2d_grad: torch.Tensor, visible: torch.Tensor) -> None:
        """Record the gradient norm of the projected centres for one rendered view."""
        self.grad_sum[visible] += means2d_grad[visible].norm(dim=1).to(self.grad_sum.dtype)
        self.seen[visible] += 1

    @torch.no_grad()
    def update(
        self, params: dict[str, torch.Tensor], adam: Adam, generator: torch.Generator
    ) -> tuple[dict[str, torch.Tensor], DensityStep]:
        """Return the new parameter set; ``adam`` is resized to match it."""
        cfg = self.config
        scales = torch.exp(params["log_scales"])
        prune = torch.sigmoid(params["opacity_logits"]) < cfg.prune_opacity
        mean_grad = self.grad_sum / self.seen.clamp(min=1)
        hot = (mean_grad > cfg.grad_threshold) & ~prune

        # Every densified Gaussian adds exactly one primitive; honour the cap.
        room = max(cfg.max_gaussians - int((~prune).sum()), 0)
        if int(hot.sum()) > room:
            ranked = torch.argsort(torch.where(hot, mean_grad, -torch.ones_like(mean_grad)))
            hot = torch.zeros_like(hot)
            hot[ranked[len(ranked) - room :]] = True

        split = hot & (scales.max(dim=1).values > cfg.split_scale * self.extent)
        clone = hot & ~split
        keep = ~(prune | split)

        # Children of a split: positions drawn from the parent density, mu + R (s * eps).
        parents = split.nonzero(as_tuple=True)[0].repeat(SPLIT_CHILDREN)
        noise = torch.randn(
            (len(parents), 3), generator=generator, device=scales.device, dtype=scales.dtype
        )
        rotations = quat_to_rotmat(params["quats"][parents])
        children = {name: p[parents].clone() for name, p in params.items()}
        children["means"] += (rotations @ (scales[parents] * noise)[..., None]).squeeze(-1)
        children["log_scales"] -= math.log(SPLIT_SCALE_DIVISOR)

        added = int(clone.sum()) + len(parents)
        new_params = {
            name: torch.cat([p[keep], p[clone], children[name]]).requires_grad_(True)
            for name, p in params.items()
        }
        adam.resize(keep, added)
        self._reset(len(new_params["means"]), scales.device)
        return new_params, DensityStep(int(clone.sum()), int(split.sum()), int(prune.sum()))
