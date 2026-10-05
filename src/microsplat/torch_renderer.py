"""Differentiable PyTorch twin of :mod:`microsplat.reference`.

Same equations, two differences in how they are evaluated:

* **Tiles.** The image is cut into square tiles and each tile only composites the Gaussians
  whose footprint can reach it. The footprint used for this test is exact with respect to the
  ``alpha_min`` threshold (see :func:`_extent`), so culling changes nothing in the output.
* **Bounded memory.** Tiles are processed in chunks of at most ``chunk_elements`` values of
  the (tile, Gaussian, pixel) tensor, grouped by how many Gaussians they hold so that padding
  stays small.

Everything is a plain tensor expression, so ``torch.autograd`` provides the gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from microsplat.camera import Camera
from microsplat.gaussians import GaussianCloud
from microsplat.settings import SH_C0, RasterSettings

PARAM_NAMES = ("means", "log_scales", "quats", "opacity_logits", "sh_dc")

#: Relative inflation of the culling box. The box is exact in real arithmetic; the margin only
#: absorbs rounding so that the tiled result equals the dense one in floating point too.
EXTENT_MARGIN = 1.01

DEFAULT_TILE_SIZE = 16
DEFAULT_CHUNK_ELEMENTS = 1 << 19


@dataclass
class TorchRender:
    """Output of :func:`render`; ``color``, ``depth`` and ``alpha`` match the reference."""

    color: torch.Tensor  # (H, W, 3)
    depth: torch.Tensor  # (H, W)
    alpha: torch.Tensor  # (H, W)
    means2d: torch.Tensor  # (N, 2) projected centres, part of the autograd graph
    cov2d: torch.Tensor  # (N, 3) projected covariance entries (xx, xy, yy), with low-pass
    visible: torch.Tensor  # (N,) bool, Gaussians that reached at least one tile


def cloud_to_params(
    cloud: GaussianCloud,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Copy a :class:`GaussianCloud` into a dictionary of tensors."""
    return {
        name: torch.tensor(
            getattr(cloud, name), device=device, dtype=dtype, requires_grad=requires_grad
        )
        for name in PARAM_NAMES
    }


def params_to_cloud(params: dict[str, torch.Tensor]) -> GaussianCloud:
    return GaussianCloud(**{name: params[name].detach().cpu().numpy() for name in PARAM_NAMES})


def quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (N, 3, 3) of (N, 4) quaternions ``(w, x, y, z)``, normalised here."""
    w, x, y, z = (quats / quats.norm(dim=1, keepdim=True)).unbind(dim=1)
    rows = [
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=1),
    ]
    return torch.stack(rows, dim=1)


def covariance_3d(log_scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """``Sigma = R S S^T R^T`` (Kerbl et al. 2023, eq. 6), shape (N, 3, 3)."""
    m = quat_to_rotmat(quats) * torch.exp(log_scales)[:, None, :]
    return m @ m.transpose(1, 2)


def project_gaussians(
    params: dict[str, torch.Tensor], camera: Camera, settings: RasterSettings
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """EWA projection ``Sigma' = J W Sigma W^T J^T + low_pass I`` (Kerbl et al. 2023, eq. 5).

    Returns the centres (N, 2), the covariance entries (N, 3) as (xx, xy, yy), the depths (N,)
    and the mask (N,) of Gaussians in front of the near plane. Centres behind the near plane
    are moved to depth 1 before the division so that they produce finite values and zero
    gradients; the mask removes them from the image.
    """
    means = params["means"]
    rot = torch.as_tensor(camera.rotation, dtype=means.dtype, device=means.device)
    trans = torch.as_tensor(camera.translation, dtype=means.dtype, device=means.device)
    cam = means @ rot.T + trans
    in_front = cam[:, 2] > settings.near
    tx, ty = cam[:, 0], cam[:, 1]
    tz = torch.where(in_front, cam[:, 2], torch.ones_like(cam[:, 2]))

    means2d = torch.stack([camera.fx * tx / tz + camera.cx, camera.fy * ty / tz + camera.cy], 1)

    zero = torch.zeros_like(tz)
    jac = torch.stack(
        [
            torch.stack([camera.fx / tz, zero, -camera.fx * tx / tz**2], dim=1),
            torch.stack([zero, camera.fy / tz, -camera.fy * ty / tz**2], dim=1),
        ],
        dim=1,
    )
    jw = jac @ rot
    cov = jw @ covariance_3d(params["log_scales"], params["quats"]) @ jw.transpose(1, 2)
    cov2d = torch.stack(
        [cov[:, 0, 0] + settings.low_pass, cov[:, 0, 1], cov[:, 1, 1] + settings.low_pass], dim=1
    )
    return means2d, cov2d, cam[:, 2], in_front


def _extent(
    cov2d: torch.Tensor, opacity: torch.Tensor, in_front: torch.Tensor, alpha_min: float
) -> torch.Tensor:
    """Half-width and half-height (N, 2) of the region where a Gaussian can be visible.

    ``o exp(-0.5 d^T Sigma'^-1 d) >= alpha_min`` is the ellipse ``d^T Sigma'^-1 d <= k`` with
    ``k = 2 ln(o / alpha_min)``, whose axis-aligned bounding box has half-sizes
    ``sqrt(k Sigma'_xx)`` and ``sqrt(k Sigma'_yy)``. Gaussians that never reach ``alpha_min``,
    or that sit behind the near plane, get a negative extent and overlap nothing. With
    ``alpha_min = 0`` the footprint is unbounded.
    """
    if alpha_min <= 0.0:
        half = torch.full_like(cov2d[:, :2], torch.inf)
    else:
        k = 2.0 * torch.log(opacity / alpha_min)
        half = EXTENT_MARGIN * torch.sqrt(k.clamp(min=0.0)[:, None] * cov2d[:, [0, 2]])
        half = torch.where((k > 0.0)[:, None], half, -torch.ones_like(half))
    return torch.where(in_front[:, None], half, -torch.ones_like(half))


def _tile_overlaps(
    means2d: torch.Tensor, half: torch.Tensor, tiles_x: int, tiles_y: int, tile_size: int
) -> torch.Tensor:
    """Boolean matrix (tiles, N): can Gaussian n touch a pixel centre of tile t?"""
    device = means2d.device
    origin_x = torch.arange(tiles_x, device=device).repeat(tiles_y) * tile_size
    origin_y = torch.arange(tiles_y, device=device).repeat_interleave(tiles_x) * tile_size
    first_x = (origin_x + 0.5)[:, None]
    last_x = (origin_x + tile_size - 0.5)[:, None]
    first_y = (origin_y + 0.5)[:, None]
    last_y = (origin_y + tile_size - 0.5)[:, None]
    mx, my = means2d[None, :, 0], means2d[None, :, 1]
    hx, hy = half[None, :, 0], half[None, :, 1]
    return (
        (hx >= 0)
        & (mx + hx >= first_x)
        & (mx - hx <= last_x)
        & (my + hy >= first_y)
        & (my - hy <= last_y)
    )


def _chunks_of_similar_tiles(counts: torch.Tensor, pixels: int, budget: int) -> list[torch.Tensor]:
    """Group tile indices so that ``tiles * max_count * pixels <= budget`` in every group.

    Tiles are visited by increasing Gaussian count: tiles of a group then need similar padding.
    """
    order = torch.argsort(counts).tolist()
    count_list = counts.tolist()
    chunks, current = [], []
    for tile in order:
        if current and (len(current) + 1) * count_list[tile] * pixels > budget:
            chunks.append(current)
            current = []
        current.append(tile)
    if current:
        chunks.append(current)
    return [torch.tensor(chunk, device=counts.device) for chunk in chunks]


def render(
    params: dict[str, torch.Tensor],
    camera: Camera,
    background: torch.Tensor,
    settings: RasterSettings | None = None,
    *,
    tile_size: int = DEFAULT_TILE_SIZE,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
) -> TorchRender:
    """Render one view; see :func:`microsplat.reference.render` for the equations."""
    settings = settings or RasterSettings()
    means2d, cov2d, depth, in_front = project_gaussians(params, camera, settings)
    opacity = torch.sigmoid(params["opacity_logits"])
    colour = torch.clamp(0.5 + SH_C0 * params["sh_dc"], min=0.0)
    dtype, device = means2d.dtype, means2d.device

    # Inverse of [[a, b], [b, c]] is [[c, -b], [-b, a]] / det; the low-pass keeps det > 0.
    a, b, c = cov2d.unbind(dim=1)
    det = a * c - b * b
    conic = torch.stack([c / det, -b / det, a / det], dim=1)

    tiles_x = -(-camera.width // tile_size)
    tiles_y = -(-camera.height // tile_size)
    pixels = tile_size * tile_size

    with torch.no_grad():
        order = torch.argsort(depth, stable=True)
        half = _extent(cov2d, opacity, in_front, settings.alpha_min)
        overlaps = _tile_overlaps(means2d[order], half[order], tiles_x, tiles_y, tile_size)
        counts = overlaps.sum(dim=1)
        visible = torch.zeros_like(in_front)
        visible[order] = overlaps.any(dim=0)
        chunks = _chunks_of_similar_tiles(counts, pixels, chunk_elements)

    offset = torch.arange(tile_size, dtype=dtype, device=device) + 0.5
    background = background.to(dtype=dtype, device=device)

    colors, depths, alphas = [], [], []
    for tiles in chunks:
        k_max = int(counts[tiles].max())
        if k_max == 0:
            colors.append(background.expand(len(tiles), pixels, 3))
            depths.append(torch.zeros(len(tiles), pixels, dtype=dtype, device=device))
            alphas.append(torch.zeros(len(tiles), pixels, dtype=dtype, device=device))
            continue

        # Padded (tiles, K) table of depth-sorted Gaussian indices.
        row, col = overlaps[tiles].nonzero(as_tuple=True)
        starts = torch.cumsum(counts[tiles], dim=0) - counts[tiles]
        slot = torch.arange(len(row), device=device) - starts[row]
        index = torch.zeros(len(tiles), k_max, dtype=torch.long, device=device)
        valid = torch.zeros(len(tiles), k_max, dtype=torch.bool, device=device)
        index[row, slot] = order[col]
        valid[row, slot] = True

        # Pixel-centre coordinates along the two sides of each tile, relative to each centre.
        tile_x = (tiles % tiles_x * tile_size).to(dtype)[:, None, None]
        tile_y = (tiles // tiles_x * tile_size).to(dtype)[:, None, None]
        dx = tile_x + offset - means2d[index, 0, None]  # (tiles, K, S)
        dy = tile_y + offset - means2d[index, 1, None]
        qa, qb, qc = conic[index].unbind(dim=2)
        # -0.5 d^T Q d = -0.5 qa dx^2 - qb dx dy - 0.5 qc dy^2. Only the cross term needs the
        # full (S, S) pixel grid; the two squares are computed per row and per column.
        exponent = (
            (-0.5 * qa[..., None] * dx * dx)[:, :, None, :]
            + (-0.5 * qc[..., None] * dy * dy)[:, :, :, None]
            - (qb[..., None] * dy)[:, :, :, None] * dx[:, :, None, :]
        ).flatten(start_dim=2)  # (tiles, K, P), row-major pixels
        # Padding slots get zero opacity, hence zero alpha.
        slot_opacity = (opacity[index] * valid)[..., None]
        alpha = torch.clamp(slot_opacity * torch.exp(exponent), max=settings.alpha_max)
        alpha = torch.where(alpha >= settings.alpha_min, alpha, torch.zeros_like(alpha))

        # T_i = prod_{j<i} (1 - alpha_j): exclusive cumulative product along the depth order.
        through = torch.cumprod(1.0 - alpha, dim=1)
        transmittance = torch.cat([torch.ones_like(through[:, :1]), through[:, :-1]], dim=1)
        weight = alpha * transmittance
        final = through[:, -1]
        colors.append(
            torch.einsum("tkp,tkc->tpc", weight, colour[index]) + final[..., None] * background
        )
        depths.append(torch.einsum("tkp,tk->tp", weight, depth[index]))
        alphas.append(1.0 - final)

    # Undo the chunk ordering, then lay the tiles out as an image and crop the padding.
    inverse = torch.argsort(torch.cat(chunks))

    def assemble(parts: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.cat(parts)[inverse]
        channels = stacked.shape[2:]
        grid = stacked.reshape(tiles_y, tiles_x, tile_size, tile_size, *channels)
        image = grid.transpose(1, 2).reshape(tiles_y * tile_size, tiles_x * tile_size, *channels)
        return image[: camera.height, : camera.width]

    return TorchRender(
        color=assemble(colors),
        depth=assemble(depths),
        alpha=assemble(alphas),
        means2d=means2d,
        cov2d=cov2d,
        visible=visible,
    )


def background_tensor(background: np.ndarray | tuple[float, float, float]) -> torch.Tensor:
    return torch.as_tensor(np.asarray(background, dtype=np.float64))
