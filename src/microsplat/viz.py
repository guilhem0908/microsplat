"""Figures: projected-ellipse overlay, error map, four-panel frames, GIF and PSNR plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from microsplat import reference
from microsplat.camera import Camera
from microsplat.gaussians import GaussianCloud
from microsplat.settings import RasterSettings

PANEL_SIZE = 192
PANEL_GAP = 8
HEADER_HEIGHT = 22
FOOTER_HEIGHT = 24
FONT_SIZE = 13
CANVAS_COLOUR = (18, 20, 26)
TEXT_COLOUR = (225, 228, 235)
MUTED_COLOUR = (150, 155, 168)
PANEL_TITLES = ("target", "render", "abs. error x4", "projected ellipses")

ERROR_GAIN = 4.0
ELLIPSE_SIGMAS = 1.5  # contour drawn for each Gaussian, in standard deviations
ELLIPSE_SEGMENTS = 24
ELLIPSE_MIN_OPACITY = 0.05
GIF_COLOURS = 96

# Plot styling: light surface, two categorical hues, recessive grid.
PLOT_SURFACE = "#fcfcfb"
PLOT_TEXT = "#0b0b0b"
PLOT_TEXT_SECONDARY = "#52514e"
PLOT_GRID = "#e6e5e0"
VARIANT_COLOURS = {"adaptive": "#2a78d6", "fixed": "#eb6834"}
VARIANT_LABELS = {"adaptive": "adaptive density control", "fixed": "fixed set of Gaussians"}


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def error_map(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Mean absolute error over channels, amplified and mapped to a perceptual colour map."""
    error = np.abs(np.clip(image, 0.0, 1.0) - target).mean(axis=-1)
    return matplotlib.colormaps["magma"](np.clip(ERROR_GAIN * error, 0.0, 1.0))[..., :3]


def ellipse_overlay(cloud: GaussianCloud, camera: Camera, size: int) -> Image.Image:
    """Outline of every projected Gaussian, far ones first, in the Gaussian's own colour.

    The contour ``(x - mu')^T Sigma'^-1 (x - mu') = ELLIPSE_SIGMAS^2`` is traced from the
    eigen-decomposition of the projected covariance.
    """
    settings = RasterSettings()
    scale = size / camera.width
    proj = reference.project_gaussians(cloud, camera, settings.low_pass)
    canvas = Image.new("RGB", (size, size), CANVAS_COLOUR)
    draw = ImageDraw.Draw(canvas)
    angles = np.linspace(0.0, 2.0 * np.pi, ELLIPSE_SEGMENTS, endpoint=False)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    opacities, colours = cloud.opacities, to_uint8(cloud.colours)
    for i in np.argsort(-proj.depth):
        if proj.depth[i] <= settings.near or opacities[i] < ELLIPSE_MIN_OPACITY:
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(proj.cov2d[i])
        axes = eigenvectors * (ELLIPSE_SIGMAS * np.sqrt(eigenvalues))
        outline = (proj.means2d[i][:, None] + axes @ circle) * scale
        draw.polygon([tuple(p) for p in outline.T], outline=tuple(int(c) for c in colours[i]))
    return canvas


def _panel(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(to_uint8(image))
    if image.size != (PANEL_SIZE, PANEL_SIZE):
        image = image.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.BICUBIC)
    return image


def four_panel_frame(
    target: np.ndarray, image: np.ndarray, cloud: GaussianCloud, camera: Camera, caption: str
) -> Image.Image:
    """Target, render, amplified error and ellipse overlay side by side, with a caption."""
    panels = [
        _panel(target),
        _panel(image),
        _panel(error_map(image, target)),
        ellipse_overlay(cloud, camera, PANEL_SIZE),
    ]
    width = len(panels) * PANEL_SIZE + (len(panels) + 1) * PANEL_GAP
    height = HEADER_HEIGHT + PANEL_SIZE + FOOTER_HEIGHT
    frame = Image.new("RGB", (width, height), CANVAS_COLOUR)
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=FONT_SIZE)
    for k, (panel, title) in enumerate(zip(panels, PANEL_TITLES, strict=True)):
        left = PANEL_GAP + k * (PANEL_SIZE + PANEL_GAP)
        frame.paste(panel, (left, HEADER_HEIGHT))
        draw.text((left + 2, 3), title, fill=MUTED_COLOUR, font=font)
    draw.text((PANEL_GAP + 2, HEADER_HEIGHT + PANEL_SIZE + 4), caption, fill=TEXT_COLOUR, font=font)
    return frame


def save_gif(frames: list[Image.Image], durations_ms: list[int], path: str | Path) -> None:
    """Write an endlessly looping GIF; frames are quantised without dithering to stay small."""
    quantised = [frame.quantize(colors=GIF_COLOURS, dither=Image.Dither.NONE) for frame in frames]
    quantised[0].save(
        path,
        save_all=True,
        append_images=quantised[1:],
        duration=durations_ms,
        loop=0,
        optimize=True,
        disposal=1,
    )


def plot_psnr_curves(history: list[dict], scene_names: tuple[str, ...], path: str | Path) -> None:
    """Held-out PSNR against iteration: one panel per scene, one colour per variant.

    ``history`` holds one record per (scene, variant, seed, iteration). Thin lines are the
    individual seeds, the thick line is their mean.
    """
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figure, axes = plt.subplots(1, len(scene_names), figsize=(4.3 * len(scene_names), 3.3))
    figure.patch.set_facecolor(PLOT_SURFACE)
    for axis, scene in zip(np.atleast_1d(axes), scene_names, strict=True):
        axis.set_facecolor(PLOT_SURFACE)
        for variant, colour in VARIANT_COLOURS.items():
            rows = [h for h in history if h["scene"] == scene and h["variant"] == variant]
            seeds = sorted({h["seed"] for h in rows})
            if not seeds:
                continue
            curves = []
            for seed in seeds:
                points = sorted((h["iteration"], h["psnr"]) for h in rows if h["seed"] == seed)
                iterations, values = map(np.array, zip(*points, strict=True))
                axis.plot(iterations, values, color=colour, linewidth=0.8, alpha=0.4)
                curves.append(values)
            mean = np.mean(curves, axis=0)
            axis.plot(
                iterations,
                mean,
                color=colour,
                linewidth=2.0,
                solid_capstyle="round",
                label=VARIANT_LABELS[variant],
            )
            axis.annotate(
                f"{mean[-1]:.1f} dB",
                (iterations[-1], mean[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                color=PLOT_TEXT,
                fontsize=8,
            )
        axis.set_title(scene, loc="left", color=PLOT_TEXT, fontsize=10, fontweight="bold")
        axis.set_title(
            f"thin: each of {len(seeds)} seeds, thick: mean",
            loc="right",
            color=PLOT_TEXT_SECONDARY,
            fontsize=8,
        )
        axis.set_xlabel("iteration", color=PLOT_TEXT_SECONDARY)
        axis.set_ylabel("held-out PSNR (dB)", color=PLOT_TEXT_SECONDARY)
        axis.grid(True, color=PLOT_GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(colors=PLOT_TEXT_SECONDARY, length=0)
        for side in ("top", "right", "left", "bottom"):
            axis.spines[side].set_visible(False)
        axis.margins(x=0.0)
        axis.set_xlim(right=iterations[-1] * 1.14)
        axis.legend(frameon=False, loc="lower right", labelcolor=PLOT_TEXT_SECONDARY, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=PLOT_SURFACE)
    plt.close(figure)
