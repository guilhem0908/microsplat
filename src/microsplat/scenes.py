"""Self-contained ground-truth scenes: nothing to download, exact poses, exact depth.

Two scenes with different difficulty:

* ``shapes`` is an analytic ray tracer (three Lambertian spheres on a checkered disc, one
  directional light with hard shadows). Its surfaces are opaque and sharp-edged, so no finite
  set of Gaussians reproduces it exactly, and its depth is known in closed form.
* ``knot`` is a hidden set of Gaussians laid along a trefoil knot and rendered by the NumPy
  reference rasteriser. The model can represent it exactly, which isolates the optimisation
  from the representation error.

Both are view-independent, as degree-0 spherical harmonics require.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np

from microsplat import reference
from microsplat.camera import Camera, orbit_cameras
from microsplat.gaussians import GaussianCloud
from microsplat.settings import RasterSettings

SCENE_NAMES = ("shapes", "knot")

# Camera rig shared by both scenes.
ORBIT_RADIUS = 4.4
ORBIT_TARGET = (0.0, 0.0, 0.25)
FOV_X_DEG = 42.0
TRAIN_ELEVATIONS_DEG = (20.0, 50.0)
TRAIN_VIEWS_PER_RING = 12
HELD_OUT_ELEVATION_DEG = 35.0
HELD_OUT_VIEWS = 8

# shapes: geometry and shading.
DISC_RADIUS = 1.4
CHECKER_SIZE = 0.5
CHECKER_ALBEDOS = ((0.85, 0.85, 0.80), (0.30, 0.36, 0.44))
SPHERES = (  # centre, radius, albedo
    ((-0.55, -0.35, 0.45), 0.45, (0.85, 0.22, 0.18)),
    ((0.60, -0.20, 0.32), 0.32, (0.20, 0.42, 0.88)),
    ((0.05, 0.62, 0.25), 0.25, (0.95, 0.76, 0.15)),
)
LIGHT_DIRECTION = (0.40, -0.30, 0.85)
AMBIENT = 0.35
DIFFUSE = 0.65
SHADOW_BIAS = 1e-6
SUPERSAMPLING = 4  # rays per pixel side used to anti-alias the target images
SHAPES_BACKGROUND = (0.06, 0.07, 0.10)
SHAPES_BOUNDS = ((-1.5, -1.5, -0.2), (1.5, 1.5, 1.0))

# knot: the hidden Gaussian set.
KNOT_GAUSSIANS = 160
KNOT_SIZE = 0.42
KNOT_CENTRE_HEIGHT = 0.30
KNOT_SCALES = (0.11, 0.045, 0.045)  # along the tangent, then across
KNOT_OPACITY = 0.9
KNOT_BACKGROUND = (0.06, 0.07, 0.10)
KNOT_BOUNDS = ((-1.4, -1.4, -0.3), (1.4, 1.4, 0.9))


@dataclass
class View:
    """One posed ground-truth observation."""

    camera: Camera
    image: np.ndarray  # (H, W, 3) in [0, 1]
    depth: np.ndarray  # (H, W) camera-space z of the visible surface, NaN on the background


@dataclass
class Scene:
    name: str
    background: np.ndarray
    box_min: np.ndarray
    box_max: np.ndarray
    hidden_cloud: GaussianCloud | None = None

    @property
    def extent(self) -> float:
        """Half of the bounding-box diagonal; the length scale of learning rates."""
        return 0.5 * float(np.linalg.norm(self.box_max - self.box_min))

    def observe(self, camera: Camera) -> View:
        if self.hidden_cloud is None:
            return _ray_trace(camera, self.background)
        out = reference.render(self.hidden_cloud, camera, self.background, RasterSettings())
        solid = out.alpha > 0.5
        depth = np.where(solid, out.depth / np.where(solid, out.alpha, 1.0), np.nan)
        return View(camera, out.color, depth)


def make_scene(name: str) -> Scene:
    if name == "shapes":
        return Scene(name, np.array(SHAPES_BACKGROUND), *map(np.array, SHAPES_BOUNDS))
    if name == "knot":
        bounds = map(np.array, KNOT_BOUNDS)
        return Scene(name, np.array(KNOT_BACKGROUND), *bounds, hidden_cloud=_knot_cloud())
    raise ValueError(f"unknown scene {name!r}, expected one of {SCENE_NAMES}")


def training_cameras(resolution: int) -> list[Camera]:
    """Two rings of cameras; the second ring is rotated by half a step."""
    cameras = []
    for ring, elevation in enumerate(TRAIN_ELEVATIONS_DEG):
        cameras += orbit_cameras(
            TRAIN_VIEWS_PER_RING,
            radius=ORBIT_RADIUS,
            elevation_deg=elevation,
            target=ORBIT_TARGET,
            width=resolution,
            height=resolution,
            fov_x_deg=FOV_X_DEG,
            azimuth_offset_deg=ring * 180.0 / TRAIN_VIEWS_PER_RING,
        )
    return cameras


def held_out_cameras(resolution: int, count: int = HELD_OUT_VIEWS) -> list[Camera]:
    """Held-out ring at an elevation and at azimuths that no training camera uses."""
    return orbit_cameras(
        count,
        radius=ORBIT_RADIUS,
        elevation_deg=HELD_OUT_ELEVATION_DEG,
        target=ORBIT_TARGET,
        width=resolution,
        height=resolution,
        fov_x_deg=FOV_X_DEG,
        azimuth_offset_deg=90.0 / TRAIN_VIEWS_PER_RING,
    )


def _ray_trace(camera: Camera, background: np.ndarray) -> View:
    """Anti-aliased image and centre-ray depth of the ``shapes`` scene."""
    steps = (np.arange(SUPERSAMPLING) + 0.5) / SUPERSAMPLING
    image = np.zeros((camera.height, camera.width, 3))
    for oy in steps:
        for ox in steps:
            colour, _ = _trace(camera.position, camera.pixel_rays((ox, oy)), background)
            image += colour
    image /= SUPERSAMPLING**2
    directions = camera.pixel_rays()
    _, distance = _trace(camera.position, directions, background)
    # Distance along the ray -> depth along the optical axis (third row of R).
    depth = distance * (directions @ camera.rotation[2])
    return View(camera, image, depth)


def _trace(
    origin: np.ndarray, directions: np.ndarray, background: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Shade unit rays ``origin + s * d``; returns colours and hit distances (NaN = miss)."""
    nearest = np.full(directions.shape[:-1], np.inf)
    albedo = np.zeros(directions.shape)
    normal = np.zeros(directions.shape)

    # Checkered disc in the plane z = 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -origin[2] / directions[..., 2]
        hit = origin + s[..., None] * directions
        on_disc = (s > 0) & (hit[..., 0] ** 2 + hit[..., 1] ** 2 <= DISC_RADIUS**2)
    parity = np.floor(hit[..., 0] / CHECKER_SIZE) + np.floor(hit[..., 1] / CHECKER_SIZE)
    nearest = np.where(on_disc, s, nearest)
    albedo[on_disc] = np.asarray(CHECKER_ALBEDOS)[parity[on_disc].astype(int) % 2]
    normal[on_disc] = (0.0, 0.0, 1.0)

    for centre, radius, colour in SPHERES:
        s = _sphere_hit(origin, directions, np.asarray(centre), radius)
        closer = s < nearest
        nearest = np.where(closer, s, nearest)
        albedo[closer] = colour
        normal[closer] = (origin + s[closer, None] * directions[closer] - centre) / radius

    light = np.asarray(LIGHT_DIRECTION) / np.linalg.norm(LIGHT_DIRECTION)
    surface = np.isfinite(nearest)
    points = origin + np.where(surface, nearest, 0.0)[..., None] * directions
    lit = np.ones(directions.shape[:-1], dtype=bool)
    for centre, radius, _ in SPHERES:
        to_light = np.broadcast_to(light, points.shape)
        blocker = _sphere_hit(points + SHADOW_BIAS * normal, to_light, np.asarray(centre), radius)
        lit &= ~np.isfinite(blocker)
    lambert = np.clip(normal @ light, 0.0, None) * lit
    shaded = albedo * (AMBIENT + DIFFUSE * lambert)[..., None]
    colour = np.where(surface[..., None], shaded, background)
    return colour, np.where(surface, nearest, np.nan)


def _sphere_hit(
    origins: np.ndarray, directions: np.ndarray, centre: np.ndarray, radius: float
) -> np.ndarray:
    """Smallest positive ``s`` with ``|o + s d - c| = r`` for unit ``d``, or inf."""
    oc = origins - centre
    half_b = (oc * directions).sum(axis=-1)
    disc = half_b**2 - ((oc * oc).sum(axis=-1) - radius**2)
    root = np.sqrt(np.where(disc > 0, disc, np.nan))
    s = -half_b - root
    return np.where(np.isfinite(s) & (s > 0), s, np.inf)


def _knot_cloud() -> GaussianCloud:
    """Elongated Gaussians along a trefoil knot, coloured by position along the curve."""
    s = np.linspace(0.0, 2.0 * np.pi, KNOT_GAUSSIANS, endpoint=False)
    curve = np.stack(
        [np.sin(s) + 2 * np.sin(2 * s), np.cos(s) - 2 * np.cos(2 * s), -np.sin(3 * s)], axis=1
    )
    tangent = np.stack(
        [np.cos(s) + 4 * np.cos(2 * s), -np.sin(s) + 4 * np.sin(2 * s), -3 * np.cos(3 * s)], axis=1
    )
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    means = KNOT_SIZE * curve + np.array([0.0, 0.0, KNOT_CENTRE_HEIGHT])
    # Shortest-arc quaternion taking the x axis onto the tangent: (1 + x.t, x cross t).
    quats = np.stack([1.0 + tangent[:, 0], np.zeros_like(s), -tangent[:, 2], tangent[:, 1]], axis=1)
    colours = np.array([colorsys.hsv_to_rgb(v / (2.0 * np.pi), 0.75, 0.95) for v in s])
    return GaussianCloud.from_activated(
        means=means,
        scales=np.tile(KNOT_SCALES, (KNOT_GAUSSIANS, 1)),
        quats=quats,
        opacities=np.full(KNOT_GAUSSIANS, KNOT_OPACITY),
        colours=colours,
    )
