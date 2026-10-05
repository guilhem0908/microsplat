"""Pinhole camera model and orbit view sampling.

Conventions (the OpenCV / COLMAP ones): the camera looks along +z, x points right and y points
down. A world point ``p`` has camera coordinates ``t = R p + T`` and continuous image
coordinates ``u = fx * t_x / t_z + cx``, ``v = fy * t_y / t_z + cy``. The pixel in column ``i``
and row ``j`` covers ``[i, i+1) x [j, j+1)``, so its centre is ``(i + 0.5, j + 0.5)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

WORLD_UP = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Camera:
    """A posed pinhole camera.

    Attributes:
        rotation: (3, 3) world-to-camera rotation ``R``.
        translation: (3,) world-to-camera translation ``T``.
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in continuous image coordinates.
        width, height: Image size in pixels.
    """

    rotation: np.ndarray
    translation: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def position(self) -> np.ndarray:
        """Camera centre in world coordinates, ``-R^T T``."""
        return -self.rotation.T @ self.translation

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        """Map (N, 3) world points to camera coordinates."""
        return points @ self.rotation.T + self.translation

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (N, 3) world points; returns (N, 2) image coordinates and (N,) depths."""
        cam = self.world_to_camera(points)
        depth = cam[:, 2]
        u = self.fx * cam[:, 0] / depth + self.cx
        v = self.fy * cam[:, 1] / depth + self.cy
        return np.stack([u, v], axis=1), depth

    def pixel_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the (H, W) grids of pixel-centre coordinates ``u`` and ``v``."""
        u = np.arange(self.width, dtype=np.float64) + 0.5
        v = np.arange(self.height, dtype=np.float64) + 0.5
        return np.meshgrid(u, v, indexing="xy")

    def pixel_rays(self, offsets: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
        """World-space unit directions of the rays through every pixel, shape (H, W, 3).

        ``offsets`` is the position inside the pixel, (0.5, 0.5) being the centre.
        """
        u = np.arange(self.width, dtype=np.float64) + offsets[0]
        v = np.arange(self.height, dtype=np.float64) + offsets[1]
        uu, vv = np.meshgrid(u, v, indexing="xy")
        cam_dirs = np.stack(
            [(uu - self.cx) / self.fx, (vv - self.cy) / self.fy, np.ones_like(uu)], axis=-1
        )
        world_dirs = cam_dirs @ self.rotation  # row-vector form of R^T d
        return world_dirs / np.linalg.norm(world_dirs, axis=-1, keepdims=True)

    def resized(self, scale: float) -> Camera:
        """Same pose and field of view at ``scale`` times the resolution."""
        return replace(
            self,
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
            width=round(self.width * scale),
            height=round(self.height * scale),
        )


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    *,
    width: int,
    height: int,
    fov_x_deg: float,
    up: tuple[float, float, float] = WORLD_UP,
) -> Camera:
    """Build a camera at ``eye`` looking at ``target`` with square pixels.

    The camera axes expressed in the world frame are ``z = forward``, ``x = forward x up``
    (right) and ``y = z x x`` (down); they are the rows of ``R``.
    """
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    focal = 0.5 * width / np.tan(0.5 * np.deg2rad(fov_x_deg))
    return Camera(
        rotation=rotation,
        translation=-rotation @ eye,
        fx=focal,
        fy=focal,
        cx=0.5 * width,
        cy=0.5 * height,
        width=width,
        height=height,
    )


def orbit_cameras(
    count: int,
    *,
    radius: float,
    elevation_deg: float,
    target: np.ndarray,
    width: int,
    height: int,
    fov_x_deg: float,
    azimuth_offset_deg: float = 0.0,
) -> list[Camera]:
    """``count`` cameras evenly spaced in azimuth on a circle around the world z axis."""
    target = np.asarray(target, dtype=np.float64)
    elevation = np.deg2rad(elevation_deg)
    cameras = []
    for k in range(count):
        azimuth = np.deg2rad(azimuth_offset_deg) + 2.0 * np.pi * k / count
        offset = radius * np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )
        cameras.append(
            look_at(target + offset, target, width=width, height=height, fov_x_deg=fov_x_deg)
        )
    return cameras
