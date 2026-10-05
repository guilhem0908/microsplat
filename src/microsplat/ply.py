"""Read and write Gaussian scenes in the ``.ply`` layout of the INRIA reference code.

That layout is what 3DGS viewers and trainers exchange: one binary little-endian float32
vertex per Gaussian holding the *unconstrained* parameters, i.e. log-scales, opacity logits
and unnormalised quaternions ``(w, x, y, z)``, plus an unused normal. Only the degree-0
spherical-harmonic coefficients are stored here (no ``f_rest_*`` properties).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from microsplat.gaussians import GaussianCloud

PLY_PROPERTIES = (
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)  # fmt: skip
_END_HEADER = b"end_header\n"


def _header(count: int) -> bytes:
    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    lines += [f"property float {name}" for name in PLY_PROPERTIES]
    return ("\n".join(lines) + "\n").encode("ascii") + _END_HEADER


def save_ply(path: str | Path, cloud: GaussianCloud) -> None:
    """Write ``cloud`` to ``path`` as float32."""
    table = np.concatenate(
        [
            cloud.means,
            np.zeros_like(cloud.means),
            cloud.sh_dc,
            cloud.opacity_logits[:, None],
            cloud.log_scales,
            cloud.quats,
        ],
        axis=1,
    )
    with open(path, "wb") as handle:
        handle.write(_header(len(cloud)))
        handle.write(table.astype("<f4").tobytes())


def load_ply(path: str | Path) -> GaussianCloud:
    """Read a degree-0 scene written by :func:`save_ply` or by another 3DGS tool.

    Properties are located by name, so extra float properties such as ``f_rest_*`` are
    skipped.
    """
    data = Path(path).read_bytes()
    split = data.index(_END_HEADER) + len(_END_HEADER)
    header = data[:split].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("only binary little-endian PLY files are supported")
    count = next(int(line.split()[2]) for line in header if line.startswith("element vertex"))
    properties = [line.split() for line in header if line.startswith("property")]
    if any(kind != "float" for _, kind, _ in properties):
        raise ValueError("only float32 properties are supported")
    names = [name for _, _, name in properties]
    table = np.frombuffer(data, dtype="<f4", count=count * len(names), offset=split)
    table = table.reshape(count, len(names)).astype(np.float64)

    def columns(*wanted: str) -> np.ndarray:
        return table[:, [names.index(name) for name in wanted]]

    return GaussianCloud(
        means=columns("x", "y", "z"),
        log_scales=columns("scale_0", "scale_1", "scale_2"),
        quats=columns("rot_0", "rot_1", "rot_2", "rot_3"),
        opacity_logits=columns("opacity")[:, 0],
        sh_dc=columns("f_dc_0", "f_dc_1", "f_dc_2"),
    )
