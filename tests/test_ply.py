"""The exported file follows the INRIA layout and survives a round trip."""

from __future__ import annotations

import numpy as np

from microsplat import reference, verify
from microsplat.ply import PLY_PROPERTIES, load_ply, save_ply

FLOAT32_BYTES = 4


def test_header_lists_the_standard_properties_in_order(tmp_path):
    path = tmp_path / "scene.ply"
    save_ply(path, verify.random_scene(7, seed=0))
    header = path.read_bytes().split(b"end_header\n")[0].decode("ascii").splitlines()
    assert header[:3] == ["ply", "format binary_little_endian 1.0", "element vertex 7"]
    assert header[3:] == [f"property float {name}" for name in PLY_PROPERTIES]
    assert PLY_PROPERTIES[:3] == ("x", "y", "z")
    assert PLY_PROPERTIES[-4:] == ("rot_0", "rot_1", "rot_2", "rot_3")


def test_payload_is_one_float32_record_per_gaussian(tmp_path):
    path = tmp_path / "scene.ply"
    save_ply(path, verify.random_scene(11, seed=1))
    payload = path.read_bytes().split(b"end_header\n", 1)[1]
    assert len(payload) == 11 * len(PLY_PROPERTIES) * FLOAT32_BYTES


def test_file_stores_unconstrained_parameters(tmp_path):
    cloud = verify.random_scene(5, seed=2)
    path = tmp_path / "scene.ply"
    save_ply(path, cloud)
    payload = path.read_bytes().split(b"end_header\n", 1)[1]
    table = np.frombuffer(payload, dtype="<f4").reshape(5, len(PLY_PROPERTIES))
    column = PLY_PROPERTIES.index
    np.testing.assert_allclose(table[:, column("opacity")], cloud.opacity_logits, rtol=1e-6)
    np.testing.assert_allclose(table[:, column("scale_0")], np.log(cloud.scales[:, 0]), rtol=1e-6)
    np.testing.assert_allclose(table[:, column("rot_0")], cloud.quats[:, 0], rtol=1e-6)
    assert not table[:, column("nx") : column("nz") + 1].any()


def test_round_trip_preserves_parameters_and_render(tmp_path, camera, background):
    cloud = verify.random_scene(40, seed=3)
    path = tmp_path / "scene.ply"
    save_ply(path, cloud)
    loaded = load_ply(path)
    for name in ("means", "log_scales", "quats", "opacity_logits", "sh_dc"):
        np.testing.assert_allclose(getattr(loaded, name), getattr(cloud, name), rtol=1e-6)
    before = reference.render(cloud, camera, background).color
    after = reference.render(loaded, camera, background).color
    # float32 storage moves the image by rounding only.
    assert np.abs(after - before).max() < 1e-5
