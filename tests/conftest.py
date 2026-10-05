"""Shared fixtures: a camera that sees the random test scenes, and a background colour."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from microsplat.camera import Camera
from microsplat.verify import CHECK_BACKGROUND, check_camera

TEST_THREADS = 4


@pytest.fixture(autouse=True, scope="session")
def _few_torch_threads() -> None:
    """The tensors in these tests are small; many threads only add scheduling noise."""
    torch.set_num_threads(TEST_THREADS)


@pytest.fixture
def camera() -> Camera:
    return check_camera(40, 28)


@pytest.fixture
def background() -> np.ndarray:
    return np.array(CHECK_BACKGROUND)
