"""PSNR and SSIM against closed forms and a direct sliding-window implementation."""

from __future__ import annotations

import numpy as np
import torch

from microsplat.metrics import (
    DSSIM_WEIGHT,
    SSIM_C1,
    SSIM_C2,
    SSIM_SIGMA,
    SSIM_WINDOW,
    photometric_loss,
    psnr,
    ssim,
)


def _images(seed: int, height: int = 24, width: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    a = torch.rand((height, width, 3), dtype=torch.float64, generator=generator)
    b = (a + 0.1 * torch.randn(a.shape, dtype=torch.float64, generator=generator)).clamp(0, 1)
    return a, b


def _ssim_by_sliding_window(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM computed window by window, straight from the definition."""
    x = np.arange(SSIM_WINDOW) - (SSIM_WINDOW - 1) / 2.0
    g = np.exp(-0.5 * (x / SSIM_SIGMA) ** 2)
    window = np.outer(g, g) / np.outer(g, g).sum()
    values = []
    for channel in range(a.shape[2]):
        for i in range(a.shape[0] - SSIM_WINDOW + 1):
            for j in range(a.shape[1] - SSIM_WINDOW + 1):
                p = a[i : i + SSIM_WINDOW, j : j + SSIM_WINDOW, channel]
                q = b[i : i + SSIM_WINDOW, j : j + SSIM_WINDOW, channel]
                mu_p, mu_q = (window * p).sum(), (window * q).sum()
                var_p = (window * p * p).sum() - mu_p**2
                var_q = (window * q * q).sum() - mu_q**2
                cov = (window * p * q).sum() - mu_p * mu_q
                values.append(
                    (2 * mu_p * mu_q + SSIM_C1)
                    * (2 * cov + SSIM_C2)
                    / ((mu_p**2 + mu_q**2 + SSIM_C1) * (var_p + var_q + SSIM_C2))
                )
    return float(np.mean(values))


def test_psnr_of_a_uniform_error_has_a_closed_form():
    target = torch.full((8, 8, 3), 0.5, dtype=torch.float64)
    assert np.isclose(psnr(target + 0.1, target).item(), 20.0)
    assert np.isclose(psnr(target + 0.01, target).item(), 40.0)


def test_ssim_of_identical_images_is_one():
    a, _ = _images(0)
    assert np.isclose(ssim(a, a).item(), 1.0)


def test_ssim_is_symmetric_and_drops_with_noise():
    a, b = _images(1)
    assert np.isclose(ssim(a, b).item(), ssim(b, a).item())
    noisier = (a + 0.3 * torch.randn(a.shape, dtype=torch.float64)).clamp(0, 1)
    assert ssim(a, noisier) < ssim(a, b) < 1.0


def test_ssim_matches_a_direct_sliding_window_implementation():
    a, b = _images(2)
    assert np.isclose(ssim(a, b).item(), _ssim_by_sliding_window(a.numpy(), b.numpy()), atol=1e-12)


def test_photometric_loss_is_zero_only_for_a_perfect_image():
    a, b = _images(3)
    assert np.isclose(photometric_loss(a, a).item(), 0.0, atol=1e-12)
    expected = (1 - DSSIM_WEIGHT) * (a - b).abs().mean() + DSSIM_WEIGHT * (1 - ssim(a, b))
    assert np.isclose(photometric_loss(a, b).item(), expected.item())
    assert photometric_loss(a, b).item() > 0.0
