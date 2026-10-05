"""Image metrics and the photometric loss, in PyTorch so that the loss is differentiable.

SSIM follows Wang et al., "Image Quality Assessment: From Error Visibility to Structural
Similarity", IEEE TIP 13(4), 2004: an 11 x 11 Gaussian window with sigma = 1.5, constants
``C1 = (0.01 L)^2`` and ``C2 = (0.03 L)^2`` for a dynamic range ``L = 1``, and the mean over
every window position that fits entirely inside the image (no padding).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_C1 = 0.01**2
SSIM_C2 = 0.03**2

#: Weight of the D-SSIM term in the training loss (Kerbl et al. 2023, eq. 7, lambda = 0.2).
DSSIM_WEIGHT = 0.2


def psnr(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB for images in [0, 1]: ``-10 log10(MSE)``."""
    return -10.0 * torch.log10(torch.mean((image - target) ** 2))


def _gaussian_kernel(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    x = torch.arange(SSIM_WINDOW, dtype=dtype, device=device) - (SSIM_WINDOW - 1) / 2.0
    kernel = torch.exp(-0.5 * (x / SSIM_SIGMA) ** 2)
    return kernel / kernel.sum()


def _blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian filter of a (C, 1, H, W) batch, without padding."""
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    return F.conv2d(x, kernel.view(1, 1, 1, -1))


def ssim(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean structural similarity of two (H, W, C) images in [0, 1]."""
    kernel = _gaussian_kernel(image.dtype, image.device)
    x = image.permute(2, 0, 1).unsqueeze(1)
    y = target.permute(2, 0, 1).unsqueeze(1)
    mu_x, mu_y = _blur(x, kernel), _blur(y, kernel)
    var_x = _blur(x * x, kernel) - mu_x**2
    var_y = _blur(y * y, kernel) - mu_y**2
    cov_xy = _blur(x * y, kernel) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + SSIM_C1) * (2.0 * cov_xy + SSIM_C2)
    denominator = (mu_x**2 + mu_y**2 + SSIM_C1) * (var_x + var_y + SSIM_C2)
    return (numerator / denominator).mean()


def photometric_loss(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``(1 - lambda) L1 + lambda (1 - SSIM)`` (Kerbl et al. 2023, eq. 7)."""
    l1 = torch.mean(torch.abs(image - target))
    return (1.0 - DSSIM_WEIGHT) * l1 + DSSIM_WEIGHT * (1.0 - ssim(image, target))
