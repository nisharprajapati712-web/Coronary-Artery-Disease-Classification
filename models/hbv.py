"""
Hessian-Based Vesselness (HBV) filter.

Enhances tubular/coronary vessel structures in CCTA images while suppressing
background noise.  Multi-scale Hessian eigenvalue analysis is used following
the Frangi vesselness formulation described in the paper.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _hessian_2d(image: np.ndarray, sigma: float) -> np.ndarray:
    """Compute Hessian matrix components at a single scale."""
    # Gaussian smoothing at the given scale
    smoothed = ndimage.gaussian_filter(image, sigma=sigma, mode="reflect")

    # Second-order derivatives
    dxx = ndimage.gaussian_filter(smoothed, sigma=sigma, order=[2, 0], mode="reflect")
    dyy = ndimage.gaussian_filter(smoothed, sigma=sigma, order=[0, 2], mode="reflect")
    dxy = ndimage.gaussian_filter(smoothed, sigma=sigma, order=[1, 1], mode="reflect")

    return np.stack([dxx, dxy, dxy, dyy], axis=-1).reshape(
        image.shape[0], image.shape[1], 2, 2
    )


def _eigvalsh_2d(hessian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of 2x2 Hessian at each pixel (sorted ascending)."""
    a = hessian[..., 0, 0]
    b = hessian[..., 0, 1]
    c = hessian[..., 1, 1]

    trace = a + c
    disc = np.sqrt(np.maximum((a - c) ** 2 + 4.0 * b**2, 0.0))

    lambda1 = (trace - disc) / 2.0   # smaller eigenvalue
    lambda2 = (trace + disc) / 2.0   # larger eigenvalue
    return lambda1, lambda2


def vesselness_response(
    image: np.ndarray,
    scales: list[float] | None = None,
    beta: float = 0.5,
    gamma: float = 0.5,
    c: float = 1e-10,
) -> np.ndarray:
    """
    Compute multi-scale vesselness response (Eqs. 2-3 in the paper).

    Args:
        image: 2D grayscale CCTA image, float in [0, 1].
        scales: list of Gaussian sigma values.
        beta:   sensitivity parameter for blobness ratio (RB).
        gamma:  sensitivity parameter for structuredness (S).
        c:      small constant for numerical stability.

    Returns:
        2D vesselness-enhanced image in [0, 1].
    """
    if scales is None:
        scales = config.HBV_SCALES

    image = image.astype(np.float64)
    vesselness = np.zeros_like(image)

    for sigma in scales:
        hessian = _hessian_2d(image, sigma)
        lambda1, lambda2 = _eigvalsh_2d(hessian)

        # Blobness measure RB (Eq. 3 in paper)
        denom = np.maximum(np.abs(lambda1), c)
        rb2 = (lambda2 / denom) ** 2

        # Structuredness measure S
        s2 = lambda1**2 + lambda2**2

        # Vesselness filter response (Frangi-style)
        vessel_scale = np.zeros_like(image)
        mask = (lambda2 < 0) & (lambda1 < 0)
        if np.any(mask):
            vessel_scale[mask] = (
                np.exp(-rb2[mask] / (2.0 * beta**2))
                * (1.0 - np.exp(-s2[mask] / (2.0 * gamma**2)))
            )

        # Take pixel-wise maximum across scales
        vesselness = np.maximum(vesselness, vessel_scale)

    # Normalize to [0, 1]
    vmin, vmax = vesselness.min(), vesselness.max()
    if vmax > vmin:
        vesselness = (vesselness - vmin) / (vmax - vmin)

    return vesselness.astype(np.float32)


def apply_hbv(
    image: np.ndarray,
    scales: list[float] | None = None,
    beta: float = 0.5,
    gamma: float = 0.5,
) -> np.ndarray:
    """
    Apply the HBV filter to a single image (or single image batch channel).

    Accepts a 2D array or a 3D array (batch).  Normalizes input to [0, 1]
    internally and returns the enhanced image with the same shape.
    """
    if image.ndim == 2:
        img = image.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
        return vesselness_response(img, scales, beta, gamma)

    if image.ndim == 3:
        out = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            out[i] = apply_hbv(image[i], scales, beta, gamma)
        return out

    raise ValueError(f"Expected 2D or 3D input, got {image.ndim}D")


if __name__ == "__main__":
    # Quick smoke test
    rng = np.random.default_rng(0)
    test = rng.random((64, 64)).astype(np.float32)
    result = apply_hbv(test)
    print("HBV output shape:", result.shape, "range:", result.min(), "-", result.max())