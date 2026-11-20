"""
Synthetic CCTA-like image generator.

Generates coronary artery images resembling CCTA scans with:
  - Tubular vessel structures (coronary arteries) with branching
  - Atherosclerotic plaque features (bright/high-intensity regions) for CAD class
  - Imaging noise, motion artifacts, and contrast variations
  - Ground-truth vessel masks for DDPM segmentation training

This allows the complete DSCNN-SAM pipeline to run end-to-end without the
Kaggle CCTA dataset.  When real CCTA data is available, set
config.USE_SYNTHETIC_DATA = False.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _draw_vessel(
    img: np.ndarray,
    mask: np.ndarray,
    start: tuple,
    length: int,
    angle: float,
    width: float,
    intensity: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a curved vessel segment onto the image and mask."""
    h, w = img.shape
    x, y = start
    steps = max(2, length // 3)
    for i in range(steps):
        t = i / steps
        # Curved path
        curvature = np.sin(t * np.pi * 2) * 15.0
        px = int(np.clip(x + np.cos(np.radians(angle)) * length * t + curvature * (1 - t), 0, w - 1))
        py = int(np.clip(y + np.sin(np.radians(angle)) * length * t + curvature * t, 0, h - 1))
        r = max(1, int(width * (1.0 - 0.5 * t)))
        # Draw filled circle for vessel
        y0 = max(0, py - r)
        y1 = min(h, py + r)
        x0 = max(0, px - r)
        x1 = min(w, px + r)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if (xx - px) ** 2 + (yy - py) ** 2 <= r * r:
                    img[yy, xx] = intensity
                    mask[yy, xx] = 1.0
    return img, mask


def _draw_plaque(img: np.ndarray, center: tuple, radius: int, intensity: float = 0.95):
    """Draw an atherosclerotic plaque (bright blob) at the given center."""
    h, w = img.shape
    cx, cy = center
    for yy in range(max(0, cy - radius), min(h, cy + radius)):
        for xx in range(max(0, cx - radius), min(w, cx + radius)):
            if (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius:
                img[yy, xx] = intensity


def generate_sample(
    rng: np.random.Generator,
    img_size: int = 224,
    has_cad: bool = True,
    with_plaque: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Generate a single synthetic CCTA-like image.

    Args:
        rng:       numpy random generator
        img_size:  image size (square)
        has_cad:   whether this is a CAD (True) or non-CAD (False) sample
        with_plaque: include plaque features if CAD

    Returns:
        (image [H, W] uint8, vessel_mask [H, W] float, label int (1=CAD))
    """
    img = np.zeros((img_size, img_size), dtype=np.float64)
    mask = np.zeros((img_size, img_size), dtype=np.float64)

    # Background tissue intensity with smooth variations
    y, x = np.mgrid[0:img_size, 0:img_size]
    tissue = 0.15 + 0.05 * np.sin(x / 40.0) * np.cos(y / 50.0)
    img += tissue

    # Draw main coronary artery branches from the top-left (aorta origin)
    start = (img_size // 8, img_size // 2)
    num_branches = rng.integers(2, 4)
    for b in range(num_branches):
        angle = rng.uniform(-60, 60)
        length = rng.integers(int(img_size * 0.5), int(img_size * 0.85))
        width = rng.uniform(3.0, 7.0)
        img, mask = _draw_vessel(
            img, mask, start, int(length), float(angle), float(width),
            intensity=rng.uniform(0.55, 0.75),
        )

    # Additional small branch vessels
    for _ in range(rng.integers(1, 3)):
        sx = rng.integers(20, img_size - 20)
        sy = rng.integers(20, img_size - 20)
        img, mask = _draw_vessel(
            img, mask, (int(sx), int(sy)), int(rng.integers(40, 100)),
            float(rng.uniform(0, 360)), float(rng.uniform(1.5, 3.5)),
            intensity=rng.uniform(0.5, 0.65),
        )

    # CAD-specific features: atherosclerotic plaques
    label = 1 if has_cad else 0
    if has_cad and with_plaque:
        num_plaques = rng.integers(1, 4)
        for _ in range(num_plaques):
            # Find a vessel pixel for realistic plaque placement
            vessel_idx = np.argwhere(mask > 0.5)
            if len(vessel_idx) > 0:
                idx = vessel_idx[rng.integers(0, len(vessel_idx))]
                cy, cx = int(idx[0]), int(idx[1])
                radius = int(rng.uniform(3, 10))
                _draw_plaque(img, (cx, cy), radius, intensity=rng.uniform(0.85, 1.0))

    # Imaging noise (Gaussian + salt-pepper)
    noise_sigma = rng.uniform(0.02, 0.08)
    img = img + rng.normal(0, noise_sigma, img.shape)
    if rng.random() < 0.3:
        # Salt-and-pepper
        salt = rng.random(img.shape) < 0.005
        pepper = rng.random(img.shape) < 0.005
        img[salt] = 1.0
        img[pepper] = 0.0

    # Motion/streak artifacts (linear streaks)
    if rng.random() < 0.4:
        num_streaks = rng.integers(1, 4)
        for _ in range(num_streaks):
            sx = rng.integers(0, img_size)
            sy = rng.integers(0, img_size)
            angle = rng.uniform(0, 360)
            dx = np.cos(np.radians(angle)) * 40
            dy = np.sin(np.radians(angle)) * 40
            for t in range(-5, 6):
                px = int(np.clip(sx + dx * t / 5, 0, img_size - 1))
                py = int(np.clip(sy + dy * t / 5, 0, img_size - 1))
                img[py, px] = min(1.0, img[py, px] + 0.3)

    # Contrast variation
    contrast = rng.uniform(0.8, 1.3)
    img = np.clip((img - 0.5) * contrast + 0.5, 0, 1)

    # Convert to uint8
    img_uint8 = (img * 255).astype(np.uint8)

    return img_uint8, mask, label


class SyntheticCCTAGenerator:
    """
    Generates a synthetic CCTA dataset with patient-level partitions matching
    the paper's Table 2 distribution.
    """

    def __init__(self, seed: int = config.SEED):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.samples: list[dict] = []  # each: {image, mask, label, patient_id}

    def generate(self, num_patients: int = None, images_per_patient: int = 6):
        """
        Generate the full synthetic dataset.

        Args:
            num_patients:     number of patients (paper: 200)
            images_per_patient: slices per patient (paper yields ~12 images/patient)
        """
        num_patients = num_patients or config.NUM_PATIENTS
        self.samples = []

        for pid in range(num_patients):
            # Balanced CAD / non-CAD patients
            has_cad = pid % 2 == 0
            for _ in range(images_per_patient):
                img, mask, label = generate_sample(self.rng, config.IMG_SIZE, has_cad=has_cad)
                self.samples.append({
                    "image": img,
                    "mask": mask,
                    "label": label,
                    "patient_id": pid,
                })

        return self.samples

    def save(self, output_dir: str = None):
        """Save generated images to disk organized by class and patient."""
        output_dir = output_dir or os.path.join(config.DATA_DIR, "synthetic")
        os.makedirs(output_dir, exist_ok=True)

        for i, sample in enumerate(self.samples):
            label = sample["label"]
            pid = sample["patient_id"]
            subdir = os.path.join(output_dir, f"label_{label}", f"patient_{pid:03d}")
            os.makedirs(subdir, exist_ok=True)
            Image.fromarray(sample["image"]).save(os.path.join(subdir, f"img_{i:05d}.png"))
            np.save(os.path.join(subdir, f"mask_{i:05d}.npy"), sample["mask"])

        summary = {
            "num_samples": len(self.samples),
            "num_patients": len({s["patient_id"] for s in self.samples}),
            "cad_samples": sum(1 for s in self.samples if s["label"] == 1),
            "non_cad_samples": sum(1 for s in self.samples if s["label"] == 0),
        }
        return summary


if __name__ == "__main__":
    gen = SyntheticCCTAGenerator()
    samples = gen.generate(num_patients=8, images_per_patient=6)
    print(f"Generated {len(samples)} synthetic samples")
    summary = gen.save()
    print("Summary:", summary)

    # Quick visualization of one sample
    import matplotlib.pyplot as plt
    sample = samples[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(sample["image"], cmap="gray")
    axes[0].set_title(f"Image (label={sample['label']})")
    axes[1].imshow(sample["mask"], cmap="hot")
    axes[1].set_title("Vessel mask")
    img = sample["image"].astype(np.float64) / 255.0
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.hbv import apply_hbv
    enhanced = apply_hbv(img)
    axes[2].imshow(enhanced, cmap="gray")
    axes[2].set_title("HBV enhanced")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "synthetic_preview.png"))
    print(f"Preview saved to {os.path.join(config.OUTPUT_DIR, 'synthetic_preview.png')}")