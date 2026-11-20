"""
CCTA dataset loader with patient-level partitioning.

The paper (Table 2) uses a 70/15/15 patient-level split:
  - Training:   140 patients (70 CAD / 70 non-CAD) → 828 CAD + 827 non-CAD images
  - Validation:  30 patients (15 CAD / 15 non-CAD) → 177 CAD + 177 non-CAD images
  - Testing:     30 patients (15 CAD / 15 non-CAD) → 177 CAD + 178 non-CAD images

Patient-level separation prevents data leakage between train and test.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.synthetic import SyntheticCCTAGenerator


class CCTADataset(Dataset):
    """
    PyTorch dataset for CCTA images.

    Supports:
      - Synthetic mode (config.USE_SYNTHETIC_DATA=True): in-memory samples
      - Real mode: reads PNG images + .npy masks from config.DATA_DIR
    """

    def __init__(
        self,
        samples: list[dict] | None = None,
        root_dir: str | None = None,
        transform: bool = True,
        use_hbv: bool = True,
    ):
        self.samples = samples
        self.root_dir = root_dir or config.DATA_DIR
        self.transform = transform
        self.use_hbv = use_hbv

        if self.samples is None:
            self.samples = self._load_from_disk()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_from_disk(self) -> list[dict]:
        """Load real CCTA images from disk (expected: .png + .npy masks)."""
        samples = []
        for label_name in ["label_0", "label_1"]:
            label_dir = os.path.join(self.root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            label = int(label_name.split("_")[1])
            for patient_dir in sorted(os.listdir(label_dir)):
                patient_path = os.path.join(label_dir, patient_dir)
                if not os.path.isdir(patient_path):
                    continue
                patient_id = int(patient_dir.replace("patient_", ""))
                for fname in sorted(os.listdir(patient_path)):
                    if fname.endswith(".png"):
                        img_path = os.path.join(patient_path, fname)
                        mask_path = img_path.replace(".png", ".npy")
                        mask = (
                            np.load(mask_path)
                            if os.path.exists(mask_path)
                            else np.zeros((config.IMG_SIZE, config.IMG_SIZE))
                        )
                        samples.append({
                            "image_path": img_path,
                            "image": None,
                            "mask": mask.astype(np.float32),
                            "label": label,
                            "patient_id": patient_id,
                        })
        return samples

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, image_path: str) -> str | None:
        """
        Path where the fully preprocessed (resized + HBV-filtered) version of
        this image is cached. HBV/resize are deterministic functions of the
        image alone, so the result is identical across every epoch and every
        training phase -- computing it once and reusing it avoids redoing the
        expensive Hessian filtering (and PNG decode/resize) on every one of
        the ~200 epochs in the full pipeline.
        """
        if config.CACHE_DIR is None:
            return None
        rel = os.path.relpath(image_path, config.DATA_DIR)
        key = f"sz{config.IMG_SIZE}_hbv{int(self.use_hbv)}"
        cache_path = os.path.join(config.CACHE_DIR, key, rel)
        return os.path.splitext(cache_path)[0] + ".npy"

    @staticmethod
    def _save_cache(cache_path: str, img: np.ndarray) -> None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = f"{cache_path}.tmp{os.getpid()}"
        try:
            with open(tmp_path, "wb") as f:
                np.save(f, img)
            os.replace(tmp_path, cache_path)
        except OSError:
            # Another worker may be writing the same file concurrently; the
            # cache is a pure optimization, so a failed write is harmless.
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        cache_path = None
        if sample["image"] is None:
            cache_path = self._cache_path(sample["image_path"])

        if cache_path is not None and os.path.exists(cache_path):
            img = np.load(cache_path)
        else:
            if sample["image"] is not None:
                img = sample["image"]
            else:
                img = np.array(Image.open(sample["image_path"]).convert("L"))
                img = img.astype(np.float32) / 255.0

            # Downsize to the configured resolution before HBV so the expensive
            # Hessian filtering also runs on the smaller image, not just the model.
            if img.shape[0] != config.IMG_SIZE or img.shape[1] != config.IMG_SIZE:
                img_pil = Image.fromarray(img.astype(np.float32), mode="F")
                img_pil = img_pil.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)
                img = np.array(img_pil, dtype=np.float32)

            if self.use_hbv:
                from models.hbv import apply_hbv
                img = apply_hbv(img)

            if cache_path is not None:
                self._save_cache(cache_path, img)

        mask = sample["mask"]

        # Convert to tensor [1, H, W]
        img_t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).unsqueeze(0)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).unsqueeze(0)
        label = torch.tensor(sample["label"], dtype=torch.long)

        return img_t, mask_t, label


def prepare_synthetic_samples(num_patients: int | None = None, images_per_patient: int = 6):
    """
    Generate synthetic samples on demand (keeps in memory for speed).
    Matches paper's balanced 50/50 CAD/non-CAD patient distribution.

    Returns:
        list of {image, mask, label, patient_id}
    """
    gen = SyntheticCCTAGenerator(seed=config.SEED)
    return gen.generate(
        num_patients=num_patients or config.NUM_PATIENTS,
        images_per_patient=images_per_patient,
    )


def patient_level_split(
    samples: list[dict],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = config.SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Perform patient-level split (no data leakage across patients).

    Returns (train, val, test) sample lists.
    """
    rng = np.random.default_rng(seed)

    # Group samples by patient
    patients = {}
    for s in samples:
        pid = s["patient_id"]
        patients.setdefault(pid, []).append(s)

    patient_ids = list(patients.keys())
    rng.shuffle(patient_ids)

    n_train = int(len(patient_ids) * ratios[0])
    n_val = int(len(patient_ids) * ratios[1])

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]

    def collect(ids):
        out = []
        for pid in ids:
            out.extend(patients[pid])
        return out

    return collect(train_ids), collect(val_ids), collect(test_ids)


def build_dataloaders(
    batch_size: int = None,
    val_batch_size: int = None,
    num_patients: int | None = None,
    images_per_patient: int = 6,
    use_hbv: bool = True,
    seed: int = config.SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Build train/val/test dataloaders with patient-level split.

    Val/test loaders use a larger batch size than train by default -- they're
    forward-only (no SAM double backward pass), so they don't hit the memory
    constraints that capped the training batch size.

    Returns:
        (train_loader, val_loader, test_loader, dataset_stats)
    """
    batch_size = batch_size or config.SAM_BATCH_SIZE
    val_batch_size = val_batch_size or config.VAL_BATCH_SIZE

    if config.USE_SYNTHETIC_DATA:
        samples = prepare_synthetic_samples(num_patients, images_per_patient)
    else:
        samples = None  # loaded from disk in CCTADataset

    if samples is not None:
        train_s, val_s, test_s = patient_level_split(samples, config.SPLIT_RATIOS, seed)
        train_ds = CCTADataset(samples=train_s, use_hbv=use_hbv)
        val_ds = CCTADataset(samples=val_s, use_hbv=use_hbv)
        test_ds = CCTADataset(samples=test_s, use_hbv=use_hbv)
    else:
        # Real data: load from disk and split by patient directory
        full_ds = CCTADataset(use_hbv=use_hbv)
        samples = full_ds.samples
        train_s, val_s, test_s = patient_level_split(samples, config.SPLIT_RATIOS, seed)
        train_ds = CCTADataset(samples=train_s, use_hbv=use_hbv)
        val_ds = CCTADataset(samples=val_s, use_hbv=use_hbv)
        test_ds = CCTADataset(samples=test_s, use_hbv=use_hbv)

    # persistent_workers avoids respawning the worker pool (and re-importing
    # torch/scipy in each one) on every single epoch -- with ~200 epochs in
    # the full pipeline that respawn cost adds up.
    persistent = config.NUM_WORKERS > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        persistent_workers=persistent, prefetch_factor=4 if persistent else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        persistent_workers=persistent, prefetch_factor=4 if persistent else None,
    )
    test_loader = DataLoader(
        test_ds, batch_size=val_batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        persistent_workers=persistent, prefetch_factor=4 if persistent else None,
    )

    stats = {
        "num_train_patients": len({s["patient_id"] for s in train_s}),
        "num_val_patients": len({s["patient_id"] for s in val_s}),
        "num_test_patients": len({s["patient_id"] for s in test_s}),
        "num_train_samples": len(train_s),
        "num_val_samples": len(val_s),
        "num_test_samples": len(test_s),
        "train_cad": sum(1 for s in train_s if s["label"] == 1),
        "train_non_cad": sum(1 for s in train_s if s["label"] == 0),
        "val_cad": sum(1 for s in val_s if s["label"] == 1),
        "val_non_cad": sum(1 for s in val_s if s["label"] == 0),
        "test_cad": sum(1 for s in test_s if s["label"] == 1),
        "test_non_cad": sum(1 for s in test_s if s["label"] == 0),
    }

    return train_loader, val_loader, test_loader, stats


if __name__ == "__main__":
    train_l, val_l, test_l, stats = build_dataloaders(batch_size=4, num_patients=20, images_per_patient=6)
    print("Dataset stats:", stats)
    x, mask, y = next(iter(train_l))
    print("Batch shapes:", x.shape, mask.shape, y.shape)