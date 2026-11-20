"""
Training pipeline for the DSCNN-SAM model.

Implements the full training procedure:
  1. HBV preprocessing (applied in the dataset loader)
  2. DDPM vessel segmentation training
  3. DSCNN classifier training with SAM optimization
  4. Evaluation on validation set
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.dataset import build_dataloaders
from models.ddpm import build_ddpm
from models.dscnn import build_dscnn
from models.sam import build_sam_optimizer
from utils.metrics import evaluate_model, format_metric_table


def _save_checkpoint_atomic(path: str, state: dict) -> None:
    """
    Save a training checkpoint without risking a corrupt/truncated file if the
    process is killed mid-write (e.g. a machine reboot) -- write to a temp
    file first, then atomically rename it into place.
    """
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# DDPM training
# ---------------------------------------------------------------------------
def train_ddpm(
    train_loader,
    val_loader,
    device: str = config.DEVICE,
    epochs: int = None,
    lr: float = None,
    save_path: str = None,
    resume: bool = False,
) -> dict:
    """
    Train the DDPM vessel segmentation module.

    Loss: Dice + BCE (paper Table 1) combined with noise-prediction MSE.
    Optimizer: Adam with lr=0.0001 (paper Table 1).

    If resume=True and a resume checkpoint exists in the current version
    directory, training continues from the last fully-completed epoch
    (model + optimizer state + history) instead of starting over.
    """
    epochs = epochs or config.DDPM_EPOCHS
    lr = lr or config.DDPM_LR
    version_dir = config.get_current_version_dir()
    save_path = save_path or os.path.join(version_dir, "ddpm_model.pt")
    resume_path = os.path.join(version_dir, "ddpm_resume.pt")

    model = build_ddpm().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    start_epoch = 1

    if resume and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = ckpt["history"]
        best_val_loss = ckpt["best_val_loss"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed DDPM training from epoch {start_epoch}/{epochs} (best_val_loss={best_val_loss:.4f})")

    if start_epoch > epochs:
        print("DDPM training already complete per resume checkpoint.")
        return history

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, masks, _ in tqdm(train_loader, desc=f"DDPM Epoch {epoch}/{epochs}", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            # Resize to DDPM internal resolution
            images = F.interpolate(images, size=(config.DDPM_IMG_SIZE, config.DDPM_IMG_SIZE), mode="bilinear", align_corners=False)
            masks = F.interpolate(masks, size=(config.DDPM_IMG_SIZE, config.DDPM_IMG_SIZE), mode="nearest")

            # Random diffusion timesteps
            t = torch.randint(0, config.DDPM_TIMESTEPS, (images.size(0),), device=device)

            with torch.autocast(device_type="cuda", dtype=config.AMP_DTYPE, enabled=config.USE_AMP):
                loss, _, _ = model.p_losses(images, t, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train = epoch_loss / max(num_batches, 1)
        history["train_loss"].append(avg_train)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                images = F.interpolate(images, size=(config.DDPM_IMG_SIZE, config.DDPM_IMG_SIZE), mode="bilinear", align_corners=False)
                masks = F.interpolate(masks, size=(config.DDPM_IMG_SIZE, config.DDPM_IMG_SIZE), mode="nearest")
                t = torch.randint(0, config.DDPM_TIMESTEPS, (images.size(0),), device=device)
                with torch.autocast(device_type="cuda", dtype=config.AMP_DTYPE, enabled=config.USE_AMP):
                    loss, _, _ = model.p_losses(images, t, masks)
                val_loss += loss.item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        history["val_loss"].append(avg_val)

        # ETA estimation
        epoch_time = time.time() - epoch_start
        remaining_epochs = epochs - epoch
        eta_seconds = remaining_epochs * epoch_time
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        if epoch % 10 == 0 or epoch == 1:
            print(f"DDPM Epoch {epoch}/{epochs}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}, ETA {eta_str}")

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)

        # Resume checkpoint (every epoch, regardless of improvement)
        _save_checkpoint_atomic(resume_path, {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_val_loss": best_val_loss,
        })

    print(f"DDPM training complete. Best val loss: {best_val_loss:.4f}")
    return history


# ---------------------------------------------------------------------------
# DSCNN training with SAM
# ---------------------------------------------------------------------------
def train_dscnn_sam(
    train_loader,
    val_loader,
    device: str = config.DEVICE,
    epochs: int = None,
    lr: float = None,
    rho: float = None,
    save_path: str = None,
    resume: bool = False,
) -> tuple[nn.Module, dict]:
    """
    Train the DSCNN classifier using SAM optimization.

    SAM procedure per batch:
      1. Forward + backward → gradients at current params
      2. sam.first_step(zero_grad=True) → perturb params (ascent)
      3. Forward + backward at perturbed params
      4. sam.second_step(zero_grad=True) → restore params + base optimizer update

    If resume=True and a resume checkpoint exists in the current version
    directory, training continues from the last fully-completed epoch
    (model + base-optimizer state + history + early-stop counter) instead of
    starting over.
    """
    epochs = epochs or config.SAM_EPOCHS
    lr = lr or config.SAM_LR
    rho = rho or config.SAM_RHO
    version_dir = config.get_current_version_dir()
    save_path = save_path or os.path.join(version_dir, "dscnn_sam_model.pt")
    resume_path = os.path.join(version_dir, "sam_resume.pt")

    model = build_dscnn().to(device)
    if config.USE_COMPILE:
        model = torch.compile(model)
    sam, base_optimizer = build_sam_optimizer(
        model, rho=rho, lr=lr, weight_decay=config.SAM_WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": [], "val_loss": []}
    best_val_acc = 0.0
    epochs_without_improvement = 0
    patience = config.SAM_EARLY_STOP_PATIENCE
    val_every = max(1, config.VAL_EVERY_N_EPOCHS)
    # epochs_without_improvement now counts validation *checks*, not epochs,
    # since we only get new information on validation epochs -- scale the
    # patience accordingly so it still means ~`patience` epochs, not checks.
    patience_checks = max(1, patience // val_every)
    start_epoch = 1

    if resume and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        sam.base_optimizer.load_state_dict(ckpt["base_optimizer_state"])
        history = ckpt["history"]
        best_val_acc = ckpt["best_val_acc"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed DSCNN-SAM training from epoch {start_epoch}/{epochs} (best_val_acc={best_val_acc:.4f})")

    if start_epoch > epochs:
        print("DSCNN-SAM training already complete per resume checkpoint.")
        model.load_state_dict(torch.load(save_path, map_location=device))
        return model, history

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, _, labels in tqdm(train_loader, desc=f"DSCNN-SAM Epoch {epoch}/{epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            # ---- First step: compute gradients at current params ----
            with torch.autocast(device_type="cuda", dtype=config.AMP_DTYPE, enabled=config.USE_AMP):
                logits = model(images)
                loss = criterion(logits, labels)
            loss.backward()
            sam.first_step(zero_grad=True)

            # ---- Second step: compute gradients at perturbed params ----
            with torch.autocast(device_type="cuda", dtype=config.AMP_DTYPE, enabled=config.USE_AMP):
                logits_perturbed = model(images)
                loss_perturbed = criterion(logits_perturbed, labels)
            loss_perturbed.backward()
            sam.second_step(zero_grad=True)

            epoch_loss += loss.item()
            num_batches += 1

        avg_train = epoch_loss / max(num_batches, 1)
        history["train_loss"].append(avg_train)

        # Validation -- only every `val_every` epochs (plus always on the
        # first and last) to trim wall time; skipped epochs carry forward
        # the last known val_acc so history/plots stay a well-formed series.
        do_validate = (epoch % val_every == 0) or (epoch == 1) or (epoch == epochs)
        if do_validate:
            val_metrics = evaluate_model(model, val_loader, device)
            val_acc = val_metrics["accuracy"]
        else:
            val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
        history["val_acc"].append(val_acc)
        history["val_loss"].append(1.0 - val_acc)

        # ETA estimation
        epoch_time = time.time() - epoch_start
        remaining_epochs = epochs - epoch
        eta_seconds = remaining_epochs * epoch_time
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        if epoch % 10 == 0 or epoch == 1:
            print(f"DSCNN-SAM Epoch {epoch}/{epochs}: train_loss={avg_train:.4f}, val_acc={val_acc:.4f}, ETA {eta_str}")

        # Save best model / early-stop bookkeeping (only on validation epochs --
        # there's no new information to act on otherwise)
        if do_validate:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), save_path)
            else:
                epochs_without_improvement += 1

        # Resume checkpoint (every epoch, regardless of improvement)
        _save_checkpoint_atomic(resume_path, {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "base_optimizer_state": sam.base_optimizer.state_dict(),
            "history": history,
            "best_val_acc": best_val_acc,
            "epochs_without_improvement": epochs_without_improvement,
        })

        if do_validate and epochs_without_improvement >= patience_checks:
            print(
                f"Early stopping at epoch {epoch}/{epochs} "
                f"(no val_acc improvement in {patience} epochs). "
                f"Best val accuracy: {best_val_acc:.4f}"
            )
            break

    print(f"DSCNN-SAM training complete. Best val accuracy: {best_val_acc:.4f}")
    return model, history


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_full_training(
    num_patients: int | None = None,
    images_per_patient: int = 6,
    use_hbv: bool = True,
    do_train_ddpm: bool = True,
    do_train_classifier: bool = True,
    ddpm_epochs: int | None = None,
    classifier_epochs: int | None = None,
    device: str = config.DEVICE,
    resume: bool = False,
) -> dict:
    """
    Run the complete DSCNN-SAM training pipeline.

    Returns:
        dict with trained models, dataloaders, and history.
    """
    print("=" * 60)
    print("DSCNN-SAM CAD Detection - Training Pipeline")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset mode: {'Synthetic' if config.USE_SYNTHETIC_DATA else 'Real CCTA'}")
    print()

    # 1. Build dataloaders with patient-level split
    print("[1/3] Building dataloaders with patient-level split...")
    train_loader, val_loader, test_loader, stats = build_dataloaders(
        batch_size=config.SAM_BATCH_SIZE,
        num_patients=num_patients,
        images_per_patient=images_per_patient,
        use_hbv=use_hbv,
    )
    print(f"  Train: {stats['num_train_patients']} patients, {stats['num_train_samples']} images")
    print(f"  Val:   {stats['num_val_patients']} patients, {stats['num_val_samples']} images")
    print(f"  Test:  {stats['num_test_patients']} patients, {stats['num_test_samples']} images")
    print()

    results = {
        "stats": stats,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }

    # 2. Train DDPM segmentation
    if do_train_ddpm:
        print("[2/3] Training DDPM vessel segmentation...")
        ddpm_history = train_ddpm(
            train_loader, val_loader, device=device, epochs=ddpm_epochs, resume=resume
        )
        results["ddpm_history"] = ddpm_history
    else:
        print("[2/3] Skipping DDPM training (using pre-trained if available)...")

    # 3. Train DSCNN with SAM
    if do_train_classifier:
        print("[3/3] Training DSCNN classifier with SAM optimization...")
        model, history = train_dscnn_sam(
            train_loader, val_loader, device=device, epochs=classifier_epochs, resume=resume
        )
        results["model"] = model
        results["history"] = history
    else:
        print("[3/3] Skipping classifier training...")

    print()
    print("Training pipeline complete.")
    return results


if __name__ == "__main__":
    results = run_full_training(
        num_patients=40,       # smaller for quick smoke test
        images_per_patient=6,
        do_train_ddpm=True,
        do_train_classifier=True,
    )
    print("\nFinal validation metrics:")
    model = results["model"]
    val_metrics = evaluate_model(model, results["val_loader"], config.DEVICE)
    print(format_metric_table(val_metrics))