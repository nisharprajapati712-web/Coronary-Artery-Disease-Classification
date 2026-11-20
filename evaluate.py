"""
Evaluation utilities for the DSCNN-SAM model.

Implements:
  - Confusion matrix visualization (Fig. 4)
  - ROC curve with AUC (Fig. 5)
  - Classification report (Fig. 6)
  - Ablation study (Table 3): excluding HBV / DDPM / SAM
  - K-fold cross-validation (Table 6)
  - Computational analysis (Table 4): params, FLOPs, latency, throughput
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.dataset import build_dataloaders, patient_level_split, prepare_synthetic_samples
from models.dscnn import build_dscnn
from models.sam import build_sam_optimizer
from utils.metrics import compute_metrics, evaluate_model, format_metric_table


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """Plot and save the confusion matrix (paper Fig. 4)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix - DSCNN-SAM",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.close(fig)
    return cm


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
    """Plot and save the ROC curve (paper Fig. 5)."""
    # Use probability of CAD class (index 1)
    scores = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - DSCNN-SAM")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curve saved to {save_path}")
    plt.close(fig)
    return roc_auc


def plot_classification_report(metrics: dict, save_path: str = None):
    """Plot the classification report (paper Fig. 6)."""
    classes = config.CLASS_NAMES
    values = [metrics["precision"], metrics["recall"], metrics["f1"]]
    labels = ["Precision", "Recall", "F1-Score"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.35
    bars = ax.bar(x, values, width, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Score")
    ax.set_title("Classification Report - DSCNN-SAM")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Classification report saved to {save_path}")
    plt.close(fig)


def plot_training_history(history: dict, save_path: str = None):
    """Plot training loss and validation accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="Val Accuracy", color="green")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Validation Accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history saved to {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ablation study (Table 3)
# ---------------------------------------------------------------------------
def run_ablation_study(
    num_patients: int = 60,
    images_per_patient: int = 6,
    epochs: int = 30,
    device: str = config.DEVICE,
) -> dict:
    """
    Ablation study: evaluate contribution of each component.

    Configurations:
      1. Excluding HBV (use_hbv=False)
      2. Excluding DDPM (skip DDPM training; classifier on raw enhanced images)
      3. Excluding SAM (use plain Adam instead of SAM)
      4. Proposed DSCNN-SAM (full pipeline)

    Returns dict of results per configuration.
    """
    results = {}

    # ---- 1. Excluding HBV ----
    print("\n=== Ablation: Excluding HBV ===")
    train_l, val_l, test_l, _ = build_dataloaders(
        batch_size=config.SAM_BATCH_SIZE,
        num_patients=num_patients,
        images_per_patient=images_per_patient,
        use_hbv=False,
    )
    model = build_dscnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.SAM_LR, weight_decay=config.SAM_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    _train_plain(model, train_l, optimizer, criterion, epochs, device)
    results["Excluding HBV"] = evaluate_model(model, test_l, device)

    # ---- 2. Excluding DDPM ----
    print("\n=== Ablation: Excluding DDPM ===")
    train_l, val_l, test_l, _ = build_dataloaders(
        batch_size=config.SAM_BATCH_SIZE,
        num_patients=num_patients,
        images_per_patient=images_per_patient,
        use_hbv=True,
    )
    model = build_dscnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.SAM_LR, weight_decay=config.SAM_WEIGHT_DECAY)
    _train_plain(model, train_l, optimizer, criterion, epochs, device)
    results["Excluding DDPM"] = evaluate_model(model, test_l, device)

    # ---- 3. Excluding SAM ----
    print("\n=== Ablation: Excluding SAM ===")
    train_l, val_l, test_l, _ = build_dataloaders(
        batch_size=config.SAM_BATCH_SIZE,
        num_patients=num_patients,
        images_per_patient=images_per_patient,
        use_hbv=True,
    )
    model = build_dscnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.SAM_LR, weight_decay=config.SAM_WEIGHT_DECAY)
    _train_plain(model, train_l, optimizer, criterion, epochs, device)
    results["Excluding SAM"] = evaluate_model(model, test_l, device)

    # ---- 4. Proposed DSCNN-SAM ----
    print("\n=== Ablation: Proposed DSCNN-SAM ===")
    train_l, val_l, test_l, _ = build_dataloaders(
        batch_size=config.SAM_BATCH_SIZE,
        num_patients=num_patients,
        images_per_patient=images_per_patient,
        use_hbv=True,
    )
    model = build_dscnn().to(device)
    sam, _ = build_sam_optimizer(model, rho=config.SAM_RHO, lr=config.SAM_LR, weight_decay=config.SAM_WEIGHT_DECAY)
    _train_sam(model, train_l, sam, criterion, epochs, device)
    results["Proposed DSCNN-SAM"] = evaluate_model(model, test_l, device)

    return results


def _train_plain(model, train_loader, optimizer, criterion, epochs, device):
    """Simple training loop without SAM."""
    model.train()
    for epoch in range(1, epochs + 1):
        for images, _, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()


def _train_sam(model, train_loader, sam, criterion, epochs, device):
    """Training loop with SAM."""
    model.train()
    for epoch in range(1, epochs + 1):
        for images, _, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            sam.first_step(zero_grad=True)
            logits_p = model(images)
            loss_p = criterion(logits_p, labels)
            loss_p.backward()
            sam.second_step(zero_grad=True)


# ---------------------------------------------------------------------------
# K-fold validation (Table 6)
# ---------------------------------------------------------------------------
def run_kfold_validation(
    num_patients: int = 60,
    images_per_patient: int = 6,
    epochs: int = 20,
    k: int = config.K_FOLDS,
    device: str = config.DEVICE,
) -> dict:
    """
    Patient-level k-fold cross-validation.

    Returns dict with per-fold metrics and mean ± std.
    """
    samples = prepare_synthetic_samples(num_patients, images_per_patient)

    # Group by patient
    patients = {}
    for s in samples:
        patients.setdefault(s["patient_id"], []).append(s)
    patient_ids = np.array(sorted(patients.keys()))
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(patient_ids)

    folds = np.array_split(patient_ids, k)
    fold_results = []

    for fold_idx in range(k):
        print(f"\n=== Fold {fold_idx + 1}/{k} ===")
        test_ids = set(folds[fold_idx].tolist())
        train_ids = [pid for pid in patient_ids.tolist() if pid not in test_ids]

        train_s = []
        test_s = []
        for pid in train_ids:
            train_s.extend(patients[pid])
        for pid in test_ids:
            test_s.extend(patients[pid])

        from data.dataset import CCTADataset
        from torch.utils.data import DataLoader

        train_ds = CCTADataset(samples=train_s, use_hbv=True)
        test_ds = CCTADataset(samples=test_s, use_hbv=True)
        train_l = DataLoader(train_ds, batch_size=config.SAM_BATCH_SIZE, shuffle=True)
        test_l = DataLoader(test_ds, batch_size=config.SAM_BATCH_SIZE, shuffle=False)

        model = build_dscnn().to(device)
        sam, _ = build_sam_optimizer(model, rho=config.SAM_RHO, lr=config.SAM_LR, weight_decay=config.SAM_WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        _train_sam(model, train_l, sam, criterion, epochs, device)

        metrics = evaluate_model(model, test_l, device)
        fold_results.append(metrics)
        print(f"  Fold {fold_idx + 1} accuracy: {metrics['accuracy']:.4f}")

    # Aggregate
    summary = {}
    for key in ["accuracy", "precision", "recall", "f1", "specificity"]:
        values = [r[key] for r in fold_results]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "per_fold": values,
        }

    return {"per_fold": fold_results, "summary": summary}


# ---------------------------------------------------------------------------
# Computational analysis (Table 4)
# ---------------------------------------------------------------------------
def compute_computational_metrics(model, input_size: int = 224, device: str = "cpu"):
    """
    Estimate parameters, FLOPs, latency, throughput, and memory.

    Returns dict with:
      - parameters (M)
      - flops (G)
      - memory (GB)
      - latency (s/image)
      - throughput (images/s)
    """
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6

    # Estimate FLOPs for the spiking CNN (approximate)
    # Conv layers: 2 * C_out * C_in * K^2 * H_out * W_out
    flops = 0.0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            out_h = input_size // 2  # after pooling
            out_w = input_size // 2
            flops += (
                2.0
                * module.out_channels
                * module.in_channels
                * module.kernel_size[0] ** 2
                * out_h
                * out_w
            )
            input_size //= 2
    flops_g = flops / 1e9

    # Memory estimate (params * 4 bytes)
    memory_gb = total_params * 4 / (1024**3)

    # Latency / throughput
    x = torch.randn(1, 1, 224, 224).to(device)
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = model(x)
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.time()
        num_iters = 10
        for _ in range(num_iters):
            _ = model(x)
        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.time() - start

    latency = elapsed / num_iters
    throughput = 1.0 / latency

    return {
        "parameters_m": params_m,
        "flops_g": flops_g,
        "memory_gb": memory_gb,
        "latency_s": latency,
        "throughput": throughput,
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
def run_full_evaluation(
    model,
    test_loader,
    history: dict | None = None,
    device: str = config.DEVICE,
    output_dir: str = None,
) -> dict:
    """
    Run the complete evaluation suite:
      - Test metrics
      - Confusion matrix plot
      - ROC curve plot
      - Classification report plot
      - Training history plot
      - Computational analysis

    Returns dict with all results.
    """
    output_dir = output_dir or config.get_current_version_dir()
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("DSCNN-SAM Evaluation")
    print("=" * 60)

    # Test metrics
    test_metrics = evaluate_model(model, test_loader, device)
    print("\nTest Set Metrics:")
    print(format_metric_table(test_metrics))

    # Confusion matrix (Fig. 4)
    cm = plot_confusion_matrix(
        test_metrics["y_true"], test_metrics["y_pred"],
        save_path=os.path.join(output_dir, "confusion_matrix.png"),
    )

    # ROC curve (Fig. 5)
    roc_auc = plot_roc_curve(
        test_metrics["y_true"], test_metrics["y_prob"],
        save_path=os.path.join(output_dir, "roc_curve.png"),
    )
    print(f"\nROC AUC: {roc_auc:.4f}")

    # Classification report (Fig. 6)
    plot_classification_report(
        test_metrics,
        save_path=os.path.join(output_dir, "classification_report.png"),
    )

    # Training history
    if history:
        plot_training_history(history, save_path=os.path.join(output_dir, "training_history.png"))

    # Computational analysis (Table 4)
    print("\nComputational Analysis:")
    comp = compute_computational_metrics(model, device=device)
    print(f"  Parameters: {comp['parameters_m']:.2f} M")
    print(f"  FLOPs:      {comp['flops_g']:.2f} G")
    print(f"  Memory:     {comp['memory_gb']:.2f} GB")
    print(f"  Latency:    {comp['latency_s']:.4f} s/image")
    print(f"  Throughput: {comp['throughput']:.2f} images/s")

    return {
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "computational": comp,
    }


if __name__ == "__main__":
    # Quick smoke test
    from data.dataset import build_dataloaders
    train_l, val_l, test_l, stats = build_dataloaders(
        batch_size=4, num_patients=20, images_per_patient=6
    )
    model = build_dscnn()
    results = run_full_evaluation(model, test_l, device="cpu")
    print("\nDone.")