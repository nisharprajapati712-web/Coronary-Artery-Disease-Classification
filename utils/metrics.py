"""
Evaluation metrics for CAD detection.

Implements the paper's Eqs. (11)-(14):
  Accuracy  = (TP + TN) / (TP + TN + FP + FN)
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)   (Sensitivity)
  F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)
  Specificity = TN / (TN + FP)
"""

from __future__ import annotations

import numpy as np
import torch

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute classification metrics from integer labels (0=non-CAD, 1=CAD).

    Args:
        y_true: ground-truth labels (shape [N])
        y_pred: predicted labels (shape [N])

    Returns:
        dict with accuracy, precision, recall, f1, specificity, TP, TN, FP, FN
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)          # sensitivity
    specificity = tn / (tn + fp + 1e-12)
    f1 = 2.0 * (precision * recall) / (precision + recall + 1e-12)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_model(model, data_loader, device: str = "cpu") -> dict:
    """
    Evaluate a PyTorch classification model on a dataloader.

    Returns:
        dict with metrics + y_true, y_pred, y_prob arrays.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, _, labels in data_loader:
            images = images.to(device)
            with torch.autocast(device_type="cuda", dtype=config.AMP_DTYPE, enabled=config.USE_AMP and device == "cuda"):
                logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_metrics(y_true, y_pred)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_prob"] = y_prob

    return metrics


def format_metric_table(results: dict) -> str:
    """Format metrics as a readable table string."""
    header = f"{'Metric':<14}{'Value':<10}\n{'-' * 24}"
    lines = [header]
    for key in ["accuracy", "precision", "recall", "f1", "specificity"]:
        lines.append(f"{key:<14}{results[key]:.4f}")
    lines.append(f"{'TP':<14}{results.get('tp', 0)}")
    lines.append(f"{'TN':<14}{results.get('tn', 0)}")
    lines.append(f"{'FP':<14}{results.get('fp', 0)}")
    lines.append(f"{'FN':<14}{results.get('fn', 0)}")
    return "\n".join(lines)


if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    print(format_metric_table(compute_metrics(y_true, y_pred)))