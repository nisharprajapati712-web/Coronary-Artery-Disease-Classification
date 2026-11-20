"""
Smoke test to verify the complete DSCNN-SAM pipeline runs end-to-end.
Uses very small dataset and reduced timesteps for quick verification.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def run():
    print("=" * 60)
    print("DSCNN-SAM Smoke Test")
    print("=" * 60)

    from data.dataset import build_dataloaders
    from models.hbv import apply_hbv
    from models.ddpm import build_ddpm
    from models.dscnn import build_dscnn
    from models.sam import build_sam_optimizer
    from utils.metrics import evaluate_model, format_metric_table

    device = "cpu"

    # 1. Data loaders
    print("\n[1] Building dataloaders...")
    train_loader, val_loader, test_loader, stats = build_dataloaders(
        batch_size=4, num_patients=8, images_per_patient=4, use_hbv=True,
    )
    print(f"  Train: {stats['num_train_samples']} images | Val: {stats['num_val_samples']} | Test: {stats['num_test_samples']}")

    # 2. HBV filter
    print("\n[2] Testing HBV filter...")
    from data.synthetic import generate_sample
    import numpy as np
    rng = np.random.default_rng(0)
    img, mask, label = generate_sample(rng, config.IMG_SIZE, has_cad=True)
    img_f = img.astype(np.float32) / 255.0
    enhanced = apply_hbv(img_f)
    print(f"  Input: {img_f.shape}, Enhanced: {enhanced.shape}, range [{enhanced.min():.3f}, {enhanced.max():.3f}]")

    # 3. DDPM forward + loss
    print("\n[3] Testing DDPM...")
    ddpm = build_ddpm()
    x = torch.rand(2, 1, config.DDPM_IMG_SIZE, config.DDPM_IMG_SIZE)
    t = torch.randint(0, config.DDPM_TIMESTEPS, (2,)).long()
    target = (torch.rand(2, 1, config.DDPM_IMG_SIZE, config.DDPM_IMG_SIZE) > 0.7).float()
    loss, l_mse, l_seg = ddpm.p_losses(x, t, target)
    print(f"  DDPM loss OK: {loss.item():.4f} (mse={l_mse.item():.4f}, seg={l_seg.item():.4f})")

    # 4. DSCNN + SAM training step
    print("\n[4] Testing DSCNN + SAM training step...")
    # Use reduced timesteps for the smoke test
    orig_timesteps = config.DSCNN_TIMESTEPS
    config.DSCNN_TIMESTEPS = 2

    model = build_dscnn()
    sam, _ = build_sam_optimizer(model, rho=0.5, lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    images, _, labels = next(iter(train_loader))
    t0 = time.time()
    # SAM two-step
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    sam.first_step(zero_grad=True)
    logits_p = model(images)
    loss_p = criterion(logits_p, labels)
    loss_p.backward()
    sam.second_step(zero_grad=True)
    elapsed = time.time() - t0
    print(f"  SAM training step OK in {elapsed:.2f}s")

    # Restore timesteps
    config.DSCNN_TIMESTEPS = orig_timesteps

    # 5. Evaluation
    print("\n[5] Testing evaluation...")
    metrics = evaluate_model(model, test_loader, device)
    print(format_metric_table(metrics))

    # 6. Plots
    print("\n[6] Testing plots...")
    from evaluate import (
        plot_confusion_matrix, plot_roc_curve,
        plot_classification_report, compute_computational_metrics,
    )
    out_dir = config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    y_prob = metrics["y_prob"]
    plot_confusion_matrix(y_true, y_pred, os.path.join(out_dir, "smoke_cm.png"))
    plot_roc_curve(y_true, y_prob, os.path.join(out_dir, "smoke_roc.png"))
    plot_classification_report(metrics, os.path.join(out_dir, "smoke_report.png"))
    comp = compute_computational_metrics(model, device="cpu")
    print(f"  Params: {comp['parameters_m']:.2f}M | FLOPs: {comp['flops_g']:.2f}G | "
          f"Latency: {comp['latency_s']:.3f}s | Throughput: {comp['throughput']:.2f}/s")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED - All components functional")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run()