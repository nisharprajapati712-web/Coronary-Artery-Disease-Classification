"""
End-to-end execution of the DSCNN-SAM model for CAD detection.

Usage:
    python main.py                    # Full pipeline (train + evaluate)
    python main.py --quick            # Quick smoke test with fewer patients/epochs
    python main.py --ablation         # Run ablation study (Table 3)
    python main.py --kfold            # Run k-fold validation (Table 6)
    python main.py --evaluate-only    # Evaluate a saved model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.dataset import build_dataloaders
from models.dscnn import build_dscnn
from train import run_full_training
from evaluate import (
    run_full_evaluation,
    run_ablation_study,
    run_kfold_validation,
)
from utils.metrics import format_metric_table


def parse_args():
    parser = argparse.ArgumentParser(description="DSCNN-SAM CAD Detection")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--kfold", action="store_true", help="Run k-fold validation")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate saved model")
    parser.add_argument("--resume", action="store_true", help="Resume the most recent run from its last checkpoint")
    parser.add_argument("--num-patients", type=int, default=None, help="Number of patients")
    parser.add_argument("--images-per-patient", type=int, default=6, help="Images per patient")
    parser.add_argument("--epochs", type=int, default=None, help="Classifier epochs")
    parser.add_argument("--ddpm-epochs", type=int, default=None, help="DDPM epochs")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Device (cpu/cuda)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resuming continues into the same version directory as the last run;
    # a fresh run always gets a new one.
    if args.resume:
        version_dir = config.get_current_version_dir()
    else:
        version_dir = config.get_next_version_dir()
    print("=" * 70)
    print("DSCNN-SAM: Coronary Artery Disease Detection")
    print("Deep Spiking Convolutional Neural Network with")
    print("Sharpness-Aware Minimization")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Dataset: {'Synthetic CCTA' if config.USE_SYNTHETIC_DATA else 'Real CCTA'}")
    if args.resume:
        print(f"Resuming version: {os.path.basename(version_dir)} -> {version_dir}")
    else:
        print(f"Output version: {os.path.basename(version_dir)} -> {version_dir}")
    print()

    # Record run metadata (skip on resume so the original start timestamp/config survives)
    if not args.resume:
        run_meta = {
            "version": os.path.basename(version_dir),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": args.device,
            "dataset_mode": "Synthetic" if config.USE_SYNTHETIC_DATA else "Real CCTA",
            "num_patients": args.num_patients or config.NUM_PATIENTS,
            "images_per_patient": args.images_per_patient or 6,
            "epochs": args.epochs or config.SAM_EPOCHS,
            "ddpm_epochs": args.ddpm_epochs or config.DDPM_EPOCHS,
            "mode": "quick" if args.quick else "full",
        }
        with open(os.path.join(version_dir, "run_metadata.json"), "w") as f:
            json.dump(run_meta, f, indent=2)

    # ------------------------------------------------------------------
    # Quick mode: reduced dataset and epochs
    # ------------------------------------------------------------------
    if args.quick:
        print("Running in QUICK mode (reduced dataset/epochs for smoke test)")
        num_patients = args.num_patients or 30
        images_per_patient = args.images_per_patient or 6
        epochs = args.epochs or 5
        ddpm_epochs = args.ddpm_epochs or 3
    else:
        num_patients = args.num_patients or config.NUM_PATIENTS
        images_per_patient = args.images_per_patient or 6
        epochs = args.epochs or config.SAM_EPOCHS
        ddpm_epochs = args.ddpm_epochs or config.DDPM_EPOCHS

    # ------------------------------------------------------------------
    # Ablation study
    # ------------------------------------------------------------------
    if args.ablation:
        print("\n" + "=" * 70)
        print("ABLATION STUDY (Table 3)")
        print("=" * 70)
        ablation_results = run_ablation_study(
            num_patients=min(num_patients, 60),
            images_per_patient=images_per_patient,
            epochs=min(epochs, 15),
            device=args.device,
        )
        print("\nAblation Results:")
        print(f"{'Configuration':<25}{'Acc':<10}{'Prec':<10}{'Rec':<10}{'F1':<10}{'Spec':<10}")
        print("-" * 75)
        for name, metrics in ablation_results.items():
            print(
                f"{name:<25}"
                f"{metrics['accuracy']:.4f}    "
                f"{metrics['precision']:.4f}    "
                f"{metrics['recall']:.4f}    "
                f"{metrics['f1']:.4f}    "
                f"{metrics['specificity']:.4f}"
            )
        return

    # ------------------------------------------------------------------
    # K-fold validation
    # ------------------------------------------------------------------
    if args.kfold:
        print("\n" + "=" * 70)
        print("K-FOLD CROSS-VALIDATION (Table 6)")
        print("=" * 70)
        kfold_results = run_kfold_validation(
            num_patients=min(num_patients, 60),
            images_per_patient=images_per_patient,
            epochs=min(epochs, 10),
            k=config.K_FOLDS,
            device=args.device,
        )
        print("\nK-Fold Results:")
        print(f"{'Fold':<8}{'Acc':<10}{'Prec':<10}{'Rec':<10}{'F1':<10}{'Spec':<10}")
        print("-" * 58)
        for i, m in enumerate(kfold_results["per_fold"], 1):
            print(
                f"{i:<8}"
                f"{m['accuracy']:.4f}    "
                f"{m['precision']:.4f}    "
                f"{m['recall']:.4f}    "
                f"{m['f1']:.4f}    "
                f"{m['specificity']:.4f}"
            )
        s = kfold_results["summary"]
        print("-" * 58)
        print(
            f"{'Mean':<8}"
            f"{s['accuracy']['mean']:.4f}    "
            f"{s['precision']['mean']:.4f}    "
            f"{s['recall']['mean']:.4f}    "
            f"{s['f1']['mean']:.4f}    "
            f"{s['specificity']['mean']:.4f}"
        )
        print(
            f"{'Std':<8}"
            f"{s['accuracy']['std']:.4f}    "
            f"{s['precision']['std']:.4f}    "
            f"{s['recall']['std']:.4f}    "
            f"{s['f1']['std']:.4f}    "
            f"{s['specificity']['std']:.4f}"
        )
        return

    # ------------------------------------------------------------------
    # Evaluate-only mode
    # ------------------------------------------------------------------
    if args.evaluate_only:
        print("\nEvaluating saved model...")
        model_path = os.path.join(config.get_current_version_dir(), "dscnn_sam_model.pt")
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}. Train first.")
            return

        train_loader, val_loader, test_loader, stats = build_dataloaders(
            batch_size=config.SAM_BATCH_SIZE,
            num_patients=num_patients,
            images_per_patient=images_per_patient,
            use_hbv=True,
        )
        model = build_dscnn()
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        model = model.to(args.device)

        run_full_evaluation(model, test_loader, device=args.device)
        return

    # ------------------------------------------------------------------
    # Full training + evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FULL TRAINING PIPELINE")
    print("=" * 70)

    results = run_full_training(
        num_patients=num_patients,
        images_per_patient=images_per_patient,
        use_hbv=True,
        do_train_ddpm=True,
        do_train_classifier=True,
        ddpm_epochs=ddpm_epochs,
        classifier_epochs=epochs,
        device=args.device,
        resume=args.resume,
    )

    # Override epochs if specified (already handled in run_full_training via config)
    # For quick mode we need to retrain with fewer epochs
    if args.quick and "model" not in results:
        print("\nQuick mode: retraining classifier with reduced epochs...")
        from train import train_dscnn_sam
        model, history = train_dscnn_sam(
            results["train_loader"],
            results["val_loader"],
            device=args.device,
            epochs=epochs,
        )
        results["model"] = model
        results["history"] = history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    model = results.get("model")
    if model is None:
        print("No trained model found. Run training first.")
        return

    eval_results = run_full_evaluation(
        model,
        results["test_loader"],
        history=results.get("history"),
        device=args.device,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("\nDataset Distribution:")
    stats = results["stats"]
    print(f"  Training:   {stats['num_train_patients']} patients, {stats['num_train_samples']} images")
    print(f"  Validation: {stats['num_val_patients']} patients, {stats['num_val_samples']} images")
    print(f"  Testing:    {stats['num_test_patients']} patients, {stats['num_test_samples']} images")

    print("\nTest Set Performance:")
    print(format_metric_table(eval_results["test_metrics"]))
    print(f"\nROC AUC: {eval_results['roc_auc']:.4f}")

    print("\nOutputs saved to:", version_dir)
    print("  - run_metadata.json")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - classification_report.png")
    print("  - training_history.png")
    print("  - dscnn_sam_model.pt")
    print("  - ddpm_model.pt")


if __name__ == "__main__":
    main()