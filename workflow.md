# DSCNN-SAM: Complete Workflow Documentation

## 1. Paper Reference

**Title:** Coronary Artery Disease Detection using Deep Spiking Convolutional Neural Network with Sharpness-Aware Minimization

**Proposed Model:** DSCNN-SAM (Deep Spiking Convolutional Neural Network with Sharpness-Aware Minimization)

**Reported Accuracy:** 99.44% on CCTA dataset

---

## 2. Dataset Details

### 2.1 Primary Dataset: CCTA (Coronary Computed Tomography Angiography)

| Property | Value |
|---|---|
| **Dataset Name** | CCTA Dataset (Coronary Artery Segmentation) |
| **Source** | Kaggle |
| **Link** | https://www.kaggle.com/datasets/kvipularya/coronary-artery-segmentation-of-ccta |
| **Total Patients** | 200 |
| **Male** | 108 |
| **Female** | 92 |
| **Mean Age** | 50.6 ± 12.1 years |
| **CAD Patients** | 100 (confirmed atherosclerosis) |
| **Non-CAD Patients** | 100 |
| **Total Images** | 2,364 (1,182 CAD + 1,182 non-CAD) |
| **Image Size** | 224 × 224 × 1 (grayscale) |
| **Imaging Modality** | Multi-detector CT systems |
| **Clinical Indication** | Chest pain / known CAD |
| **Classes** | Binary: CAD (1) / non-CAD (0) |

### 2.2 Dataset Split (Table 2 - Patient-Level, No Data Leakage)

| Partition | Patients | CAD Patients | Non-CAD Patients | CAD Images | Non-CAD Images |
|---|---|---|---|---|---|
| **Training (70%)** | 140 | 70 | 70 | 828 | 827 |
| **Validation (15%)** | 30 | 15 | 15 | 177 | 177 |
| **Testing (15%)** | 30 | 15 | 15 | 177 | 178 |
| **Total** | **200** | **100** | **100** | **1,182** | **1,182** |

**Important:** Patient-level partitioning is performed BEFORE image extraction to prevent data leakage (images from the same patient never appear in both train and test).

### 2.3 Synthetic Data Fallback

The code includes a **synthetic CCTA-like generator** (`data/synthetic.py`) that produces:
- Tubular vessel structures (coronary arteries) with branching
- Atherosclerotic plaque features (bright/high-intensity regions) for CAD class
- Imaging noise (Gaussian + salt-pepper)
- Motion/streak artifacts
- Contrast variations
- Ground-truth vessel masks for DDPM segmentation training

This allows the full pipeline to run end-to-end without Kaggle credentials. Set `USE_SYNTHETIC_DATA = False` in `config.py` to use the real dataset.

---

## 3. Proposed Model Architecture (DSCNN-SAM)

### 3.1 Pipeline Overview (4 Stages)

```
Input CCTA Image (224×224×1)
    │
    ▼
[Stage 1] HBV Preprocessing
    │  Hessian-Based Vesselness filter
    │  - Enhances coronary vessel visibility
    │  - Suppresses background noise
    │  - Multi-scale Hessian eigenvalue analysis (σ = 1, 2, 3)
    ▼
[Stage 2] DDPM Segmentation
    │  Denoising Diffusion Probabilistic Model
    │  - U-Net backbone with encoder-decoder skip connections
    │  - 1000 linear diffusion steps
    │  - Produces binary vessel probability maps
    ▼
[Stage 3] DSCNN Detection
    │  Deep Spiking Convolutional Neural Network
    │  - LIF neurons (Leaky Integrate-and-Fire)
    │  - 20 timesteps (1ms each, 20ms total simulation)
    │  - Converts features into spike-based temporal representations
    │  - Global Average Pooling + FC + Softmax (2 classes)
    ▼
[Stage 4] SAM Optimization
    │  Sharpness-Aware Minimization
    │  - ρ = 0.5 neighbourhood radius
    │  - Converges to flat minima for better generalization
    ▼
Output: CAD / non-CAD Classification
```

### 3.2 Model Components

#### Stage 1: HBV (Hessian-Based Vesselness) Filter
- **File:** `models/hbv.py`
- **Purpose:** Enhance tubular/coronary vessel structures, suppress noise
- **Method:** Multi-scale Hessian eigenvalue analysis (Frangi-style)
- **Scales:** σ = [1.0, 2.0, 3.0]
- **Parameters:** β = 0.5 (blobness), γ = 0.5 (structuredness)
- **Equations:** Paper Eqs. (2)-(3) - vesselness response based on eigenvalue ratios

#### Stage 2: DDPM (Denoising Diffusion Probabilistic Model)
- **File:** `models/ddpm.py`
- **Purpose:** Isolate coronary artery regions (vessel segmentation)
- **Backbone:** U-Net with encoder-decoder + skip connections
- **Diffusion Steps:** 1000 (linear noise schedule, β: 0.0001 → 0.02)
- **Internal Resolution:** 64×64
- **Loss:** Dice + BCE (paper Table 1)
- **Optimizer:** Adam, lr = 0.0001
- **Epochs:** 50
- **Sampling:** DDIM-style reverse diffusion

#### Stage 3: DSCNN (Deep Spiking Convolutional Neural Network)
- **File:** `models/dscnn.py` + `models/spiking.py`
- **Purpose:** CAD classification using spike-based temporal learning
- **Neuron Model:** Leaky Integrate-and-Fire (LIF)
- **Time Steps (T):** 20 (1ms each, 20ms total simulation)
- **Spiking Threshold:** 1.0
- **Membrane Time Constant (τm):** 2.0
- **Surrogate Gradient:** Straight-Through Estimator (STE)
- **Architecture:**
  - SpikeEncoding (signed spike trains)
  - Spiking Conv Block 1: 1→32 channels
  - Spiking Conv Block 2: 32→64 channels
  - Spiking Conv Block 3: 64→128 channels
  - MaxPool after each block
  - Global Average Pooling
  - FC: 128→256→2 (Softmax)

#### Stage 4: SAM (Sharpness-Aware Minimization)
- **File:** `models/sam.py`
- **Purpose:** Improve generalization by converging to flat minima
- **Neighbourhood Radius (ρ):** 0.5
- **Base Optimizer:** Adam
- **Learning Rate:** 0.001
- **Weight Decay:** 0.0001
- **Method:** Two-step gradient (ascent perturbation → true update)

---

## 4. Hyperparameters (Table 1)

| Hyperparameter | Value |
|---|---|
| SNN neuron model | Leaky Integrate and Fire (LIF) |
| Network (DDPM) | U-Net |
| Optimizer | Adam |
| Time Steps (T) | 20 |
| Spiking threshold (V_th) | 1.0 |
| Physical timestep | 1 ms |
| Total simulation duration | 20 ms |
| SAM rho (ρ) | 0.5 |
| Base optimizer | Adam |
| Learning Rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 16 |
| Number of epochs (DSCNN) | 150 |
| Number of epochs (DDPM) | 50 |
| Loss Function (DDPM) | Dice + BCE |
| Noise Schedule | Linear |
| Diffusion steps | 1000 |
| Input image size | 224 × 224 × 1 |

---

## 5. Complete File Structure

```
cad-dscnn-sam/
├── config.py              # All hyperparameters (paper Table 1)
├── main.py                # End-to-end entry point
├── train.py               # Training pipeline (DDPM + DSCNN-SAM)
├── evaluate.py            # Evaluation (confusion matrix, ROC, ablation, k-fold)
├── test_smoke.py          # Component verification test
├── workflow.md            # This documentation
├── README.md              # Quick-start guide
├── requirements.txt       # Dependencies
├── data/
│   ├── __init__.py
│   ├── dataset.py         # CCTA loader + patient-level split
│   └── synthetic.py       # Synthetic CCTA-like generator
├── models/
│   ├── __init__.py
│   ├── hbv.py             # Hessian-Based Vesselness filter
│   ├── ddpm.py            # DDPM with U-Net backbone
│   ├── spiking.py         # LIF neuron + surrogate gradient
│   ├── dscnn.py           # Deep Spiking CNN classifier
│   └── sam.py             # SAM optimizer wrapper
├── utils/
│   ├── __init__.py
│   └── metrics.py         # Accuracy, precision, recall, F1, specificity
└── outputs/               # Generated results (plots, models)
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── classification_report.png
    ├── training_history.png
    ├── dscnn_sam_model.pt
    └── ddpm_model.pt
```

---

## 6. Complete Workflow (Execution Flow)

### 6.1 Entry Point: `main.py`

```
main.py
    │
    ├── Parse arguments (--quick, --ablation, --kfold, --evaluate-only, etc.)
    │
    ├── If --ablation: Run ablation study (Table 3)
    │   ├── Excluding HBV
    │   ├── Excluding DDPM
    │   ├── Excluding SAM
    │   └── Proposed DSCNN-SAM
    │
    ├── If --kfold: Run 5-fold cross-validation (Table 6)
    │
    ├── If --evaluate-only: Load saved model and evaluate
    │
    └── Full training + evaluation:
        ├── [1/3] Build dataloaders (patient-level split)
        ├── [2/3] Train DDPM segmentation (50 epochs)
        ├── [3/3] Train DSCNN-SAM classifier (150 epochs)
        └── Evaluate: confusion matrix, ROC, classification report, computational analysis
```

### 6.2 Training Pipeline: `train.py`

```
run_full_training()
    │
    ├── build_dataloaders() → train/val/test loaders
    │
    ├── train_ddpm() → DDPM vessel segmentation
    │   ├── Adam optimizer (lr=0.0001)
    │   ├── Dice + BCE loss
    │   ├── 50 epochs
    │   └── Saves best model to outputs/ddpm_model.pt
    │
    └── train_dscnn_sam() → DSCNN classifier with SAM
        ├── SAM optimizer (ρ=0.5, Adam base, lr=0.001)
        ├── CrossEntropyLoss
        ├── 150 epochs
        ├── SAM two-step per batch:
        │   1. Forward + backward → gradients
        │   2. sam.first_step() → perturb params
        │   3. Forward + backward at perturbed params
        │   4. sam.second_step() → restore + update
        └── Saves best model to outputs/dscnn_sam_model.pt
```

### 6.3 Evaluation: `evaluate.py`

```
run_full_evaluation()
    ├── Test set metrics (accuracy, precision, recall, F1, specificity)
    ├── Confusion matrix plot (Fig. 4)
    ├── ROC curve with AUC (Fig. 5)
    ├── Classification report plot (Fig. 6)
    ├── Training history plot
    └── Computational analysis (params, FLOPs, latency, throughput)
```

---

## 7. Evaluation Metrics (Eqs. 11-14)

| Metric | Formula |
|---|---|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) |
| **Precision** | TP / (TP + FP) |
| **Recall (Sensitivity)** | TP / (TP + FN) |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) |
| **Specificity** | TN / (TN + FP) |

Where: TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative

---

## 8. Experimental Results (from Paper)

### 8.1 Ablation Study (Table 3)

| Model Configuration | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Specificity (%) |
|---|---|---|---|---|---|
| Excluding HBV | 96.80 ± 0.24 | 96.20 ± 0.28 | 96.50 ± 0.25 | 96.30 ± 0.26 | 96.65 ± 0.23 |
| Excluding DDPM | 97.50 ± 0.19 | 97.00 ± 0.21 | 97.20 ± 0.20 | 97.10 ± 0.20 | 97.42 ± 0.18 |
| Excluding SAM | 98.70 ± 0.15 | 98.30 ± 0.18 | 98.40 ± 0.16 | 98.30 ± 0.17 | 98.58 ± 0.15 |
| **Proposed DSCNN-SAM** | **99.44 ± 0.11** | **98.97 ± 0.10** | **98.94 ± 0.10** | **98.95 ± 0.10** | **99.43 ± 0.09** |

### 8.2 K-Fold Validation (Table 6)

| Fold | Accuracy (%) |
|---|---|
| 1 | 99.31 ± 0.10 |
| 2 | 99.58 ± 0.09 |
| 3 | 99.42 ± 0.11 |
| 4 | 99.55 ± 0.08 |
| 5 | 99.34 ± 0.12 |
| **Mean ± SD** | **99.44 ± 0.12** |

### 8.3 Comparative Analysis (Table 8)

| Method | Accuracy (%) |
|---|---|
| AO-UNet++ CNN | 99.50 ± 0.14 |
| PlaqueNet | 93.12 ± 0.31 |
| ML-derived FFRCT | 87.46 ± 0.42 |
| SVM | 75.59 ± 0.52 |
| MCSVM | 60.12 ± 0.71 |
| Nisha K. Prajapati [21] | 98.33 ± 0.17 |
| Nisha K. Prajapati [22] | 97.19 ± 0.22 |
| Nishaben Kantilal Prajapati [23] | 99.42 ± 0.12 |
| **Proposed DSCNN-SAM** | **99.44 ± 0.11** |

---

## 9. Dependencies (requirements.txt)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
Pillow>=9.5.0
tqdm>=4.65.0
```

---

## 10. Usage Commands

```bash
# Install dependencies
cd cad-dscnn-sam
pip install -r requirements.txt

# Quick smoke test (synthetic data, reduced epochs)
python main.py --quick

# Full training + evaluation (paper settings)
python main.py

# Ablation study (Table 3)
python main.py --ablation

# K-fold cross-validation (Table 6)
python main.py --kfold

# Evaluate a saved model
python main.py --evaluate-only

# Custom options
python main.py --num-patients 200 --images-per-patient 6 --epochs 150 --device cpu
```

---

## 11. Key Implementation Notes

1. **Patient-level splitting** prevents data leakage - critical for medical imaging
2. **Surrogate gradient (sigmoid)** enables backpropagation through non-differentiable spike firing with wide gradient flow
3. **SAM two-step optimization** finds flat minima for better generalization
4. **DDPM** is used as an auxiliary vessel-extraction module (not for classification)
5. **Synthetic data generator** allows end-to-end testing without Kaggle credentials
6. **ETA estimation** shows remaining training time every 10 epochs
7. **Best model checkpointing** saves the best-performing model during training
8. **No early stopping** - training always runs the full configured epoch count
9. **Deterministic spike encoding** - quantization-based (no random sampling) for stable training
10. **Stateful LIF neurons** - membrane potentials maintained across timesteps for temporal integration
11. **Versioned outputs** - each execution saves to `outputs/v1`, `outputs/v2`, `outputs/v3`, etc.

## 11.1 Versioned Output System

Each execution of `main.py` creates a new versioned output directory:

```
outputs/
├── version_counter.json   # Tracks current version (auto-incremented)
├── v1/                    # First run
│   ├── run_metadata.json  # Run config (epochs, patients, device, timestamp)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── classification_report.png
│   ├── training_history.png
│   ├── dscnn_sam_model.pt
│   └── ddpm_model.pt
├── v2/                    # Second run (after code update)
└── v3/                    # Third run
```

This allows comparing results across code changes and tracking model improvements.

---

## 12. Execution Time Estimates

| Scenario | DDPM (50 epochs) | DSCNN-SAM (150 epochs) | Total |
|---|---|---|---|
| CPU (Intel i7) | ~1.5 hours | ~11 hours | **~12.5 hours** |
| GPU (CUDA) | ~2 minutes | ~10-15 minutes | **~15-20 minutes** |
| Quick mode | ~3 minutes | ~10 minutes | **~15 minutes** |

Here are all the commands needed to execute the DSCNN-SAM project:

## 1. Setup Commands (First Time Only)

```bash
# Navigate to project folder
cd C:\Users\DELL\OneDrive\Desktop\Antigravity\cad-dscnn-sam

# Install dependencies
pip install -r requirements.txt
```

## 2. Execution Commands

### Quick Smoke Test (~15 min)
```bash
python main.py --quick
```

### Full Training + Evaluation (paper settings: 50 DDPM + 150 DSCNN-SAM epochs)
```bash
python main.py
```

### Ablation Study (Table 3)
```bash
python main.py --ablation
```

### K-Fold Cross-Validation (Table 6)
```bash
python main.py --kfold
```

### Evaluate a Saved Model
```bash
python main.py --evaluate-only
```

### Component Verification Test
```bash
python test_smoke.py
```

## 3. Custom Options

```bash
# Specify number of patients
python main.py --num-patients 200

# Specify images per patient
python main.py --images-per-patient 6

# Override epochs
python main.py --epochs 150 --ddpm-epochs 50

# Use GPU (if available)
python main.py --device cuda

# Use CPU
python main.py --device cpu

# Combined example
python main.py --num-patients 200 --images-per-patient 6 --epochs 150 --ddpm-epochs 50 --device cpu
```

## 4. Quick Reference (Copy-Paste Ready)

```bash
cd C:\Users\DELL\OneDrive\Desktop\Antigravity\cad-dscnn-sam
pip install -r requirements.txt
python main.py --quick
```

## 5. Output Files (after execution)

All results are saved to `outputs/`:
- `confusion_matrix.png` — Confusion matrix (Fig. 4)
- `roc_curve.png` — ROC curve with AUC (Fig. 5)
- `classification_report.png` — Classification report (Fig. 6)
- `training_history.png` — Loss/accuracy curves
- `dscnn_sam_model.pt` — Trained DSCNN-SAM classifier
- `ddpm_model.pt` — Trained DDPM segmentation model