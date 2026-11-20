# DSCNN-SAM: Coronary Artery Disease Detection

Implementation of the paper:

> **"Coronary Artery Disease Detection using Deep Spiking Convolutional Neural Network with Sharpness-Aware Minimization"**

## Overview

The proposed **DSCNN-SAM** model detects Coronary Artery Disease (CAD) from CCTA images through a 4-stage pipeline:

1. **HBV Preprocessing** – Hessian-Based Vesselness filter enhances coronary vessel visibility and suppresses background noise.
2. **DDPM Segmentation** – Denoising Diffusion Probabilistic Model (U-Net backbone, 1000 linear diffusion steps) isolates coronary artery regions, producing binary vessel probability maps.
3. **DSCNN Detection** – Deep Spiking Convolutional Neural Network with Leaky Integrate-and-Fire (LIF) neurons converts image features into spike-based temporal representations (20 timesteps, 1ms each).
4. **SAM Optimization** – Sharpness-Aware Minimization (ρ=0.5) guides training toward flat loss minima for improved generalization.

## Project Structure

```
cad-dscnn-sam/
├── config.py              # All hyperparameters (paper Table 1)
├── main.py                # End-to-end entry point
├── train.py               # Training pipeline (DDPM + DSCNN-SAM)
├── evaluate.py            # Evaluation (confusion matrix, ROC, ablation, k-fold)
├── data/
│   ├── dataset.py         # CCTA loader + patient-level split
│   └── synthetic.py       # Synthetic CCTA-like generator (fallback)
├── models/
│   ├── hbv.py             # Hessian-Based Vesselness filter
│   ├── ddpm.py            # DDPM with U-Net backbone
│   ├── spiking.py         # LIF neuron + surrogate gradient
│   ├── dscnn.py           # Deep Spiking CNN classifier
│   └── sam.py             # SAM optimizer wrapper
└── utils/
    └── metrics.py         # Accuracy, precision, recall, F1, specificity
```

## Installation

```bash
cd cad-dscnn-sam
pip install -r requirements.txt
```

## Usage

### Quick smoke test (synthetic data, reduced epochs)

```bash
python main.py --quick
```

### Full training + evaluation

```bash
python main.py
```

### Ablation study (Table 3)

```bash
python main.py --ablation
```

### K-fold cross-validation (Table 6)

```bash
python main.py --kfold
```

### Evaluate a saved model

```bash
python main.py --evaluate-only
```

### Custom options

```bash
python main.py --num-patients 200 --images-per-patient 6 --epochs 150 --device cpu
```

## Dataset


To use the **Kaggle CCTA dataset** (`coronary-artery-segmentation-of-ccta`):

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kvipularya/coronary-artery-segmentation-of-ccta).
2. Organize images as:
   ```
   data/raw/
     label_0/patient_000/img_00000.png
     label_0/patient_000/mask_00000.npy
     ...
     label_1/patient_001/img_00000.png
     label_1/patient_001/mask_00000.npy
     ...
   ```
3. Set `USE_SYNTHETIC_DATA = False` in `config.py`.





