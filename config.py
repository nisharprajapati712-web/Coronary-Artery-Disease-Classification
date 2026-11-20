"""
Configuration for the DSCNN-SAM model for CAD detection.
All hyperparameters are taken from the paper:
"Coronary Artery Disease Detection using Deep Spiking Convolutional Neural
 Network with Sharpness-Aware Minimization" (Table 1).
"""

import os
import json
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")       # Place Kaggle CCTA images here
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")    # Cached preprocessed (resized+HBV) images
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Versioned output management
# Each execution creates a new versioned output directory (v1, v2, v3, ...)
# so results from different runs are stored separately and can be compared.
# ---------------------------------------------------------------------------
VERSION_FILE = os.path.join(OUTPUT_DIR, "version_counter.json")


def _load_version_counter() -> int:
    """Load the current version counter from disk."""
    if os.path.exists(VERSION_FILE):
        try:
            with open(VERSION_FILE, "r") as f:
                return int(json.load(f).get("version", 0))
        except (json.JSONDecodeError, ValueError, KeyError):
            return 0
    return 0


def _save_version_counter(version: int) -> None:
    """Persist the version counter to disk."""
    with open(VERSION_FILE, "w") as f:
        json.dump({"version": version}, f)


def get_next_version_dir() -> str:
    """
    Create and return the next versioned output directory (v1, v2, v3, ...).

    Each call increments the version counter and creates a new directory
    under outputs/, e.g. outputs/v1, outputs/v2, etc.
    """
    version = _load_version_counter() + 1
    version_dir = os.path.join(OUTPUT_DIR, f"v{version}")
    os.makedirs(version_dir, exist_ok=True)
    _save_version_counter(version)
    return version_dir


def get_current_version_dir() -> str:
    """Return the current (latest) versioned output directory."""
    version = _load_version_counter()
    if version == 0:
        return OUTPUT_DIR
    return os.path.join(OUTPUT_DIR, f"v{version}")

# If True, generate synthetic CCTA-like images instead of requiring the
# Kaggle dataset.  Set to False once the real dataset is available.
USE_SYNTHETIC_DATA = False

# Image settings
IMG_SIZE = 224                     # 224 x 224 x 1 as described in the paper
IMG_CHANNELS = 1

# ---------------------------------------------------------------------------
# Dataset (Tables 1 & 2)
# ---------------------------------------------------------------------------
NUM_PATIENTS = 200
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test
SEED = 42

# ---------------------------------------------------------------------------
# HBV (Hessian-Based Vesselness) filter
# ---------------------------------------------------------------------------
HBV_SCALES = [1.0, 2.0, 3.0]       # multi-scale sigma values
HBV_BETA = 0.5                     # sensitivity controlling response sharpness
HBV_GAMMA = 0.5
HBV_C = 1e-10                      # small constant to avoid division by zero

# ---------------------------------------------------------------------------
# DDPM segmentation (Table 1)
# ---------------------------------------------------------------------------
DDPM_IMG_SIZE = 64                 # internal resolution for DDPM (memory efficient)
DDPM_TIMESTEPS = 1000              # diffusion steps
DDPM_BETA_START = 0.0001
DDPM_BETA_END = 0.02
DDPM_NET = "U-Net"                 # U-Net backbone
DDPM_OPTIMIZER = "Adam"
DDPM_LR = 0.0001
DDPM_LOSS = "Dice + BCE"
DDPM_EPOCHS = 50
DDPM_BATCH_SIZE = 16
DDPM_BASE_CHANNELS = 32            # U-Net base channel count

# ---------------------------------------------------------------------------
# DSCNN classifier (Table 1)
# ---------------------------------------------------------------------------
DSCNN_NEURON = "Leaky Integrate and Fire (LIF)"
DSCNN_TIMESTEPS = 20               # T = 20 discrete timesteps
DSCNN_PHYSICAL_TIMESTEP_MS = 1.0   # 1 ms per step
DSCNN_SIMULATION_DURATION_MS = 20.0  # total simulation 20 ms
DSCNN_THRESHOLD = 1.0              # spiking threshold (V_th)
DSCNN_TAU_M = 2.0                  # membrane time constant for LIF decay
DSCNN_INIT_V = 0.0                 # initial membrane potential
DSCNN_NUM_CLASSES = 2              # CAD / non-CAD

# ---------------------------------------------------------------------------
# SAM optimization (Table 1)
# ---------------------------------------------------------------------------
SAM_RHO = 0.5                      # neighbourhood radius rho
SAM_BASE_OPTIMIZER = "Adam"
SAM_LR = 0.001
SAM_WEIGHT_DECAY = 0.0001
SAM_BATCH_SIZE = 32
# Paper Table 1 specifies 150 as the training ceiling, but at ~16 min/epoch
# that's up to ~40 hours in the worst case. Observed val_acc plateaus around
# epoch 20-30 (0.62 -> 0.73 -> flat), so capping here to fit an 8-hour budget
# trades away the unused tail of the 150-epoch ceiling, not real accuracy --
# early stopping below still triggers sooner if it plateaus earlier than this.
SAM_EPOCHS = 28
# Stop early once val_acc hasn't improved for this many epochs.
SAM_EARLY_STOP_PATIENCE = 20

# Validation is forward-only (no SAM double backward), so it tolerates a much
# bigger batch than training without the memory issues batch=64 caused there.
VAL_BATCH_SIZE = 128
# Run validation every N epochs instead of every epoch to trim wall time;
# does not change the trained weights, only how often we check them. Early
# stopping patience is still honored in epoch-equivalent terms.
VAL_EVERY_N_EPOCHS = 2

# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
NUM_WORKERS = 8
PIN_MEMORY = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mixed precision (bf16 autocast) — halves memory and speeds up GPU compute
# with negligible accuracy impact (bf16 has fp32's exponent range, so no
# gradient scaler is required, unlike fp16).
USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16

# torch.compile would fuse the many small sequential ops in the LIF timestep
# loop into fewer GPU kernel launches, but Triton's compiler backend requires
# the python3-dev headers (Python.h), which aren't installed on this system --
# disabled until that's available (`sudo apt install python3-dev`).
USE_COMPILE = False

# K-fold validation (Table 6)
K_FOLDS = 5

# Repeated experimental analysis (Table 7)
REPEAT_RUNS = 5
REPEAT_SEEDS = [42, 7, 123, 2024, 99]

# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------
CLASS_NAMES = ["non-CAD", "CAD"]