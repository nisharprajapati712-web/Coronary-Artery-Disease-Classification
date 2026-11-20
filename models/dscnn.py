"""
Deep Spiking Convolutional Neural Network (DSCNN) classifier.

The DSCNN:
  1. Converts input intensities into spike-based temporal representations
     (signed spike trains via SpikeEncoding).
  2. Processes spikes through successive spiking convolutional layers
     (Conv → LIF) over T timesteps.
  3. Aggregates spike outputs via global average pooling + fully connected
     classification layer with Softmax for CAD / non-CAD.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.spiking import LIFNeuron, SpikeEncoding


class SpikingConvBlock(nn.Module):
    """Conv2d → BatchNorm → LIF spiking neuron."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        threshold: float = 1.0,
        tau_m: float = 2.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = LIFNeuron(threshold=threshold, tau_m=tau_m)

    def forward(
        self,
        x: torch.Tensor,
        state_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.bn(self.conv(x))
        return self.lif(current, state_v)

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Process an entire [B, T, C, H, W] timestep sequence at once.

        Conv2d/BatchNorm2d weights are shared (time-invariant) across
        timesteps, so the conv+BN portion has no recurrence and can be
        computed in a single batched call over B*T instead of T separate
        calls -- this is the expensive part (kernel-launch-bound when done
        per-timestep). Only the LIF neuron is recurrent (membrane potential
        carried across t), so that part still loops over T, but those are
        cheap elementwise ops, not conv/BN.
        """
        B, T, C, H, W = x_seq.shape
        current = self.bn(self.conv(x_seq.reshape(B * T, C, H, W)))
        current = current.reshape(B, T, *current.shape[1:])

        v = None
        spikes = []
        for t in range(T):
            s_t, v = self.lif(current[:, t], v)
            spikes.append(s_t)
        return torch.stack(spikes, dim=1)  # [B, T, C_out, H, W]


class DSCNN(nn.Module):
    """
    Deep Spiking Convolutional Neural Network.

    Architecture (paper Fig. 1):
      Input (224x224x1)
        → SpikeEncoding (T timesteps)
        → Spiking Conv blocks
        → Global Average Pooling
        → Fully Connected → 2 neurons → Softmax
    """

    def __init__(
        self,
        num_steps: int = 20,
        threshold: float = 1.0,
        tau_m: float = 2.0,
        num_classes: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.threshold = threshold
        self.tau_m = tau_m

        self.encoding = SpikeEncoding(num_steps=num_steps, threshold=threshold)

        # Spiking convolutional backbone
        self.block1 = SpikingConvBlock(1, 32, threshold=threshold, tau_m=tau_m)
        self.block2 = SpikingConvBlock(32, 64, threshold=threshold, tau_m=tau_m)
        self.block3 = SpikingConvBlock(64, 128, threshold=threshold, tau_m=tau_m)

        self.pool = nn.MaxPool2d(2)

        # Classification head
        self.fc = nn.Linear(128, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def _compute_spatial_size(self, input_size: int) -> int:
        """Compute spatial size after pooling layers for the classifier."""
        size = input_size
        for _ in range(3):
            size = (size - 1) // 2 + 1  # maxpool with k=2, s=2
        return size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input images [B, 1, H, W], values in [0, 1]

        Returns:
            logits [B, num_classes]
        """
        # Encode into signed spike trains [B, T, C, H, W]
        spikes_t = self.encoding(x)

        # Conv+BN are time-invariant (no recurrence), so each block batches
        # all T timesteps into a single conv/BN call instead of T separate
        # calls -- only the LIF membrane potential is carried sequentially
        # across timesteps (paper: "LIF neuron model accumulates membrane
        # potential over time and generates spikes when threshold is
        # enhanced"). This is mathematically equivalent per-sample and only
        # changes the batch grouping used for BatchNorm's running statistics.
        def pool_seq(s_seq: torch.Tensor) -> torch.Tensor:
            B, T, C, H, W = s_seq.shape
            pooled = self.pool(s_seq.reshape(B * T, C, H, W))
            return pooled.reshape(B, T, *pooled.shape[1:])

        s1_seq = pool_seq(self.block1.forward_sequence(spikes_t))
        s2_seq = pool_seq(self.block2.forward_sequence(s1_seq))
        s3_seq = pool_seq(self.block3.forward_sequence(s2_seq))

        # Average over time and spatial dims → firing rate [B, 128]
        rate = s3_seq.mean(dim=(1, 3, 4))

        # Classification head
        h = F.relu(self.fc(rate))
        logits = self.fc_out(h)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return F.softmax(self.forward(x), dim=1)


def build_dscnn() -> DSCNN:
    return DSCNN(
        num_steps=config.DSCNN_TIMESTEPS,
        threshold=config.DSCNN_THRESHOLD,
        tau_m=config.DSCNN_TAU_M,
        num_classes=config.DSCNN_NUM_CLASSES,
    )


if __name__ == "__main__":
    model = build_dscnn()
    x = torch.rand(2, 1, 64, 64)
    out = model(x)
    print("DSCNN output:", out.shape)
    print("Softmax:", F.softmax(out, dim=1))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")