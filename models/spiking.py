"""
Spiking neuron components for the Deep Spiking Convolutional Neural Network
(DSCNN).

Implements:
  - Leaky Integrate-and-Fire (LIF) neuron layer with surrogate gradient
    (Straight-Through Estimator) training.
  - Spike encoding of input intensities into spike trains over time steps.

Equations follow the paper:
  - Membrane potential integration:   U(t) = tau_m * U(t-1) + I(t)
  - Fire when U >= V_th, then reset.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateGradient(torch.autograd.Function):
    """Sigmoid-like surrogate gradient for spike firing (wider gradient flow)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        return (x >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        (x,) = ctx.saved_tensors
        # Sigmoid surrogate gradient: provides gradients across a wide range
        # of membrane potentials, preventing gradient vanishing
        alpha = 5.0
        z = alpha * (x - ctx.threshold)
        sigmoid = torch.sigmoid(z)
        grad = alpha * sigmoid * (1.0 - sigmoid)
        return grad_output * grad, None


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer.

    Args:
        threshold:  firing threshold V_th (paper: 1.0)
        tau_m:      membrane time constant (leak)
        init_v:     initial membrane potential (paper: 0.0)
        reset_mode: 'subtract' subtracts threshold after firing
    """

    def __init__(
        self,
        threshold: float = 1.0,
        tau_m: float = 2.0,
        init_v: float = 0.0,
        reset_mode: str = "subtract",
    ):
        super().__init__()
        self.threshold = threshold
        self.tau_m = tau_m
        self.init_v = init_v
        self.reset_mode = reset_mode

    def forward(
        self,
        input_current: torch.Tensor,
        state_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_current: input current I(t) [B, C, H, W]
            state_v:       previous membrane potential

        Returns:
            (spikes, updated membrane potential)
        """
        if state_v is None:
            state_v = torch.full_like(input_current, self.init_v)

        # Leaky integration (Eq. 4 in paper)
        decay = math.exp(-1.0 / self.tau_m)
        v = decay * state_v * (1.0 - (state_v >= self.threshold).float())
        v = v + input_current

        # Fire
        spikes = SurrogateGradient.apply(v, self.threshold)

        # Reset
        if self.reset_mode == "subtract":
            v = v - spikes * self.threshold
        else:  # 'zero'
            v = v * (1.0 - spikes)

        return spikes, v


class SpikeEncoding(nn.Module):
    """
    Converts real-valued input intensities into signed spike trains over time.

    Uses deterministic quantization-based encoding (as described in the paper):
    - Brighter pixels (high intensity) produce more positive spikes
    - Darker pixels (low intensity) produce more negative spikes
    - The spike count over T timesteps is proportional to the intensity

    This is deterministic (no random sampling), ensuring stable training
    and consistent predictions during evaluation.
    """

    def __init__(self, num_steps: int = 20, threshold: float = 1.0):
        super().__init__()
        self.num_steps = num_steps
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: real-valued input [B, C, H, W] in [0, 1]

        Returns:
            spike trains [B, T, C, H, W] with signed binary spikes
        """
        prob = torch.clamp(x, 0.0, 1.0)
        spikes = []
        for t in range(self.num_steps):
            # Positive spike if intensity > (t+1)/T (bright pixels fire early)
            pos = (prob > (t + 1) / self.num_steps).float()
            # Negative spike if intensity < t/T (dark pixels fire)
            neg = (prob < t / self.num_steps).float()
            step_spikes = pos - neg
            spikes.append(step_spikes)
        return torch.stack(spikes, dim=1)  # [B, T, C, H, W]


if __name__ == "__main__":
    # Smoke test
    enc = SpikeEncoding(num_steps=20)
    lif = LIFNeuron(threshold=1.0, tau_m=2.0)
    x = torch.rand(2, 1, 8, 8)
    spikes_t = enc(x)
    print("Encoded spikes:", spikes_t.shape)
    v = None
    for t in range(spikes_t.shape[1]):
        out_spikes, v = lif(spikes_t[:, t], v)
    print("LIF output spikes:", out_spikes.shape, "final V:", v.mean().item())