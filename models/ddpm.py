"""
Denoising Diffusion Probabilistic Model (DDPM) for coronary artery vessel
segmentation.

Implements:
  - Forward diffusion process (Gaussian noise schedule, linear)
  - Reverse denoising network (U-Net backbone with encoder-decoder skip connections)
  - Final segmentation head producing binary vessel probability maps

As described in the paper, the DDPM is used as an auxiliary vessel-extraction
module to isolate coronary artery regions before DSCNN classification.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.time_mlp = (
            nn.Linear(time_dim, out_ch) if time_dim > 0 else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        if t_emb is not None and not isinstance(self.time_mlp, nn.Identity):
            t = self.time_mlp(self.act(t_emb))
            x = x + t[:, :, None, None]
        return x


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int = 64):
        super().__init__()
        self.double_conv = DoubleConv(in_ch, out_ch, time_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None):
        skip = self.double_conv(x, t_emb)
        return self.pool(skip), skip


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int = 64):
        super().__init__()
        # Transposed conv takes the bottleneck output (in_ch // 2 channels)
        # and upsamples it, then concat with skip (in_ch // 2) → in_ch total
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_ch, out_ch, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: Optional[torch.Tensor] = None):
        x = self.up(x)
        # Handle size mismatches from odd input sizes
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x, t_emb)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Unet(nn.Module):
    """
    U-Net backbone for the DDPM reverse denoising network.
    Encoder-decoder architecture with skip connections.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        time_dim: int = 64,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels, time_dim)
        self.down1 = Down(base_channels, base_channels * 2, time_dim)
        self.down2 = Down(base_channels * 2, base_channels * 4, time_dim)
        self.down3 = Down(base_channels * 4, base_channels * 8, time_dim)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 8, time_dim)

        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 4, time_dim)
        self.up2 = Up(base_channels * 8, base_channels * 2, time_dim)
        self.up3 = Up(base_channels * 4, base_channels, time_dim)

        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        # Encoder
        x1 = self.inc(x, t_emb)
        x2, skip1 = self.down1(x1, t_emb)
        x3, skip2 = self.down2(x2, t_emb)
        x4, skip3 = self.down3(x3, t_emb)

        # Bottleneck
        x = self.bottleneck(x4, t_emb)

        # Decoder
        x = self.up1(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up3(x, skip1, t_emb)

        return self.outc(x)


# ---------------------------------------------------------------------------
# DDPM
# ---------------------------------------------------------------------------
class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model for vessel segmentation.

    The forward diffusion process adds Gaussian noise over the diffusion
    timesteps; the reverse process learns to estimate the injected noise using
    the U-Net model, and the final segmentation head produces binary vessel
    probability maps.
    """

    def __init__(
        self,
        img_size: int = 64,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        base_channels: int = 32,
        time_dim: int = 64,
    ):
        super().__init__()
        self.img_size = img_size
        self.timesteps = timesteps
        self.base_channels = base_channels

        # Linear noise schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

        self.unet = Unet(
            in_channels=1, out_channels=1,
            base_channels=base_channels, time_dim=time_dim,
        )

    # ---- Diffusion helpers ----
    def _to(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.betas.device)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise"""
        if noise is None:
            noise = torch.randn_like(x0)
        t = t.long()
        alpha_bar = self._to(self.alpha_bar[t])[:, None, None, None]
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor, seg_target: torch.Tensor):
        """
        Training loss: MSE on predicted noise + Dice/BCE on predicted segmentation.

        Args:
            x0:          clean (vessel-enhanced) image [B, 1, H, W]
            t:            diffusion timesteps [B]
            seg_target:   binary vessel mask [B, 1, H, W]
        """
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        t_emb = t.float()

        # Predicted noise by U-Net
        pred_noise = self.unet(x_t, t_emb)

        # Noise prediction loss (MSE)
        loss_mse = F.mse_loss(pred_noise, noise)

        # Segmentation head: derive vessel probability from prediction
        seg_logits = pred_noise
        bce = F.binary_cross_entropy_with_logits(seg_logits, seg_target)

        # Dice loss
        prob = torch.sigmoid(seg_logits)
        smooth = 1.0
        intersection = (prob * seg_target).sum(dim=(1, 2, 3))
        cardinality = prob.sum(dim=(1, 2, 3)) + seg_target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + smooth) / (cardinality + smooth)
        dice_loss = 1.0 - dice.mean()

        # Combined Dice + BCE loss as in paper Table 1
        loss_seg = dice_loss + bce

        total_loss = loss_mse + loss_seg
        return total_loss, loss_mse, loss_seg

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Reverse diffusion sampling (DDIM-style) to obtain the vessel probability map.

        Args:
            x:     noisy input image [B, 1, H, W]
            steps: number of sampling steps (defaults to full timesteps)

        Returns:
            Vessel probability map in [0, 1]
        """
        steps = steps or self.timesteps
        x_t = torch.randn_like(x) * 0.1 + x  # slight noise to start
        timesteps = torch.linspace(self.timesteps - 1, 0, steps).long()

        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((x_t.size(0),), t_val.item(), device=x_t.device, dtype=torch.float)
            pred_noise = self.unet(x_t, t_batch)
            alpha_bar = self._to(self.alpha_bar[t_val])
            alpha_bar_prev = self._to(self.alpha_bar[t_val - 1] if t_val > 0 else self.alphas[0])

            # DDIM update
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            if i < len(timesteps) - 1:
                noise = torch.randn_like(x_t)
                x_t = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * noise
            else:
                x_t = x0_pred

        # Vessel probability map (segmentation head)
        prob = torch.sigmoid(x_t)
        return prob

    def generate_seg_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary vessel-region masks from input images."""
        prob = self.sample(x)
        return (prob > threshold).float()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def build_ddpm() -> DDPM:
    return DDPM(
        img_size=config.DDPM_IMG_SIZE,
        timesteps=config.DDPM_TIMESTEPS,
        beta_start=config.DDPM_BETA_START,
        beta_end=config.DDPM_BETA_END,
        base_channels=config.DDPM_BASE_CHANNELS,
    )


if __name__ == "__main__":
    model = build_ddpm()
    x = torch.randn(2, 1, 64, 64)
    t = torch.randint(0, config.DDPM_TIMESTEPS, (2,))
    target = (torch.rand(2, 1, 64, 64) > 0.7).float()
    loss, l_mse, l_seg = model.p_losses(x, t, target)
    print(f"DDPM loss: {loss.item():.4f} (mse={l_mse.item():.4f}, seg={l_seg.item():.4f})")
    prob = model.sample(x, steps=50)
    print("Sample output:", prob.shape, prob.min().item(), prob.max().item())