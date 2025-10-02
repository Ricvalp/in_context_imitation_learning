"""Conditional 1D U-Net used for diffusion denoising."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float32, device=device)
        exponent = exponent / (half_dim - 1)
        embeddings = torch.exp(exponent * torch.log(torch.tensor(10000.0)))
        angles = timesteps.float().unsqueeze(1) * embeddings.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        *,
        kernel_size: int = 3,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        cond_term = self.cond_proj(cond).unsqueeze(-1)
        h = h + cond_term
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        next_channels: int,
        cond_dim: int,
        *,
        kernel_size: int = 3,
        num_groups: int = 8,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    cond_dim,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                )
                for i in range(2)
            ]
        )
        self.downsample = None
        self.downsample_out_channels = out_channels
        if downsample:
            self.downsample_out_channels = next_channels
            self.downsample = nn.Conv1d(out_channels, next_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        for block in self.res_blocks:
            x = block(x, cond)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        cond_dim: int,
        *,
        kernel_size: int = 3,
        num_groups: int = 8,
        upsample: bool = True,
        upsample_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.upsample = None
        self.upsampled_channels = in_channels
        if upsample:
            target_channels = upsample_channels or in_channels
            self.upsample = nn.ConvTranspose1d(in_channels, target_channels, 4, stride=2, padding=1)
            self.upsampled_channels = target_channels

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    (self.upsampled_channels + skip_channels) if i == 0 else out_channels,
                    out_channels,
                    cond_dim,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                )
                for i in range(2)
            ]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                x = F.pad(x, (0, diff))
            elif diff < 0:
                x = x[..., : skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        for block in self.res_blocks:
            x = block(x, cond)
        return x


class ConditionalUNet1D(nn.Module):
    """UNet architecture that conditions on timestep and a global vector."""

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        *,
        hidden_dims: Sequence[int],
        kernel_size: int,
        num_groups: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dims[0], kernel_size=1)

        time_dim = hidden_dims[0]
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.global_embedding = nn.Sequential(
            nn.Linear(global_cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        channels = list(hidden_dims)
        self.down_blocks = nn.ModuleList()
        current_channels = channels[0]
        skip_channels: List[int] = []
        for idx, out_channels in enumerate(channels):
            downsample = idx < len(channels) - 1
            next_channels = channels[idx + 1] if downsample else out_channels
            block = DownBlock(
                current_channels,
                out_channels,
                next_channels,
                time_dim,
                kernel_size=kernel_size,
                num_groups=num_groups,
                downsample=downsample,
            )
            self.down_blocks.append(block)
            skip_channels.append(out_channels)
            current_channels = next_channels

        self.mid_block1 = ResidualBlock(
            current_channels,
            current_channels,
            time_dim,
            kernel_size=kernel_size,
            num_groups=num_groups,
        )
        self.mid_block2 = ResidualBlock(
            current_channels,
            current_channels,
            time_dim,
            kernel_size=kernel_size,
            num_groups=num_groups,
        )

        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        current_up_channels = current_channels
        for idx, out_channels in enumerate(reversed_channels):
            skip_dim = skip_channels[-(idx + 1)]
            upsample = idx < len(reversed_channels) - 1
            next_channels = reversed_channels[idx + 1] if upsample else out_channels
            block = UpBlock(
                current_up_channels,
                out_channels,
                skip_dim,
                time_dim,
                kernel_size=kernel_size,
                num_groups=num_groups,
                upsample=upsample,
                upsample_channels=next_channels if upsample else None,
            )
            self.up_blocks.append(block)
            current_up_channels = out_channels

        self.output_norm = nn.GroupNorm(num_groups, hidden_dims[0])
        self.output_proj = nn.Conv1d(hidden_dims[0], input_dim, kernel_size=1)

    def forward(self, sample: torch.Tensor, timesteps: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        # sample: (B, T, C)
        x = sample.permute(0, 2, 1)
        x = self.input_proj(x)

        time_emb = self.time_embedding(timesteps)
        global_emb = self.global_embedding(global_cond)
        cond = time_emb + global_emb

        skips: List[torch.Tensor] = []
        h = x
        for block in self.down_blocks:
            h, skip = block(h, cond)
            skips.append(skip)

        h = self.mid_block1(h, cond)
        h = self.mid_block2(h, cond)

        for block in self.up_blocks:
            skip = skips.pop()
            h = block(h, skip, cond)

        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_proj(h)
        return h.permute(0, 2, 1)


__all__ = ["ConditionalUNet1D"]
