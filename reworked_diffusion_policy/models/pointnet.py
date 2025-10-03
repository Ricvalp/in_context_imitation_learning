"""Simple PointNet-style encoder for point-cloud observations."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(sizes: Sequence[int], *, activation=nn.ReLU, layer_norm: bool = False) -> nn.Sequential:
    layers = []
    for idx in range(len(sizes) - 1):
        in_dim, out_dim = sizes[idx], sizes[idx + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        if idx < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class PointNetEncoder(nn.Module):
    """Applies a shared MLP and max-pooling over point sets."""

    def __init__(
        self,
        in_channels: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        *,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        mlp_sizes = [in_channels, *hidden_dims, out_dim]
        layers = []
        for idx in range(len(mlp_sizes) - 1):
            layers.append(nn.Linear(mlp_sizes[idx], mlp_sizes[idx + 1]))
            if use_layernorm:
                layers.append(nn.LayerNorm(mlp_sizes[idx + 1]))
            if idx < len(mlp_sizes) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        b, n, c = x.shape
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0]
        return x


class ObservationEncoder(nn.Module):
    """Encodes multi-frame point clouds plus low-dimensional agent state."""

    def __init__(
        self,
        *,
        pointnet: PointNetEncoder,
        state_dims: Sequence[int],
        n_obs_steps: int,
    ) -> None:
        super().__init__()
        self.pointnet = pointnet
        self.n_obs_steps = n_obs_steps
        if len(state_dims) < 2:
            raise ValueError("state_dims should contain at least input and output size")
        self.state_mlp = _make_mlp(state_dims, activation=nn.ReLU, layer_norm=False)
        self.state_out_dim = state_dims[-1]

    def forward(self, point_clouds: torch.Tensor, agent_pos: torch.Tensor) -> torch.Tensor:
        # point_clouds: (B, To, N, C)
        b, tobs, npts, feat = point_clouds.shape
        if tobs != self.n_obs_steps:
            raise ValueError(f"Expected {self.n_obs_steps} obs steps, got {tobs}")
        clouds = point_clouds.reshape(b * tobs, npts, feat)
        point_feats = self.pointnet(clouds)

        if agent_pos.ndim == 2:
            agent_pos = agent_pos.unsqueeze(1).expand(-1, tobs, -1)
        if agent_pos.shape[0] != b or agent_pos.shape[1] != tobs:
            raise ValueError(
                f"agent_pos should have shape (B, {tobs}, D), got {tuple(agent_pos.shape)}"
            )

        state_flat = agent_pos.reshape(b * tobs, -1)
        state_feats = self.state_mlp(state_flat)

        per_frame = torch.cat([point_feats, state_feats], dim=-1)
        per_frame = per_frame.reshape(b, tobs, -1)
        return per_frame.reshape(b, -1)


__all__ = ["PointNetEncoder", "ObservationEncoder"]
