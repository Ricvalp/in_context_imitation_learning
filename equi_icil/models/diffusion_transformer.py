"""Diffusion Transformer with AdaLN-Zero conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer-norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@dataclass
class DiffusionTransformerConfig:
    """Configuration describing the Transformer backbone."""

    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    dropout: float
    attention_dropout: float
    activation: str
    norm_first: bool
    layer_norm_eps: float = 1e-6


class _AdaLayerNormZero(nn.Module):
    """LayerNorm with AdaLN-Zero modulation."""

    def __init__(self, hidden_dim: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.proj(cond).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class DiffusionTransformerBlock(nn.Module):
    """Transformer encoder block equipped with AdaLN-Zero conditioning."""

    def __init__(self, cfg: DiffusionTransformerConfig) -> None:
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        if not cfg.norm_first:
            raise ValueError("DiffusionTransformerBlock currently requires norm_first=True.")
        self.attn_norm = nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False, eps=cfg.layer_norm_eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.attention_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(cfg.dropout)

        activation = cfg.activation.lower()
        if activation == "gelu":
            act_layer = nn.GELU()
        elif activation in ("silu", "swish"):
            act_layer = nn.SiLU()
        elif activation == "relu":
            act_layer = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation '{cfg.activation}'")

        self.mlp_norm = nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False, eps=cfg.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.mlp_dim),
            act_layer,
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_dim, cfg.hidden_dim),
        )
        self.mlp_dropout = nn.Dropout(cfg.dropout)

        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 6 * cfg.hidden_dim, bias=True),
        )
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def _apply_attention(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return self.attn_dropout(attn_out)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        cond: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(cond).chunk(6, dim=-1)

        attn_in = modulate(self.attn_norm(tokens), shift_msa, scale_msa)
        attn_out = self._apply_attention(attn_in, attn_mask=attn_mask)
        tokens = tokens + gate_msa.unsqueeze(1) * attn_out

        mlp_in = modulate(self.mlp_norm(tokens), shift_mlp, scale_mlp)
        mlp_out = self.mlp_dropout(self.mlp(mlp_in))
        tokens = tokens + gate_mlp.unsqueeze(1) * mlp_out
        return tokens


class DiffusionTransformer(nn.Module):
    """Transformer encoder with AdaLN-Zero conditioning on diffusion time."""

    def __init__(self, cfg: DiffusionTransformerConfig) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads "
                f"(got hidden_dim={cfg.hidden_dim}, num_heads={cfg.num_heads})"
            )
        if cfg.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if cfg.mlp_dim <= 0:
            raise ValueError("mlp_dim must be positive")

        self.cfg = cfg
        self.blocks = nn.ModuleList(
            DiffusionTransformerBlock(cfg) for _ in range(cfg.num_layers)
        )
        self.final_layer = _AdaLayerNormZero(cfg.hidden_dim, cfg.layer_norm_eps)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _prepare_attention_mask(
        mask: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dim() != 2:
            raise ValueError("Attention mask must have shape (L, L)")
        if mask.dtype == torch.bool:
            mask = mask.to(dtype=dtype)
            mask.masked_fill_(mask > 0, float("-inf"))
        else:
            mask = mask.to(device=device, dtype=dtype)
        return mask

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        diffusion_time_cond: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode ``tokens`` conditioned on ``diffusion_time_cond``.

        Args:
            tokens: Input sequence ``(B, N, D)``.
            diffusion_time_cond: Conditioning vector ``(B, D)`` derived from the diffusion time.
            attn_mask: Optional mask ``(N, N)`` with ``-inf`` on disallowed entries.

        Returns:
            Tensor ``(B, N, D)`` with encoded representations.
        """
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape (B, N, D)")
        if diffusion_time_cond.ndim != 2 or diffusion_time_cond.shape[-1] != tokens.shape[-1]:
            raise ValueError("diffusion_time_cond must have shape (B, D) matching tokens")

        prepared_mask = self._prepare_attention_mask(
            attn_mask,
            device=tokens.device,
            dtype=tokens.dtype,
        )

        x = tokens
        for block in self.blocks:
            x = block(
                x,
                cond=diffusion_time_cond,
                attn_mask=prepared_mask,
            )

        return self.final_layer(x, diffusion_time_cond)


__all__ = ["DiffusionTransformer", "DiffusionTransformerConfig"]
