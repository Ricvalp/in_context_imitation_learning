"""In-context diffusion policy backed by a standard Transformer encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from equi_icil.models.diffusion_transformer import (
    DiffusionTransformer,
    DiffusionTransformerConfig,
)
from equi_icil.utils.normalization import LinearNormalizer


def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    activation=nn.SiLU,
) -> nn.Sequential:
    """Small helper to build a two-layer perceptron."""
    if in_dim <= 0:
        raise ValueError("in_dim must be positive for MLP construction")
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, out_dim),
    )


class SinusoidalTimeEmbedding(nn.Module):
    """Classic sinusoidal embedding for discrete indices."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("Time embedding dimension must be positive")
        self.dim = dim
        half_dim = dim // 2
        inv_freq = base ** (
            -torch.arange(half_dim, dtype=torch.float32) / max(1, half_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: Tensor of shape ``(B, N)`` with discrete indices.
        """
        if indices.ndim != 2:
            raise ValueError("indices must have shape (B, N)")
        values = indices.to(self.inv_freq.dtype)
        angles = values.unsqueeze(-1) * self.inv_freq
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            pad = torch.zeros(
                (*emb.shape[:-1], self.dim - emb.shape[-1]),
                dtype=emb.dtype,
                device=emb.device,
            )
            emb = torch.cat([emb, pad], dim=-1)
        return emb


@dataclass
class InContextDiffusionPolicyConfig:
    """Configuration for the Transformer-based in-context diffusion policy."""

    horizon: int
    n_obs_steps: int
    sample_points: int
    action_dim: int
    agent_dim: int
    use_point_colors: bool

    transformer_hidden_dim: int
    transformer_num_layers: int
    transformer_num_heads: int
    transformer_mlp_dim: int
    transformer_dropout: float
    transformer_attention_dropout: float
    transformer_activation: str
    transformer_norm_first: bool
    transformer_mask_action_to_obs: bool
    transformer_layer_norm_eps: float = 1e-5

    scalar_embedding_hidden_dim: int = 128
    time_embedding_base: float = 10000.0
    diffusion_embedding_base: float = 10000.0
    num_inference_steps: int = 50
    noise_scheduler_cfg: Dict[str, object] = None  # type: ignore[assignment]


class InContextDiffusionPolicy(nn.Module):
    """Diffusion policy that inlines actions into a standard Transformer encoder."""

    def __init__(self, cfg: InContextDiffusionPolicyConfig) -> None:
        super().__init__()
        if cfg.sample_points <= 0:
            raise ValueError("sample_points must be positive")
        if cfg.transformer_hidden_dim <= 0:
            raise ValueError("transformer_hidden_dim must be positive")

        self.cfg = cfg
        self.num_obs_nodes = cfg.n_obs_steps * cfg.sample_points
        self.num_proprio_nodes = cfg.n_obs_steps
        self.num_action_nodes = cfg.horizon
        self.total_nodes = (
            self.num_obs_nodes + self.num_proprio_nodes + self.num_action_nodes
        )

        transformer_cfg = DiffusionTransformerConfig(
            hidden_dim=cfg.transformer_hidden_dim,
            num_layers=cfg.transformer_num_layers,
            num_heads=cfg.transformer_num_heads,
            mlp_dim=cfg.transformer_mlp_dim,
            dropout=cfg.transformer_dropout,
            attention_dropout=cfg.transformer_attention_dropout,
            activation=cfg.transformer_activation,
            norm_first=cfg.transformer_norm_first,
            layer_norm_eps=cfg.transformer_layer_norm_eps,
        )
        self.transformer = DiffusionTransformer(transformer_cfg)

        point_input_dim = 3 + (3 if cfg.use_point_colors else 0)
        self.point_encoder = _make_mlp(
            point_input_dim, cfg.scalar_embedding_hidden_dim, cfg.transformer_hidden_dim
        )

        proprio_input_dim = (
            3  # xyz (normalised)
            + 4  # quaternion
            + 1  # gripper
        )
        self.proprio_encoder = _make_mlp(
            proprio_input_dim,
            cfg.scalar_embedding_hidden_dim,
            cfg.transformer_hidden_dim,
        )

        action_input_dim = proprio_input_dim
        self.action_encoder = _make_mlp(
            action_input_dim,
            cfg.scalar_embedding_hidden_dim,
            cfg.transformer_hidden_dim,
        )

        self.output_head = nn.Sequential(
            nn.Linear(cfg.transformer_hidden_dim, cfg.transformer_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.transformer_hidden_dim, cfg.action_dim),
        )

        self.world_time_embedder = SinusoidalTimeEmbedding(
            dim=cfg.transformer_hidden_dim,
            base=cfg.time_embedding_base,
        )
        self.diffusion_time_embedder = SinusoidalTimeEmbedding(
            dim=cfg.transformer_hidden_dim,
            base=cfg.diffusion_embedding_base,
        )
        self.diffusion_time_proj = nn.Sequential(
            nn.Linear(cfg.transformer_hidden_dim, cfg.transformer_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.transformer_hidden_dim, cfg.transformer_hidden_dim),
        )

        node_type_embeddings = torch.zeros(3, cfg.transformer_hidden_dim)
        self.node_type_embeddings = nn.Parameter(node_type_embeddings)

        scheduler_args = dict(cfg.noise_scheduler_cfg or {})
        self.scheduler = DDPMScheduler(**scheduler_args)
        self.num_inference_steps = cfg.num_inference_steps

        self.horizon = cfg.horizon
        self.action_dim = cfg.action_dim
        self.n_obs_steps = cfg.n_obs_steps
        self.sample_points = cfg.sample_points
        self.use_point_colors = cfg.use_point_colors

        self.normalizer = LinearNormalizer()
        self._normalizer_ready = False

        obs_time = torch.arange(self.n_obs_steps, dtype=torch.float32) - (
            self.n_obs_steps - 1
        )
        self.register_buffer("obs_time_indices", obs_time, persistent=False)
        action_time = torch.arange(1, self.horizon + 1, dtype=torch.float32)
        self.register_buffer("action_time_indices", action_time, persistent=False)

    # ------------------------------------------------------------------
    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer = copy.deepcopy(normalizer)
        for param in self.normalizer.parameters():
            param.requires_grad_(False)
        self._normalizer_ready = True

    # ------------------------------------------------------------------
    def _normalize_positions_with_pointcloud(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalise xyz positions using the point-cloud statistics."""
        point_norm = self.normalizer["point_clouds"]
        feature_dim = point_norm.scale.shape[0]
        padded = coords.new_zeros(*coords.shape[:-1], feature_dim)
        padded[..., :3] = coords
        normalized = point_norm.normalize(padded)
        return normalized[..., :3]

    def _encode_observation_nodes(
        self, point_clouds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = point_clouds.shape[0]
        device = point_clouds.device
        dtype = point_clouds.dtype

        pos = point_clouds[..., :3]
        if pos.shape[-2] != self.sample_points:
            raise ValueError(
                f"Expected {self.sample_points} points per observation, received {pos.shape[-2]}"
            )

        norm_pos = pos.reshape(bsz, -1, 3)
        if self.use_point_colors and point_clouds.shape[-1] >= 6:
            colors = point_clouds[..., 3:6].reshape(bsz, -1, 3)
            obs_input = torch.cat([norm_pos, colors], dim=-1)
        else:
            obs_input = norm_pos
        features = self.point_encoder(obs_input)

        obs_world_time = self.obs_time_indices.view(1, -1, 1).repeat(
            bsz, 1, self.sample_points
        )
        obs_world_time = obs_world_time.reshape(bsz, -1)
        return features, obs_world_time

    def _encode_proprio_nodes(
        self, agent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = agent_state.shape[0]
        features = agent_state
        proprio_feats = self.proprio_encoder(features)
        proprio_world_time = self.obs_time_indices.view(1, -1).repeat(bsz, 1)
        return proprio_feats, proprio_world_time

    def _encode_action_nodes(
        self, action_sample: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = action_sample.shape[0]
        features = action_sample
        action_feats = self.action_encoder(features)
        action_world_time = self.action_time_indices.view(1, -1).repeat(bsz, 1)
        return action_feats, action_world_time

    def _encode_nodes(
        self,
        point_clouds: torch.Tensor,
        agent_state: torch.Tensor,
        action_sample: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_feats, obs_time = self._encode_observation_nodes(point_clouds)
        proprio_feats, proprio_time = self._encode_proprio_nodes(agent_state)
        action_feats, action_time = self._encode_action_nodes(action_sample)

        tokens = torch.cat([obs_feats, proprio_feats, action_feats], dim=1)
        world_times = torch.cat([obs_time, proprio_time, action_time], dim=1)
        if tokens.shape[1] != self.total_nodes:
            raise RuntimeError("Mismatch in assembled node counts.")
        return tokens, world_times

    def _apply_type_and_time_embeddings(
        self,
        tokens: torch.Tensor,
        world_times: torch.Tensor,
    ) -> torch.Tensor:
        emb_obs, emb_proprio, emb_action = [
            emb.to(tokens.dtype) for emb in self.node_type_embeddings
        ]
        tokens[:, : self.num_obs_nodes] += emb_obs
        tokens[
            :, self.num_obs_nodes : self.num_obs_nodes + self.num_proprio_nodes
        ] += emb_proprio
        tokens[:, self.num_obs_nodes + self.num_proprio_nodes :] += emb_action

        time_emb = self.world_time_embedder(world_times).to(tokens.dtype)
        tokens = tokens + time_emb
        return tokens

    def _forward_transformer(
        self,
        point_clouds: torch.Tensor,
        agent_state: torch.Tensor,
        action_sample: torch.Tensor,
        diffusion_time: torch.Tensor,
    ) -> torch.Tensor:
        tokens, world_times = self._encode_nodes(
            point_clouds, agent_state, action_sample
        )
        tokens = self._apply_type_and_time_embeddings(tokens, world_times)

        diffusion_time_embed = self.diffusion_time_embedder(
            diffusion_time.unsqueeze(-1)
        ).to(tokens.dtype)
        diffusion_time_embed = diffusion_time_embed.squeeze(1)
        diffusion_cond = self.diffusion_time_proj(diffusion_time_embed)

        attn_mask = None
        if self.cfg.transformer_mask_action_to_obs:
            mask = torch.zeros(
                self.total_nodes,
                self.total_nodes,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            action_start = self.num_obs_nodes + self.num_proprio_nodes
            mask[:action_start, action_start:] = float("-inf")
            attn_mask = mask

        return self.transformer(
            tokens,
            diffusion_time_cond=diffusion_cond,
            attn_mask=attn_mask,
        )

    def _extract_action_features(self, transformer_out: torch.Tensor) -> torch.Tensor:
        start = self.num_obs_nodes + self.num_proprio_nodes
        end = start + self.num_action_nodes
        return transformer_out[:, start:end, :]

    # ------------------------------------------------------------------
    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self._normalizer_ready:
            raise RuntimeError("Normalizer must be set before training")

        point_clouds = batch["point_clouds"]
        agent_state = batch["agent_pos"]
        actions = batch["action"]

        actions_norm = self.normalizer["action"].normalize(actions)
        noise = torch.randn_like(actions_norm)
        bsz = actions_norm.shape[0]
        device = actions_norm.device

        point_clouds_norm = self.normalizer["point_clouds"].normalize(point_clouds)
        agent_state_norm = agent_state.clone()
        agent_state_norm[..., :3] = self._normalize_positions_with_pointcloud(
            agent_state[..., :3]
        )

        diffusion_time_indices = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (bsz,),
            device=device,
            dtype=torch.long,
        )
        noisy_actions = self.scheduler.add_noise(actions_norm, noise, diffusion_time_indices)
        noisy_actions_world = self.normalizer["action"].unnormalize(noisy_actions)
        action_nodes = noisy_actions_world.clone()
        action_nodes[..., :3] = self._normalize_positions_with_pointcloud(
            noisy_actions_world[..., :3]
        )

        transformer_out = self._forward_transformer(
            point_clouds_norm,
            agent_state_norm,
            action_nodes,
            diffusion_time=diffusion_time_indices,
        )
        action_feats = self._extract_action_features(transformer_out)
        pred_noise = self.output_head(action_feats)

        loss = F.mse_loss(pred_noise, noise)
        return loss, {"train_mse": float(loss.detach().cpu().item())}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        point_clouds: torch.Tensor,
        agent_state: torch.Tensor,
        *,
        point_sample_idx: Optional[torch.Tensor] = None,
        point_obs_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._normalizer_ready:
            raise RuntimeError("Normalizer must be set before sampling")
        if point_sample_idx is not None or point_obs_idx is not None:
            raise ValueError("Sparse point clouds are not supported by this policy.")

        device = point_clouds.device
        bsz = point_clouds.shape[0]

        trajectory = torch.randn(
            bsz,
            self.horizon,
            self.action_dim,
            device=device,
            dtype=point_clouds.dtype,
        )

        point_clouds_norm = self.normalizer["point_clouds"].normalize(point_clouds)
        agent_state_norm = agent_state.clone()
        agent_state_norm[..., :3] = self._normalize_positions_with_pointcloud(
            agent_state[..., :3]
        )

        self.scheduler.set_timesteps(self.num_inference_steps, device=device)

        for diffusion_time_step in self.scheduler.timesteps:
            diffusion_time = torch.full(
                (bsz,),
                diffusion_time_step,
                device=device,
                dtype=torch.long,
            )
            traj_world = self.normalizer["action"].unnormalize(trajectory)
            action_nodes = traj_world.clone()
            action_nodes[..., :3] = self._normalize_positions_with_pointcloud(
                traj_world[..., :3]
            )

            transformer_out = self._forward_transformer(
                point_clouds_norm,
                agent_state_norm,
                action_nodes,
                diffusion_time=diffusion_time,
            )
            action_feats = self._extract_action_features(transformer_out)
            noise_pred = self.output_head(action_feats)
            step_output = self.scheduler.step(noise_pred, diffusion_time_step, trajectory)
            trajectory = step_output.prev_sample

        return self.normalizer["action"].unnormalize(trajectory)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        has_normalizer = any(
            key.startswith("normalizer.params_dict") for key in state_dict
        )

        restored_normalizer = None
        restored_flag = self._normalizer_ready
        if not has_normalizer and self._normalizer_ready:
            restored_normalizer = self.normalizer
            self.normalizer = LinearNormalizer()
            self._normalizer_ready = False

        result = super().load_state_dict(state_dict, strict=False)

        if has_normalizer:
            self._normalizer_ready = True
        elif restored_normalizer is not None:
            self.normalizer = restored_normalizer
            self._normalizer_ready = restored_flag

        missing_keys = list(result.missing_keys)
        if not has_normalizer:
            missing_keys = [
                key
                for key in missing_keys
                if not key.startswith("normalizer.params_dict")
            ]

        unexpected_keys = list(result.unexpected_keys)

        if strict and (missing_keys or unexpected_keys):
            error_lines = []
            if missing_keys:
                error_lines.append(
                    "Missing key(s) in state_dict: " + ", ".join(missing_keys) + "."
                )
            if unexpected_keys:
                error_lines.append(
                    "Unexpected key(s) in state_dict: "
                    + ", ".join(unexpected_keys)
                    + "."
                )
            error_msg = "\n\t".join(error_lines)
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t{error_msg}"
            )

        return type(result)(missing_keys, unexpected_keys)


__all__ = [
    "InContextDiffusionPolicy",
    "InContextDiffusionPolicyConfig",
]
