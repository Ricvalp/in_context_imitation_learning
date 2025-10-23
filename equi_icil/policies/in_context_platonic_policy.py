"""In-context diffusion policy that folds actions into the Platonic Transformer input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from equi_icil.models.platonic_transformer import PlatonicTransformer
from equi_icil.utils.normalization import LinearNormalizer


def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    activation=nn.SiLU,
) -> nn.Sequential:
    """Small helper to build a two-layer perceptron."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, out_dim),
    )


class SinusoidalTimeEmbedding(nn.Module):
    """Classic sinusoidal embedding for discrete world-time indices."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("Time embedding dimension must be positive")
        self.dim = dim
        half_dim = dim // 2
        inv_freq = base ** (-torch.arange(half_dim, dtype=torch.float32) / max(1, half_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, world_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            world_time: Tensor of shape ``(B, N)`` with discrete indices.
        """
        if world_time.ndim != 2:
            raise ValueError("world_time must have shape (B, N)")
        time = world_time.to(self.inv_freq.dtype)
        angles = time.unsqueeze(-1) * self.inv_freq
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            pad = torch.zeros(
                (*emb.shape[:-1], self.dim - emb.shape[-1]),
                dtype=emb.dtype,
                device=emb.device,
            )
            emb = torch.cat([emb, pad], dim=-1)
        return emb


def _quaternion_to_direction(quat: torch.Tensor) -> torch.Tensor:
    """Return the rotated +Z axis for quaternions stored as (x, y, z, w)."""
    if quat.shape[-1] != 4:
        raise ValueError("Quaternion tensor must have 4 components in the last dim")
    x, y, z, w = quat.unbind(dim=-1)
    norm = torch.sqrt(x * x + y * y + z * z + w * w + 1e-12)
    x = x / norm
    y = y / norm
    z = z / norm
    w = w / norm

    dir_x = 2.0 * (x * z + y * w)
    dir_y = 2.0 * (y * z - x * w)
    dir_z = 1.0 - 2.0 * (x * x + y * y)
    return torch.stack([dir_x, dir_y, dir_z], dim=-1)


@dataclass
class InContextPlatonicPolicyConfig:
    """Configuration bundle for the in-context Platonic diffusion policy."""

    horizon: int
    n_obs_steps: int
    sample_points: int
    action_dim: int
    agent_dim: int
    use_point_colors: bool

    transformer_input_scalar_dim: int
    transformer_input_vector_dim: int
    transformer_hidden_dim: int
    transformer_output_dim: int
    transformer_num_layers: int
    transformer_num_heads: int
    transformer_solid: str
    transformer_dropout: float
    transformer_drop_path_rate: float
    transformer_ffn_dim_factor: int
    transformer_rope_sigma: Optional[float]
    transformer_ape_sigma: Optional[float]
    transformer_learned_freqs: bool
    transformer_attention: bool
    transformer_use_key: bool
    transformer_mean_aggregation: bool
    transformer_ffn_readout: bool
    transformer_norm_first: bool
    transformer_layer_scale_init_value: Optional[float]
    transformer_use_cls_token: bool
    transformer_dense_mode: bool
    transformer_mask_action_to_obs: bool

    time_embedding_base: float
    scalar_embedding_hidden_dim: int
    num_inference_steps: int
    noise_scheduler_cfg: Dict[str, object]


class InContextPlatonicDiffusionPolicy(nn.Module):
    """Diffusion policy that inlines actions and proprioception into the Platonic Transformer."""

    def __init__(self, cfg: InContextPlatonicPolicyConfig) -> None:
        super().__init__()
        if not cfg.transformer_dense_mode:
            raise ValueError("In-context policy currently supports dense mode only.")
        if cfg.transformer_input_vector_dim != 1:
            raise ValueError(
                "Expected a single vector channel (orientation) for transformer input."
            )

        self.cfg = cfg
        self.num_obs_nodes = cfg.n_obs_steps * cfg.sample_points
        self.num_proprio_nodes = cfg.n_obs_steps
        self.num_action_nodes = cfg.horizon
        self.total_nodes = (
            self.num_obs_nodes + self.num_proprio_nodes + self.num_action_nodes
        )
        self.vector_output_channels = 2

        transformer = PlatonicTransformer(
            input_dim=cfg.transformer_input_scalar_dim,
            input_dim_vec=cfg.transformer_input_vector_dim,
            hidden_dim=cfg.transformer_hidden_dim,
            output_dim=cfg.transformer_output_dim,
            output_dim_vec=self.vector_output_channels,
            nhead=cfg.transformer_num_heads,
            num_layers=cfg.transformer_num_layers,
            solid_name=cfg.transformer_solid,
            spatial_dim=3,
            dense_mode=True,
            scalar_task_level="node",
            vector_task_level="node",
            ffn_readout=cfg.transformer_ffn_readout,
            mean_aggregation=cfg.transformer_mean_aggregation,
            dropout=cfg.transformer_dropout,
            norm_first=cfg.transformer_norm_first,
            drop_path_rate=cfg.transformer_drop_path_rate,
            layer_scale_init_value=cfg.transformer_layer_scale_init_value,
            attention=cfg.transformer_attention,
            ffn_dim_factor=cfg.transformer_ffn_dim_factor,
            rope_sigma=cfg.transformer_rope_sigma,
            ape_sigma=cfg.transformer_ape_sigma,
            learned_freqs=cfg.transformer_learned_freqs,
            use_key=cfg.transformer_use_key,
            use_cls_token=cfg.transformer_use_cls_token,
            time_conditioning=True,
            class_conditioning=False,
        )
        self.transformer = transformer

        scalar_dim = cfg.transformer_input_scalar_dim
        hidden_dim = cfg.scalar_embedding_hidden_dim

        point_input_dim = 3 if cfg.use_point_colors else 0
        self.point_scalar_encoder = (
            _make_mlp(point_input_dim, hidden_dim, scalar_dim)
            if point_input_dim > 0
            else None
        )
        self.proprio_scalar_encoder = _make_mlp(5, hidden_dim, scalar_dim)
        self.action_scalar_encoder = _make_mlp(5, hidden_dim, scalar_dim)

        self.world_time_embedder = SinusoidalTimeEmbedding(
            dim=scalar_dim, base=cfg.time_embedding_base
        )

        node_type_embeddings = torch.zeros(3, scalar_dim)
        self.node_type_embeddings = nn.Parameter(node_type_embeddings)

        scheduler_args = dict(cfg.noise_scheduler_cfg)
        self.scheduler = DDPMScheduler(**scheduler_args)
        self.num_inference_steps = cfg.num_inference_steps

        self.horizon = cfg.horizon
        self.action_dim = cfg.action_dim
        self.n_obs_steps = cfg.n_obs_steps
        self.sample_points = cfg.sample_points
        self.use_point_colors = cfg.use_point_colors

        self.position_dim = 3
        self.orientation_vec_dim = 3
        self.scalar_output_dim = self.action_dim - (
            self.position_dim + self.orientation_vec_dim
        )
        if self.scalar_output_dim < 0:
            raise ValueError(
                "action_dim must be at least 6 to accommodate position and orientation vectors"
            )

        if self.scalar_output_dim > 0:
            self.scalar_output_head = nn.Sequential(
                nn.Linear(cfg.transformer_output_dim, cfg.transformer_output_dim),
                nn.SiLU(),
                nn.Linear(cfg.transformer_output_dim, self.scalar_output_dim),
            )
        else:
            self.scalar_output_head = None

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
    def _split_action(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tensor.shape[-1] < 8:
            raise ValueError("Action tensor must have at least 8 dimensions (xyz + quat + gripper)")
        pos = tensor[..., :3]
        quat = tensor[..., 3:7]
        grip = tensor[..., 7:8]
        return pos, quat, grip

    def _encode_nodes(
        self,
        point_clouds: torch.Tensor,
        agent_state: torch.Tensor,
        action_sample: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (positions, scalars, vectors, time_indices)."""
        bsz = point_clouds.shape[0]
        device = point_clouds.device
        dtype = point_clouds.dtype

        # Observation nodes --------------------------------------------------
        pc_pos = point_clouds[..., :3]
        if pc_pos.shape[-2] != self.sample_points:
            raise ValueError(
                f"Expected {self.sample_points} points per observation, received {pc_pos.shape[-2]}"
            )
        obs_pos = pc_pos.reshape(bsz, -1, 3)  # (B, To * N, 3)

        if self.use_point_colors and point_clouds.shape[-1] >= 6:
            colors = point_clouds[..., 3:6].reshape(bsz, -1, 3)
            obs_scalar = self.point_scalar_encoder(colors)  # type: ignore[arg-type]  # (B, To * N, D_s)
        else:
            obs_scalar = torch.zeros(
                bsz,
                self.num_obs_nodes,
                self.cfg.transformer_input_scalar_dim,
                device=device,
                dtype=dtype,
            )
        obs_vector = torch.zeros(
            bsz,
            self.num_obs_nodes,
            self.cfg.transformer_input_vector_dim,
            3,
            device=device,
            dtype=dtype,
        )  # (B, To * N, V, 3)
        obs_world_time = self.obs_time_indices.view(1, -1, 1).repeat(
            bsz, 1, self.sample_points
        ).reshape(bsz, -1)  # (B, To * N)

        # Proprio nodes ------------------------------------------------------
        agent_pos, agent_quat, agent_grip = self._split_action(agent_state)
        proprio_pos = agent_pos  # (B, To, 3)
        direction = _quaternion_to_direction(agent_quat)
        proprio_vector = direction.unsqueeze(-2)  # (B, To, 1, 3)
        proprio_scalar_input = torch.cat([agent_quat, agent_grip], dim=-1)
        proprio_scalar = self.proprio_scalar_encoder(proprio_scalar_input)  # (B, To, D_s)
        proprio_world_time = self.obs_time_indices.view(1, -1).repeat(bsz, 1)  # (B, To)

        # Action nodes -------------------------------------------------------
        act_pos, act_quat, act_grip = self._split_action(action_sample)
        action_dir = _quaternion_to_direction(act_quat)
        action_vector = action_dir.unsqueeze(-2)  # (B, H, 1, 3)
        action_scalar_input = torch.cat([act_quat, act_grip], dim=-1)
        action_scalar = self.action_scalar_encoder(action_scalar_input)  # (B, H, D_s)
        action_world_time = self.action_time_indices.view(1, -1).repeat(bsz, 1)  # (B, H)

        action_pos = act_pos

        # Stack everything ---------------------------------------------------
        positions = torch.cat(
            [obs_pos, proprio_pos, action_pos],
            dim=1,
        )  # (B, N_total, 3)
        scalars = torch.cat(
            [obs_scalar, proprio_scalar, action_scalar],
            dim=1,
        )  # (B, N_total, D_s)
        vectors = torch.cat(
            [obs_vector, proprio_vector, action_vector],
            dim=1,
        )  # (B, N_total, V, 3)
        world_times = torch.cat(
            [obs_world_time, proprio_world_time, action_world_time],
            dim=1,
        )  # (B, N_total)

        if positions.shape[1] != self.total_nodes:
            raise RuntimeError("Mismatch in assembled node counts.")
        return positions, scalars, vectors, world_times

    def _build_attention_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create a pairwise mask blocking flows from action nodes to obs/proprio nodes."""
        mask = torch.zeros(
            batch_size,
            self.total_nodes,
            self.total_nodes,
            dtype=torch.bool,
            device=device,
        )
        action_start = self.num_obs_nodes + self.num_proprio_nodes
        if action_start > 0:
            mask[:, :action_start, action_start:] = True  # prevent information leak from actions
        return mask

    def _normalize_positions_with_pointcloud(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalise xyz positions using the point-cloud statistics.

        Args:
            coords: Tensor containing positional channels ``(..., 3)``.

        Returns:
            Tensor with the same shape, scaled like the point-cloud coordinates.
        """
        point_norm = self.normalizer["point_clouds"]
        feature_dim = point_norm.scale.shape[0]
        padded = coords.new_zeros(*coords.shape[:-1], feature_dim)
        padded[..., :3] = coords
        normalized = point_norm.normalize(padded)
        return normalized[..., :3]

    def _apply_type_and_time_embeddings(
        self,
        scalars: torch.Tensor,
        world_times: torch.Tensor,
    ) -> torch.Tensor:
        obs_embed, proprio_embed, action_embed = [
            emb.to(scalars.dtype) for emb in self.node_type_embeddings
        ]
        scalars[:, : self.num_obs_nodes] += obs_embed
        scalars[
            :, self.num_obs_nodes : self.num_obs_nodes + self.num_proprio_nodes
        ] += proprio_embed
        scalars[:, self.num_obs_nodes + self.num_proprio_nodes :] += action_embed
        time_emb = self.world_time_embedder(world_times).to(scalars.dtype)
        scalars = scalars + time_emb
        return scalars

    def _forward_transformer(
        self,
        point_clouds: torch.Tensor,
        agent_state: torch.Tensor,
        action_sample: torch.Tensor,
        diffusion_timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos, scalars, vectors, world_times = self._encode_nodes(
            point_clouds, agent_state, action_sample
        )
        scalars = self._apply_type_and_time_embeddings(scalars, world_times)
        bsz = point_clouds.shape[0]

        if self.cfg.transformer_mask_action_to_obs:
            attn_mask = self._build_attention_mask(bsz, point_clouds.device)
        else:
            attn_mask = None

        scalar_out, vector_out = self.transformer(
            x=scalars,
            pos=pos,
            batch=None,
            vec=vectors,
            mask=None,
            time_conditioning=diffusion_timesteps,
            class_conditioning=None,
            avg_num_nodes=float(self.total_nodes),
            attention_mask=attn_mask,
        )
        return scalar_out, vector_out

    def _extract_action_scalar_features(
        self, transformer_out: torch.Tensor
    ) -> torch.Tensor:
        start = self.num_obs_nodes + self.num_proprio_nodes
        end = start + self.num_action_nodes
        action_feats = transformer_out[:, start:end, :]
        return action_feats

    def _extract_action_vector_features(
        self, transformer_vectors: torch.Tensor
    ) -> torch.Tensor:
        start = self.num_obs_nodes + self.num_proprio_nodes
        end = start + self.num_action_nodes
        return transformer_vectors[:, start:end, :, :]

    # ------------------------------------------------------------------
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self._normalizer_ready:
            raise RuntimeError("Normalizer must be set before training")

        point_clouds = batch["point_clouds"]  # (B, To, N, C)
        agent_state = batch["agent_pos"]  # (B, To, D)
        actions = batch["action"]

        actions_norm = self.normalizer["action"].normalize(actions)
        noise = torch.randn_like(actions_norm)
        bsz = actions_norm.shape[0]
        device = actions_norm.device

        # Share the point-cloud normaliser across all xyz channels so the transformer
        # sees positions on a common scale regardless of their source.
        point_clouds_norm = self.normalizer["point_clouds"].normalize(point_clouds)
        agent_state_norm = agent_state.clone()
        agent_state_norm[..., :3] = self._normalize_positions_with_pointcloud(
            agent_state[..., :3]
        )

        diffusion_timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (bsz,),
            device=device,
            dtype=torch.long,
        )
        noisy_actions = self.scheduler.add_noise(actions_norm, noise, diffusion_timesteps)
        noisy_actions_world = self.normalizer["action"].unnormalize(noisy_actions)

        action_nodes = noisy_actions_world.clone()
        action_nodes[..., :3] = self._normalize_positions_with_pointcloud(
            noisy_actions_world[..., :3]
        )

        scalar_out, vector_out = self._forward_transformer(
            point_clouds_norm,
            agent_state_norm,
            action_nodes,
            diffusion_timesteps=diffusion_timesteps,
        )
        action_scalar_feats = self._extract_action_scalar_features(scalar_out)
        action_vector_feats = self._extract_action_vector_features(vector_out)

        position_noise = action_vector_feats[..., 0, :]
        orientation_vec_noise = action_vector_feats[..., 1, :]

        if self.scalar_output_dim > 0:
            scalar_noise = self.scalar_output_head(action_scalar_feats)
        else:
            scalar_noise = position_noise.new_zeros(bsz, self.horizon, 0)

        pred_noise = torch.cat(
            [position_noise, orientation_vec_noise, scalar_noise], dim=-1
        )

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

        for t in self.scheduler.timesteps:
            diffusion_timesteps = torch.full((bsz,), t, device=device, dtype=torch.long)
            traj_world = self.normalizer["action"].unnormalize(trajectory)
            action_nodes = traj_world.clone()
            action_nodes[..., :3] = self._normalize_positions_with_pointcloud(
                traj_world[..., :3]
            )
            scalar_out, vector_out = self._forward_transformer(
                point_clouds_norm,
                agent_state_norm,
                action_nodes,
                diffusion_timesteps=diffusion_timesteps,
            )
            action_scalar_feats = self._extract_action_scalar_features(scalar_out)
            action_vector_feats = self._extract_action_vector_features(vector_out)

            position_noise = action_vector_feats[..., 0, :]
            orientation_vec_noise = action_vector_feats[..., 1, :]

            if self.scalar_output_dim > 0:
                scalar_noise = self.scalar_output_head(action_scalar_feats)
            else:
                scalar_noise = position_noise.new_zeros(bsz, self.horizon, 0)

            noise_pred = torch.cat(
                [position_noise, orientation_vec_noise, scalar_noise], dim=-1
            )
            step_output = self.scheduler.step(noise_pred, t, trajectory)
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
    "InContextPlatonicDiffusionPolicy",
    "InContextPlatonicPolicyConfig",
]
