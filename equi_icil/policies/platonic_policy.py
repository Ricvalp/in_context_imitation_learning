"""Diffusion policy variant that uses the Platonic Transformer as the encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from equi_icil.models.platonic_transformer import PlatonicTransformer
from equi_icil.models.unet1d import ConditionalUNet1D
from equi_icil.utils.normalization import LinearNormalizer


def _make_mlp(
    sizes: Sequence[int], activation=nn.ReLU, layer_norm: bool = False
) -> nn.Sequential:
    """Create an MLP with optional layer norm between linear layers."""
    layers: list[nn.Module] = []
    for idx in range(len(sizes) - 1):
        in_dim, out_dim = sizes[idx], sizes[idx + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        if idx < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class PlatonicObservationEncoder(nn.Module):
    """Encodes point clouds with a Platonic Transformer plus a state MLP."""

    def __init__(
        self,
        *,
        transformer: PlatonicTransformer,
        state_dims: Sequence[int],
        n_obs_steps: int,
        scalar_feature_dim: int,
    ) -> None:
        super().__init__()
        if len(state_dims) < 2:
            raise ValueError("state_dims must provide at least input and output sizes")

        self.transformer = transformer
        self.state_mlp = _make_mlp(state_dims, activation=nn.ReLU, layer_norm=False)
        self.state_out_dim = state_dims[-1]
        self.n_obs_steps = n_obs_steps
        self.scalar_feature_dim = scalar_feature_dim

    def forward(
        self,
        point_clouds: torch.Tensor,
        agent_pos: torch.Tensor,
        *,
        point_sample_idx: Optional[torch.Tensor] = None,
        point_obs_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode stacked observations of point clouds and agent state."""
        if self.transformer.dense_mode:
            if point_clouds.ndim != 4:
                raise ValueError(
                    "Dense mode expects point_clouds shaped (B, To, N, C); "
                    f"received tensor with shape {tuple(point_clouds.shape)}"
                )
            b, tobs, npts, feat = point_clouds.shape
            if tobs != self.n_obs_steps:
                raise ValueError(
                    f"Expected {self.n_obs_steps} observation steps, received {tobs}"
                )
            if feat < 3:
                raise ValueError(
                    "Point clouds must contain xyz coordinates in the first three channels"
                )

            pos = point_clouds[..., :3]
            if self.scalar_feature_dim > 0:
                scalars = point_clouds[..., 3 : 3 + self.scalar_feature_dim]
                if scalars.shape[-1] != self.scalar_feature_dim:
                    raise ValueError(
                        f"Expected {self.scalar_feature_dim} scalar point features, received {scalars.shape[-1]}"
                    )
            else:
                scalars = point_clouds.new_zeros(b, tobs, npts, 0)

            clouds_flat = scalars.reshape(b * tobs, npts, self.scalar_feature_dim)
            pos_flat = pos.reshape(b * tobs, npts, 3)
            vec_flat = pos_flat.unsqueeze(-2)  # (B*To, N, 1, 3)

            transformer_out, _ = self.transformer(
                x=clouds_flat,
                pos=pos_flat,
                batch=None,
                vec=vec_flat,
                mask=None,
                time_conditioning=None,
                class_conditioning=None,
                avg_num_nodes=float(npts),
            )
        else:
            if point_clouds.ndim != 2:
                raise ValueError(
                    "Sparse mode expects concatenated point_clouds shaped (total_points, C); "
                    f"received tensor with shape {tuple(point_clouds.shape)}"
                )
            if point_sample_idx is None or point_obs_idx is None:
                raise ValueError(
                    "Sparse mode requires point_sample_idx and point_obs_idx tensors"
                )

            total_points, feat = point_clouds.shape
            if feat < 3:
                raise ValueError(
                    "Sparse point clouds must contain xyz coordinates in the first three channels"
                )

            b = agent_pos.shape[0]
            tobs = self.n_obs_steps
            expected_frames = b * tobs

            pos = point_clouds[:, :3]
            if self.scalar_feature_dim > 0:
                scalars = point_clouds[:, 3 : 3 + self.scalar_feature_dim]
                if scalars.shape[-1] != self.scalar_feature_dim:
                    raise ValueError(
                        f"Expected {self.scalar_feature_dim} scalar point features, received {scalars.shape[-1]}"
                    )
            else:
                scalars = point_clouds.new_zeros(
                    total_points,
                    0,
                    device=point_clouds.device,
                    dtype=point_clouds.dtype,
                )

            frame_ids = point_sample_idx * tobs + point_obs_idx
            max_frame_id = int(frame_ids.max().item()) if frame_ids.numel() > 0 else -1
            if max_frame_id >= expected_frames:
                raise ValueError(
                    f"Frame index {max_frame_id} exceeds expected frames ({expected_frames}) "
                    "computed from batch size and observation steps."
                )

            counts = torch.bincount(frame_ids, minlength=expected_frames).clamp(min=1)
            avg_nodes = float(counts.float().mean().item())

            vec = pos.unsqueeze(-2)  # (N, 1, 3)

            transformer_out, _ = self.transformer(
                x=scalars,
                pos=pos,
                batch=frame_ids,
                vec=vec,
                mask=None,
                time_conditioning=None,
                class_conditioning=None,
                avg_num_nodes=avg_nodes,
            )

        if agent_pos.ndim == 2:
            agent_pos = agent_pos.unsqueeze(1).expand(-1, self.n_obs_steps, -1)
        if agent_pos.shape[0] * agent_pos.shape[1] != transformer_out.shape[0]:
            raise ValueError(
                "Mismatch between transformer outputs and agent_pos size: "
                f"{transformer_out.shape[0]} vs {agent_pos.shape}"
            )

        state_flat = agent_pos.reshape(-1, agent_pos.shape[-1])
        state_feats = self.state_mlp(state_flat)
        per_frame = torch.cat([transformer_out, state_feats], dim=-1)
        per_frame = per_frame.reshape(agent_pos.shape[0], self.n_obs_steps, -1)
        return per_frame.reshape(agent_pos.shape[0], -1)


@dataclass
class PlatonicDiffusionPolicyConfig:
    """Configuration for the Platonic Transformer diffusion policy."""

    horizon: int
    n_obs_steps: int
    action_dim: int
    agent_dim: int
    sample_points: int
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

    state_mlp_hidden: tuple[int, ...]
    unet_hidden_dims: tuple[int, ...]
    unet_kernel_size: int
    unet_num_groups: int
    num_inference_steps: int
    noise_scheduler_cfg: Dict[str, object]


class PlatonicDiffusionPolicy(nn.Module):
    """Diffusion policy that leverages the Platonic Transformer for perception."""

    def __init__(self, cfg: PlatonicDiffusionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg

        transformer = PlatonicTransformer(
            input_dim=cfg.transformer_input_scalar_dim,
            input_dim_vec=cfg.transformer_input_vector_dim,
            hidden_dim=cfg.transformer_hidden_dim,
            output_dim=cfg.transformer_output_dim,
            output_dim_vec=0,
            nhead=cfg.transformer_num_heads,
            num_layers=cfg.transformer_num_layers,
            solid_name=cfg.transformer_solid,
            spatial_dim=3,
            dense_mode=cfg.transformer_dense_mode,
            scalar_task_level="graph",
            vector_task_level="graph",
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
            time_conditioning=False,
            class_conditioning=False,
        )

        if len(cfg.state_mlp_hidden) == 0:
            raise ValueError(
                "state_mlp_hidden must contain at least one hidden/output dim"
            )

        state_dims = (cfg.agent_dim, *cfg.state_mlp_hidden)
        self.encoder = PlatonicObservationEncoder(
            transformer=transformer,
            state_dims=state_dims,
            n_obs_steps=cfg.n_obs_steps,
            scalar_feature_dim=cfg.transformer_input_scalar_dim,
        )

        state_out_dim = state_dims[-1]
        per_frame_dim = cfg.transformer_output_dim + state_out_dim
        global_cond_dim = cfg.n_obs_steps * per_frame_dim

        self.unet = ConditionalUNet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=global_cond_dim,
            hidden_dims=cfg.unet_hidden_dims,
            kernel_size=cfg.unet_kernel_size,
            num_groups=cfg.unet_num_groups,
        )

        scheduler_args = dict(cfg.noise_scheduler_cfg)
        self.scheduler = DDPMScheduler(**scheduler_args)
        self.num_inference_steps = cfg.num_inference_steps

        self.horizon = cfg.horizon
        self.action_dim = cfg.action_dim
        self.n_obs_steps = cfg.n_obs_steps
        self.normalizer = LinearNormalizer()
        self._normalizer_ready = False

    # ------------------------------------------------------------------
    def encode_observation(
        self,
        point_clouds: torch.Tensor,
        agent_pos: torch.Tensor,
        *,
        point_sample_idx: Optional[torch.Tensor] = None,
        point_obs_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode stacked observations into a global conditioning vector."""
        return self.encoder(
            point_clouds,
            agent_pos,
            point_sample_idx=point_sample_idx,
            point_obs_idx=point_obs_idx,
        )

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Attach a frozen normaliser for training and sampling."""
        self.normalizer = copy.deepcopy(normalizer)
        for param in self.normalizer.parameters():
            param.requires_grad_(False)
        self._normalizer_ready = True

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute the diffusion loss for a batch."""
        if not self._normalizer_ready:
            raise RuntimeError("Normalizer must be set before training")

        point_clouds = batch["point_clouds"]
        agent_pos = batch["agent_pos"]
        actions = batch["action"]

        obs_norm = self.normalizer.normalize(
            {"point_clouds": point_clouds, "agent_pos": agent_pos}
        )
        actions_norm = self.normalizer["action"].normalize(actions)

        point_sample_idx = batch.get("point_cloud_sample_idx")
        point_obs_idx = batch.get("point_cloud_obs_idx")

        global_cond = self.encode_observation(
            obs_norm["point_clouds"],
            obs_norm["agent_pos"],
            point_sample_idx=point_sample_idx,
            point_obs_idx=point_obs_idx,
        )

        noise = torch.randn_like(actions_norm)
        batch_size = actions.shape[0]
        device = actions.device

        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )

        noisy_actions = self.scheduler.add_noise(actions_norm, noise, timesteps)
        pred = self.unet(noisy_actions, timesteps, global_cond)

        loss = F.mse_loss(pred, noise)
        return loss, {"train_mse": float(loss.detach().cpu().item())}

    @torch.no_grad()
    def sample(
        self,
        point_clouds: torch.Tensor,
        agent_pos: torch.Tensor,
        *,
        point_sample_idx: Optional[torch.Tensor] = None,
        point_obs_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample denoised action trajectories conditioned on observations."""
        if not self._normalizer_ready:
            raise RuntimeError("Normalizer must be set before sampling")

        device = agent_pos.device
        batch_size = agent_pos.shape[0]
        obs_norm = self.normalizer.normalize(
            {"point_clouds": point_clouds, "agent_pos": agent_pos}
        )
        global_cond = self.encode_observation(
            obs_norm["point_clouds"],
            obs_norm["agent_pos"],
            point_sample_idx=point_sample_idx,
            point_obs_idx=point_obs_idx,
        )

        trajectory = torch.randn(
            batch_size,
            self.horizon,
            self.action_dim,
            device=device,
            dtype=point_clouds.dtype,
        )

        self.scheduler.set_timesteps(self.num_inference_steps, device=device)

        for t in self.scheduler.timesteps:
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = self.unet(trajectory, timesteps, global_cond)
            step_output = self.scheduler.step(noise_pred, t, trajectory)
            trajectory = step_output.prev_sample

        return self.normalizer["action"].unnormalize(trajectory)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        """Restore the module while gracefully handling missing normaliser weights."""
        has_normalizer = any(
            key.startswith("normalizer.params_dict") for key in state_dict.keys()
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


__all__ = ["PlatonicDiffusionPolicy", "PlatonicDiffusionPolicyConfig"]
