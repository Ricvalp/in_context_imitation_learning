"""High-level diffusion policy module tying encoder and UNet together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .pointnet import ObservationEncoder, PointNetEncoder
from .unet1d import ConditionalUNet1D
from ..normalization import LinearNormalizer


@dataclass
class DiffusionPolicyConfig:
    horizon: int
    n_obs_steps: int
    action_dim: int
    agent_dim: int
    sample_points: int
    use_point_colors: bool
    pointnet_in_channels: int
    pointnet_hidden_dims: tuple[int, ...]
    pointnet_out_dim: int
    pointnet_use_layernorm: bool
    state_mlp_hidden: tuple[int, ...]
    unet_hidden_dims: tuple[int, ...]
    unet_kernel_size: int
    unet_num_groups: int
    num_inference_steps: int
    noise_scheduler_cfg: Dict[str, object]


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg: DiffusionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg

        pointnet = PointNetEncoder(
            in_channels=cfg.pointnet_in_channels,
            hidden_dims=cfg.pointnet_hidden_dims,
            out_dim=cfg.pointnet_out_dim,
            use_layernorm=cfg.pointnet_use_layernorm,
        )

        if len(cfg.state_mlp_hidden) == 0:
            raise ValueError("state_mlp_hidden must contain at least one hidden/output dim")

        state_dims = (cfg.agent_dim, *cfg.state_mlp_hidden)
        self.encoder = ObservationEncoder(
            pointnet=pointnet,
            state_dims=state_dims,
            n_obs_steps=cfg.n_obs_steps,
        )

        state_out_dim = state_dims[-1]
        per_frame_dim = pointnet.out_dim + state_out_dim
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
    def encode_observation(self, point_clouds: torch.Tensor, agent_pos: torch.Tensor) -> torch.Tensor:
        return self.encoder(point_clouds, agent_pos)

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer = copy.deepcopy(normalizer)
        for param in self.normalizer.parameters():
            param.requires_grad_(False)
        self._normalizer_ready = True

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        if not self._normalizer_ready:
            raise RuntimeError("Normalizer must be set before training")

        point_clouds = batch["point_clouds"]
        agent_pos = batch["agent_pos"]
        actions = batch["action"]

        obs_norm = self.normalizer.normalize(
            {"point_clouds": point_clouds, "agent_pos": agent_pos}
        )
        actions_norm = self.normalizer["action"].normalize(actions)

        global_cond = self.encode_observation(obs_norm["point_clouds"], obs_norm["agent_pos"])

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
    def sample(self, point_clouds: torch.Tensor, agent_pos: torch.Tensor) -> torch.Tensor:
        if not self._normalizer_ready:
            raise RuntimeError("Normalizer must be set before sampling")

        device = point_clouds.device
        batch_size = point_clouds.shape[0]
        obs_norm = self.normalizer.normalize(
            {"point_clouds": point_clouds, "agent_pos": agent_pos}
        )
        global_cond = self.encode_observation(obs_norm["point_clouds"], obs_norm["agent_pos"])

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
        has_normalizer = any(key.startswith("normalizer.params_dict") for key in state_dict.keys())

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
                    "Unexpected key(s) in state_dict: " + ", ".join(unexpected_keys) + "."
                )
            error_msg = "\n\t".join(error_lines)
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t{error_msg}"
            )

        return type(result)(missing_keys, unexpected_keys)


__all__ = ["DiffusionPolicy", "DiffusionPolicyConfig"]
