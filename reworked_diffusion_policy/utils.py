"""Utility helpers for training and evaluation."""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import numpy as np
import wandb


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExponentialMovingAverage:
    """Maintains an exponential moving average copy of a model."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = dict(self.ema_model.named_parameters())
        model_state = dict(model.named_parameters())
        for name, param in model_state.items():
            if name not in ema_state:
                continue
            ema_param = ema_state[name]
            ema_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

        ema_buffers = dict(self.ema_model.named_buffers())
        for name, buffer in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].copy_(buffer)

    def state_dict(self):
        return self.ema_model.state_dict()

    def to(self, device: torch.device | str) -> "ExponentialMovingAverage":
        self.ema_model.to(device)
        return self


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


@torch.no_grad()
def visualize_trajectories(
    point_cloud: torch.Tensor,
    gt_actions: torch.Tensor,
    pred_actions: torch.Tensor,
    *,
    point_size: float,
    axes_length: float,
    axes_radius: float,
    visualize_indefinitely: bool = False,
) -> None:
    """Launch a viser viewer comparing ground-truth and predicted trajectories."""

    try:
        import viser
    except ImportError:
        print("[visualize] viser not installed; skipping visualization.")
        return

    server = viser.ViserServer()

    cloud = point_cloud[-1]
    points = cloud[..., :3].detach().cpu().numpy()
    if cloud.shape[-1] >= 6:
        colors = cloud[..., 3:6].detach().cpu().numpy()
        colors = (colors * 255.0).clip(0, 255).astype("uint8")
    else:
        colors = None

    server.add_point_cloud(
        "/observation",
        points=points,
        colors=colors,
        point_size=point_size,
    )

    def _add_frames(prefix: str, actions: torch.Tensor) -> None:
        data = actions.detach().cpu().numpy()
        for idx, act in enumerate(data):
            if act.shape[0] < 7:
                continue
            position = act[:3]
            qx, qy, qz, qw = act[3:7]
            norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            if norm == 0:
                continue
            server.add_frame(
                f"{prefix}/step_{idx}",
                position=position,
                wxyz=(qw / norm, qx / norm, qy / norm, qz / norm),
                axes_length=axes_length,
                axes_radius=axes_radius,
            )

    _add_frames("/ground_truth", gt_actions)
    _add_frames("/prediction", pred_actions)

    print("Viser server running at:", server.get_host())
    print("Press Ctrl+C to exit visualization.")
    try:
        import time

        if visualize_indefinitely:
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        server.stop()


__all__ = [
    "set_seed",
    "ExponentialMovingAverage",
    "mse",
    "visualize_trajectories",
    "log_pointcloud_wandb",
]


def log_pointcloud_wandb(
    *,
    wandb_run,
    point_cloud: torch.Tensor,
    gt_actions: torch.Tensor,
    pred_actions: torch.Tensor,
    tag: str,
) -> None:
    """Log a combined point-cloud plus action targets to wandb."""

    if wandb_run is None:
        return

    cloud = point_cloud.detach().cpu().numpy()
    coords = cloud[..., :3]
    if cloud.shape[-1] >= 6:
        colors = cloud[..., 3:6]
        if colors.max() <= 1.0:
            colors = colors * 255.0
    else:
        colors = np.ones_like(coords) * 127.0

    base_points = np.concatenate([coords.reshape(-1, 3), colors.reshape(-1, 3)], axis=-1)

    def _actions_to_points(actions: torch.Tensor, color_rgb: np.ndarray) -> np.ndarray:
        arr = actions.detach().cpu().numpy()
        coords = arr[:, :3]
        colors = np.repeat(color_rgb[None, :], coords.shape[0], axis=0)
        return np.concatenate([coords, colors], axis=-1)

    gt_points = _actions_to_points(gt_actions, np.array([0.0, 255.0, 0.0], dtype=np.float32))
    pred_points = _actions_to_points(pred_actions, np.array([255.0, 0.0, 0.0], dtype=np.float32))

    all_points = np.concatenate([base_points, gt_points, pred_points], axis=0)

    wandb_run.log(
        {
            f"{tag}/pointcloud": wandb.Object3D(
                {
                    "points": {
                        "format": "xyzrgb",
                        "data": all_points.astype(np.float32).tolist(),
                    }
                }
            )
        }
    )
