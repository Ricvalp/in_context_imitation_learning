"""Helpers for deploying the diffusion policy inside RLBench."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, List, Optional, Sequence, Tuple

import numpy as np
import torch

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RLBENCH_ROOT = _REPO_ROOT / "rlbench_dataset_gen"
if _RLBENCH_ROOT.exists() and str(_RLBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RLBENCH_ROOT))

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
from pyrep.const import RenderMode

from .dataset import ensure_float_colors, sample_points


CAMERA_NAMES: Tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "overhead",
    "wrist",
    "front",
)


@dataclass
class SimulationInputConfig:
    sample_points: int
    n_obs_steps: int
    use_point_colors: bool
    device: torch.device


def create_action_mode() -> MoveArmThenGripper:
    return MoveArmThenGripper(EndEffectorPoseViaIK(), Discrete())


def create_observation_config(
    *,
    image_size: Sequence[int] = (128, 128),
    renderer: str = "opengl3",
) -> ObservationConfig:
    obs_config = ObservationConfig()
    obs_config.set_all(False)

    render_mode = RenderMode.OPENGL3 if renderer == "opengl3" else RenderMode.OPENGL

    for camera_name in CAMERA_NAMES:
        camera_cfg = getattr(obs_config, f"{camera_name}_camera")
        camera_cfg.set_all(False)
        camera_cfg.rgb = True
        camera_cfg.depth = False
        camera_cfg.point_cloud = True
        camera_cfg.mask = True
        camera_cfg.image_size = tuple(int(v) for v in image_size)
        camera_cfg.render_mode = render_mode
        camera_cfg.masks_as_one_channel = True
        camera_cfg.depth_in_meters = False

    obs_config.gripper_open = True
    obs_config.gripper_pose = True
    obs_config.joint_velocities = False
    obs_config.joint_positions = False
    obs_config.joint_forces = False
    obs_config.gripper_matrix = False
    obs_config.gripper_joint_positions = False
    obs_config.gripper_touch_forces = False

    return obs_config


class ObservationProcessor:
    """Converts RLBench observations into model-ready tensors."""

    def __init__(self, cfg: SimulationInputConfig) -> None:
        self.cfg = cfg
        self._mask_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def set_mask_filter(self, mask_filter: Optional[Callable[[np.ndarray], np.ndarray]]) -> None:
        self._mask_filter = mask_filter

    def extract(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        points, colors = self._merge_point_clouds(obs)
        points_t = torch.from_numpy(points.astype(np.float32))
        if self.cfg.use_point_colors and colors is not None:
            colors_np = ensure_float_colors(colors)
            colors_t = torch.from_numpy(colors_np.astype(np.float32))
        else:
            colors_t = None

        sampled_points, sampled_colors = sample_points(
            points_t,
            colors_t,
            self.cfg.sample_points,
        )

        if self.cfg.use_point_colors:
            if sampled_colors is None:
                sampled_colors = torch.zeros_like(sampled_points)
            features = torch.cat([sampled_points, sampled_colors], dim=-1)
        else:
            features = sampled_points

        agent_state = self._extract_agent_state(obs)
        return features.to(torch.float32), agent_state

    # ------------------------------------------------------------------
    def _merge_point_clouds(self, obs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        merged_points: List[np.ndarray] = []
        merged_colors: List[np.ndarray] = []

        for camera in CAMERA_NAMES:
            pc = getattr(obs, f"{camera}_point_cloud", None)
            rgb = getattr(obs, f"{camera}_rgb", None)
            mask = getattr(obs, f"{camera}_mask", None)

            if pc is None or rgb is None:
                continue

            points = np.asarray(pc, dtype=np.float32).reshape(-1, 3)
            colors = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
            if mask is None:
                mask_flat = None
                valid_mask = np.ones(points.shape[0], dtype=bool)
            else:
                mask_flat = np.asarray(mask).reshape(-1)
                valid_mask = np.ones(points.shape[0], dtype=bool)

            finite = np.isfinite(points).all(axis=1)
            valid_mask &= finite

            if mask_flat is not None:
                filter_fn = self._mask_filter
                if filter_fn is not None:
                    filtered = filter_fn(mask_flat)
                    filtered = np.asarray(filtered, dtype=bool)
                    if filtered.shape != valid_mask.shape:
                        filtered = filtered.reshape(valid_mask.shape)
                    valid_mask &= filtered

            if not np.any(valid_mask):
                continue

            merged_points.append(points[valid_mask])
            merged_colors.append(colors[valid_mask])

        if not merged_points:
            empty_points = np.zeros((1, 3), dtype=np.float32)
            if self.cfg.use_point_colors:
                empty_colors = np.zeros((1, 3), dtype=np.uint8)
            else:
                empty_colors = None
            return empty_points, empty_colors

        all_points = np.concatenate(merged_points, axis=0)
        if self.cfg.use_point_colors:
            all_colors = np.concatenate(merged_colors, axis=0)
        else:
            all_colors = None
        return all_points, all_colors

    def _extract_agent_state(self, obs) -> torch.Tensor:
        pose = np.asarray(obs.gripper_pose, dtype=np.float32).reshape(-1)
        open_amount = np.array([float(obs.gripper_open)], dtype=np.float32)
        state = np.concatenate([pose, open_amount], axis=0)
        return torch.from_numpy(state)


class ObservationHistory:
    """Maintains the sliding window of observations for conditioning."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._points: Deque[torch.Tensor] = deque(maxlen=capacity)
        self._agents: Deque[torch.Tensor] = deque(maxlen=capacity)

    def reset(self) -> None:
        self._points.clear()
        self._agents.clear()

    def append(self, features: torch.Tensor, agent_state: torch.Tensor) -> None:
        self._points.append(features.detach().clone())
        self._agents.append(agent_state.detach().clone())

    def stacked(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._points:
            raise RuntimeError("Observation history is empty; call append() first")
        points_list = list(self._points)
        agent_list = list(self._agents)
        while len(points_list) < self.capacity:
            points_list.insert(0, points_list[0])
            agent_list.insert(0, agent_list[0])
        points_tensor = torch.stack(points_list[-self.capacity :], dim=0).to(device)
        agent_tensor = torch.stack(agent_list[-self.capacity :], dim=0).to(device)
        return points_tensor, agent_tensor

    def latest_agent_state(self) -> torch.Tensor:
        if not self._agents:
            raise RuntimeError("Observation history is empty")
        return self._agents[-1]


def canonicalise_task_name(name: str) -> str:
    clean = name.replace(".py", "").strip()
    if clean.lower() == clean:
        return clean
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", clean).lower()
    return snake


def resolve_task_class(name: str):
    canonical = canonicalise_task_name(name)
    return task_file_to_task_class(canonical)


def action_plan_to_command(
    action_plan: torch.Tensor,
    *,
    last_agent_state: torch.Tensor,
) -> np.ndarray:
    action_np = action_plan.detach().cpu().numpy()
    pose = action_np[:7].copy()
    quat = pose[3:]
    norm = np.linalg.norm(quat)
    if norm < 1e-6:
        pose[3:] = last_agent_state.detach().cpu().numpy()[3:7]
    else:
        pose[3:] = quat / norm

    grip = float(action_np[7])
    grip_cmd = 1.0 if grip > 0.5 else 0.0
    return np.concatenate([pose, [grip_cmd]]).astype(np.float32)


def instantiate_environment(
    *,
    action_mode: Optional[MoveArmThenGripper] = None,
    obs_config: Optional[ObservationConfig] = None,
    headless: bool = True,
) -> Environment:
    if action_mode is None:
        action_mode = create_action_mode()
    if obs_config is None:
        obs_config = create_observation_config()

    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=headless,
    )
    env.launch()
    return env


__all__ = [
    "CAMERA_NAMES",
    "SimulationInputConfig",
    "ObservationProcessor",
    "ObservationHistory",
    "action_plan_to_command",
    "instantiate_environment",
    "resolve_task_class",
    "create_action_mode",
    "create_observation_config",
]
