"""Dataset utilities for RLBench temporal demonstration caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def ensure_float_colors(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    array = array.astype(np.float32)
    if array.size > 0 and float(array.max()) > 1.0:
        array = array / 255.0
    return array


@dataclass
class DatasetConfig:
    path: str
    sample_points: int
    n_obs_steps: int
    action_horizon: int
    use_point_colors: bool = True


class RLBenchTemporalH5Dataset(Dataset):
    """Dataset that eagerly loads the entire cache into memory."""

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.path = Path(cfg.path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        self._data: List[Dict[str, torch.Tensor]] = []
        with h5py.File(self.path, "r") as handle:
            length = int(handle.attrs["length"])
            samples = handle["samples"]
            for index in tqdm(range(length), desc="Loading dataset"):
                sample_grp = samples[str(index)]
                self._data.append(self._process_sample(sample_grp))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        if index < 0 or index >= len(self._data):
            raise IndexError(index)
        sample = self._data[index]
        return {
            "point_clouds": sample["point_clouds"].clone(),
            "agent_pos": sample["agent_pos"].clone(),
            "action": sample["action"].clone(),
        }

    # ------------------------------------------------------------------
    def _process_sample(self, sample_grp: h5py.Group) -> Dict[str, torch.Tensor]:
        obs_grp = sample_grp["observation"]
        pc_sequence = obs_grp["point_cloud_sequence"]
        proprio_sequence = torch.from_numpy(obs_grp["proprio_sequence"][()].astype(np.float32))

        point_clouds: List[torch.Tensor] = []
        agent_states: List[torch.Tensor] = []
        obs_len = int(pc_sequence.attrs.get("length", len(pc_sequence)))
        start = max(0, obs_len - self.cfg.n_obs_steps)
        indices = list(range(start, obs_len))

        for obs_idx in indices:
            frame = pc_sequence[str(obs_idx)]
            points = torch.from_numpy(frame["points"][()].astype(np.float32))
            masks = torch.from_numpy(frame["masks"][()].astype(np.bool_))
            if self.cfg.use_point_colors:
                colors = torch.from_numpy(ensure_float_colors(frame["colors"][()]))
            else:
                colors = None

            valid_points = points[masks]
            if valid_points.shape[0] == 0:
                valid_points = points
                valid_colors = colors
            else:
                valid_colors = colors[masks] if colors is not None else None

            sampled_points, sampled_colors = sample_points(
                valid_points,
                valid_colors,
                self.cfg.sample_points,
            )

            if self.cfg.use_point_colors:
                if sampled_colors is None:
                    sampled_colors = torch.zeros_like(sampled_points)
                features = torch.cat([sampled_points, sampled_colors], dim=-1)
            else:
                features = sampled_points
            point_clouds.append(features)

            state_index = min(obs_idx, proprio_sequence.shape[0] - 1)
            agent_states.append(proprio_sequence[state_index])

        if not point_clouds:
            raise RuntimeError("Sample contains no point clouds")

        while len(point_clouds) < self.cfg.n_obs_steps:
            point_clouds.insert(0, point_clouds[0].clone())
            agent_states.insert(0, agent_states[0].clone())

        point_cloud_tensor = torch.stack(point_clouds, dim=0)
        agent_state_tensor = torch.stack(agent_states, dim=0)

        action_seq = torch.from_numpy(
            sample_grp["action"]["sequence"][()].astype(np.float32)
        )
        action_seq = self._format_action(action_seq)

        return {
            "point_clouds": point_cloud_tensor,
            "agent_pos": agent_state_tensor,
            "action": action_seq,
        }

    def _format_action(self, action_seq: torch.Tensor) -> torch.Tensor:
        horizon = self.cfg.action_horizon
        steps = action_seq.shape[0]
        if steps > horizon:
            return action_seq[:horizon]
        if steps == horizon:
            return action_seq
        if steps == 0:
            pad = torch.zeros((horizon, action_seq.shape[-1]), dtype=action_seq.dtype)
        else:
            pad = action_seq[-1:].repeat(horizon - steps, 1)
        return torch.cat([action_seq, pad], dim=0)


def sample_points(
    points: torch.Tensor,
    colors: Optional[torch.Tensor],
    target_points: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    total = points.shape[0]
    if total >= target_points:
        indices = torch.randperm(total)[:target_points]
    else:
        if total == 0:
            points = torch.zeros((1, points.shape[-1]), dtype=torch.float32)
            if colors is not None:
                colors = torch.zeros((1, colors.shape[-1]), dtype=torch.float32)
            total = 1
        indices = torch.randint(0, total, (target_points,), dtype=torch.long)

    sampled_points = points.index_select(0, indices).to(torch.float32)
    sampled_colors = None
    if colors is not None:
        sampled_colors = colors.index_select(0, indices).to(torch.float32)
    return sampled_points, sampled_colors


def collate_temporal_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    point_clouds = torch.stack([item["point_clouds"] for item in batch], dim=0)
    agent_pos = torch.stack([item["agent_pos"] for item in batch], dim=0)
    actions = torch.stack([item["action"] for item in batch], dim=0)

    return {
        "point_clouds": point_clouds,
        "agent_pos": agent_pos,
        "action": actions,
    }


__all__ = [
    "DatasetConfig",
    "RLBenchTemporalH5Dataset",
    "collate_temporal_batch",
    "ensure_float_colors",
    "sample_points",
]
