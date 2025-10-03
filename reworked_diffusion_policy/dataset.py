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

from .normalization import LinearNormalizer, SingleFieldLinearNormalizer


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
    task_names: Sequence[str] | None = None


class RLBenchTemporalH5Dataset(Dataset):
    """Dataset that eagerly loads the entire cache into memory."""

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.path = Path(cfg.path).expanduser().resolve()
        self._task_names = tuple(cfg.task_names or [])

        self._data: List[Dict[str, torch.Tensor]] = []
        self._stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self._source_files = self._resolve_source_files()
        for file_path in self._source_files:
            with h5py.File(file_path, "r") as handle:
                length = int(handle.attrs["length"])
                samples = handle["samples"]
                for index in tqdm(range(length), desc=f"Loading {file_path.name}"):
                    sample_grp = samples[str(index)]
                    sample = self._process_sample(sample_grp)
                    self._data.append(sample)
                    self._update_stats(sample)

        if not self._data:
            raise RuntimeError(f"No samples loaded from {self._source_files}")

        self._normalizer = self._build_normalizer()

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

    # ------------------------------------------------------------------
    def _resolve_source_files(self) -> List[Path]:
        if self.path.is_file():
            return [self.path]
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.path}")
        if not self.path.is_dir():
            raise FileNotFoundError(f"Unsupported dataset path: {self.path}")
        if not self._task_names:
            raise ValueError("task_names must be provided when dataset path is a directory")

        resolved: List[Path] = []
        for task in self._task_names:
            task_dir = self.path / task
            if not task_dir.is_dir():
                raise FileNotFoundError(f"Task directory not found: {task_dir}")
            h5_files = sorted(task_dir.glob("*.h5"))
            if not h5_files:
                raise FileNotFoundError(f"No .h5 files found in {task_dir}")
            resolved.extend(h5_files)
        if not resolved:
            raise RuntimeError(f"No dataset files found for tasks {self._task_names}")
        return resolved

    def _update_stats(self, sample: Dict[str, torch.Tensor]) -> None:
        self._accumulate("point_clouds", sample["point_clouds"])
        self._accumulate("agent_pos", sample["agent_pos"])
        self._accumulate("action", sample["action"])

    def _accumulate(self, key: str, tensor: torch.Tensor) -> None:
        feature_dim = tensor.shape[-1]
        flat = tensor.reshape(-1, feature_dim).to(torch.float64)
        if key not in self._stats:
            self._stats[key] = {
                "count": torch.zeros(1, dtype=torch.float64),
                "sum": torch.zeros(feature_dim, dtype=torch.float64),
                "sum_sq": torch.zeros(feature_dim, dtype=torch.float64),
                "min": torch.full((feature_dim,), float("inf"), dtype=torch.float64),
                "max": torch.full((feature_dim,), float("-inf"), dtype=torch.float64),
            }

        stats = self._stats[key]
        count = flat.shape[0]
        stats["count"] += count
        stats["sum"] += flat.sum(dim=0)
        stats["sum_sq"] += (flat ** 2).sum(dim=0)
        stats["min"] = torch.minimum(stats["min"], flat.min(dim=0).values)
        stats["max"] = torch.maximum(stats["max"], flat.max(dim=0).values)

    def _build_normalizer(self) -> LinearNormalizer:
        output_min = -1.0
        output_max = 1.0
        range_eps = 1e-4

        normalizer = LinearNormalizer()
        for key, stats in self._stats.items():
            total_count = int(stats["count"].item())
            if total_count <= 0:
                raise RuntimeError(f"No statistics accumulated for field '{key}'")

            sum_vals = stats["sum"]
            sum_sq_vals = stats["sum_sq"]
            mean = sum_vals / total_count
            variance = torch.clamp(sum_sq_vals / total_count - mean ** 2, min=0.0)
            std = torch.sqrt(variance)

            input_min = stats["min"].to(torch.float32)
            input_max = stats["max"].to(torch.float32)
            input_mean = mean.to(torch.float32)
            input_std = std.to(torch.float32)

            input_range = input_max - input_min
            scale = torch.empty_like(input_range)
            offset = torch.empty_like(input_range)

            ignore = input_range < range_eps
            safe_range = input_range.clone()
            safe_range[ignore] = output_max - output_min
            scale = (output_max - output_min) / safe_range
            offset = output_min - scale * input_min
            midpoint = (output_max + output_min) / 2.0
            offset[ignore] = midpoint - input_min[ignore]

            input_stats = {
                "min": input_min,
                "max": input_max,
                "mean": input_mean,
                "std": input_std,
            }

            normalizer[key] = SingleFieldLinearNormalizer.create_manual(scale, offset, input_stats)

        return normalizer

    @property
    def normalizer(self) -> LinearNormalizer:
        return self._normalizer

    @property
    def source_files(self) -> Tuple[Path, ...]:
        return tuple(self._source_files)

    @property
    def task_names(self) -> Tuple[str, ...]:
        return tuple(self._task_names)

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
