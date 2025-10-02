"""Temporal imitation-learning dataset utilities for RLBench exports."""
from __future__ import annotations

import argparse
import pickle
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tools.datasets.common import MASK_LABEL_MAP_FILENAME, resolve_mask_ids

PointCloudTransform = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]


@dataclass(frozen=True)
class _SampleIndex:
    task: str
    variation: int
    episode: int
    step: int
    demo_path: Path
    episode_dir: Path


class RLBenchTemporalPointCloudDataset(Dataset):
    """Temporal dataset over RLBench demonstrations with merged point-clouds."""

    MERGED_POINT_CLOUD_FOLDER = "merged_point_cloud"
    LOW_DIM_PICKLE = "low_dim_obs.pkl"
    VARIATION_DESCRIPTIONS_FILE = "variation_descriptions.pkl"
    MASK_NAMES_TO_IGNORE = [
        "Floor",
        "Wall1",
        "Wall2",
        "Wall3",
        "Wall4",
        "Roof",
        "workspace",
        "diningTable_visible",
        "ResizableFloor_5_25_visibleElement"
    ]
    MASK_LABEL_MAP_FILENAME = MASK_LABEL_MAP_FILENAME
    DEFAULT_MASK_IDS_TO_IGNORE: Tuple[int, ...] = []

    def __init__(
        self,
        root: Path | str,
        task: str,
        variations: Optional[Sequence[int] | Tuple[int, int]] = None,
        *,
        point_cloud_history: int = 1,
        proprio_history: int = 1,
        future_action_window: int = 4,
        future_action_stride: int = 1,
        proprio_keys: Sequence[str] = ("gripper_pose", "gripper_open"),
        action_keys: Sequence[str] = ("gripper_pose", "gripper_open"),
        point_cloud_transform: Optional[PointCloudTransform] = None,
        cache_size: int = 16,
    ) -> None:
        if future_action_window <= 0:
            raise ValueError("future_action_window must be positive")
        if future_action_stride <= 0:
            raise ValueError("future_action_stride must be positive")
        if point_cloud_history < 0 or proprio_history < 0:
            raise ValueError("history arguments must be >= 0")
        if not proprio_keys:
            raise ValueError("proprio_keys must contain at least one attribute")
        if not action_keys:
            raise ValueError("action_keys must contain at least one attribute")

        self.root = Path(root)
        self.task = task
        self.task_root = self.root / task
        if not self.task_root.is_dir():
            raise FileNotFoundError(f"Task directory not found: {self.task_root}")

        self.point_cloud_history = point_cloud_history
        self.proprio_history = proprio_history
        self.future_action_window = future_action_window
        self.future_action_stride = future_action_stride
        self.proprio_keys = tuple(proprio_keys)
        self.action_keys = tuple(action_keys)
        self.point_cloud_transform = point_cloud_transform
        self.cache_size = cache_size

        self._mask_ignore_cache: Dict[Path, Tuple[int, ...]] = {}
        self._mask_missing_names: Dict[Path, Set[str]] = {}
        self._mask_error_logged: Set[Path] = set()

        variation_ids = self._normalise_variations(variations)
        self._demo_cache: OrderedDict[str, object] = OrderedDict()
        self._indices: List[_SampleIndex] = []
        self._build_index(variation_ids)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, item: int) -> Dict[str, object]:
        entry = self._indices[item]
        demo = self._load_demo(entry.demo_path)
        step = entry.step

        point_cloud_sequence = self._load_point_cloud_sequence(entry.episode_dir, step)
        proprio_sequence = self._build_vector_sequence(demo, step, self.proprio_history, self.proprio_keys)
        action_sequence = self._build_vector_sequence(
            demo,
            step,
            self.future_action_window - 1,
            self.action_keys,
            forward=True,
            start_offset=1,
            stride=self.future_action_stride,
        )

        return {
            "observation": {
                "point_cloud_sequence": point_cloud_sequence,
                "proprio_sequence": proprio_sequence,
            },
            "action": action_sequence,
            "meta": {
                "task": entry.task,
                "variation": entry.variation,
                "episode": entry.episode,
                "step": entry.step,
            },
        }

    # ------------------------------------------------------------------
    # Index building ----------------------------------------------------

    def _build_index(self, variation_ids: Sequence[int]) -> None:
        history_requirement = max(self.point_cloud_history, self.proprio_history)
        future_requirement = 1 + self.future_action_stride * (self.future_action_window - 1)
        for variation in variation_ids:
            variation_dir = self.task_root / f"variation{variation}"
            episodes_dir = variation_dir / "episodes"
            if not episodes_dir.is_dir():
                continue
            for episode_dir in sorted(episodes_dir.iterdir()):
                if not episode_dir.is_dir() or not episode_dir.name.startswith("episode"):
                    continue
                demo_path = episode_dir / self.LOW_DIM_PICKLE
                if not demo_path.is_file():
                    continue
                demo_len = len(self._load_demo(demo_path))
                max_start = demo_len - 1 - future_requirement
                if max_start < history_requirement:
                    continue
                for step in range(history_requirement, max_start + 1):
                    self._indices.append(
                        _SampleIndex(
                            task=self.task,
                            variation=variation,
                            episode=int(episode_dir.name.replace("episode", "")),
                            step=step,
                            demo_path=demo_path,
                            episode_dir=episode_dir,
                        )
                    )

    def _normalise_variations(
        self,
        variations: Optional[Sequence[int] | Tuple[int, int]],
    ) -> List[int]:
        if variations is None:
            return sorted(self._discover_variations())
        if isinstance(variations, tuple) and len(variations) == 2:
            start, end = variations
            if start > end:
                raise ValueError("variation range start must be <= end")
            return list(range(int(start), int(end) + 1))
        result = sorted({int(v) for v in variations})
        if not result:
            raise ValueError("variations sequence is empty")
        return result

    def _discover_variations(self) -> Iterable[int]:
        for child in self.task_root.iterdir():
            if child.is_dir() and child.name.startswith("variation"):
                suffix = child.name.replace("variation", "")
                if suffix.isdigit():
                    yield int(suffix)

    # ------------------------------------------------------------------
    # Loading helpers ---------------------------------------------------

    def _load_demo(self, path: Path):
        key = str(path)
        if key in self._demo_cache:
            self._demo_cache.move_to_end(key)
            return self._demo_cache[key]
        with path.open("rb") as handle:
            demo = pickle.load(handle)
        if self.cache_size > 0:
            self._demo_cache[key] = demo
            while len(self._demo_cache) > self.cache_size:
                self._demo_cache.popitem(last=False)
        return demo

    def _load_point_cloud_sequence(self, episode_dir: Path, step: int) -> List[Dict[str, torch.Tensor]]:
        indices = range(step - self.point_cloud_history, step + 1)
        sequence: List[Dict[str, torch.Tensor]] = []
        for idx in indices:
            pc_path = episode_dir / self.MERGED_POINT_CLOUD_FOLDER / f"{idx}.npz"
            if not pc_path.is_file():
                raise FileNotFoundError(f"Missing point cloud file: {pc_path}")
            with np.load(pc_path) as data:
                points = torch.from_numpy(data["points"].astype(np.float32))
                colors = torch.from_numpy(data["colors"].astype(np.float32)) / 255.0
                raw_masks = torch.from_numpy(data["masks"].astype(np.int64))
            ignore_ids = self._get_mask_ignore_ids(episode_dir)
            masks = self._process_masks(raw_masks, ignore_ids)
            cloud = {"points": points, "colors": colors, "masks": masks}
            if self.point_cloud_transform is not None:
                cloud = self.point_cloud_transform(cloud)
            sequence.append(cloud)
        return sequence

    def _get_mask_ignore_ids(self, episode_dir: Path) -> Tuple[int, ...]:
        variation_dir = episode_dir.parent.parent.resolve()
        cached = self._mask_ignore_cache.get(variation_dir)
        if cached is not None:
            return cached
        ignore_ids = self._load_mask_ignore_ids(variation_dir)
        cached = tuple(sorted(ignore_ids))
        self._mask_ignore_cache[variation_dir] = cached
        return cached

    def _load_mask_ignore_ids(self, variation_dir: Path) -> Set[int]:
        ignore_ids: Set[int] = set(self.DEFAULT_MASK_IDS_TO_IGNORE)
        matched, missing, error = resolve_mask_ids(
            variation_dir,
            self.MASK_NAMES_TO_IGNORE,
            map_filename=self.MASK_LABEL_MAP_FILENAME,
        )
        ignore_ids.update(matched)
        self._log_mask_warnings(variation_dir, missing, error)
        return ignore_ids

    def _log_mask_warnings(
        self,
        variation_dir: Path,
        missing: Set[str],
        error: str | None,
    ) -> None:
        if error and variation_dir not in self._mask_error_logged:
            warnings.warn(
                f"Mask label map warning for {variation_dir}: {error}. "
                "Falling back to default mask filtering.",
                RuntimeWarning,
            )
            self._mask_error_logged.add(variation_dir)

        if not missing:
            return

        seen = self._mask_missing_names.setdefault(variation_dir, set())
        new_missing = missing - seen
        if not new_missing:
            return
        seen.update(new_missing)
        warnings.warn(
            f"Mask labels not found for {variation_dir}: {', '.join(sorted(new_missing))}.",
            RuntimeWarning,
        )

    def _process_masks(self, masks: torch.Tensor, ignore_ids: Iterable[int]) -> torch.Tensor:
        ignore_list = list(ignore_ids)
        if not ignore_list:
            return torch.ones_like(masks, dtype=torch.bool)
        valid = torch.ones_like(masks, dtype=torch.bool)
        for ignore_idx in ignore_list:
            valid &= masks != ignore_idx
        return valid

    def _build_vector_sequence(
        self,
        demo,
        step: int,
        history: int,
        keys: Sequence[str],
        *,
        forward: bool = False,
        start_offset: int = 0,
        stride: int = 1,
    ) -> torch.Tensor:
        if forward:
            indices = [step + start_offset + stride * i for i in range(history + 1)]
        else:
            start = step - history
            indices = range(start, step + 1)
        vectors: List[torch.Tensor] = []
        for idx in indices:
            obs = demo[idx]
            vector_parts: List[np.ndarray] = []
            for key in keys:
                value = getattr(obs, key)
                if value is None:
                    raise ValueError(f"Observation attribute '{key}' is None at step {idx}")
                array = np.asarray(value, dtype=np.float32)
                vector_parts.append(array.ravel())
            vectors.append(torch.from_numpy(np.concatenate(vector_parts)))
        return torch.stack(vectors, dim=0)


def rlbench_temporal_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Collate samples while keeping ragged point-cloud sequences as lists."""

    point_cloud_sequences = [item["observation"]["point_cloud_sequence"] for item in batch]
    proprio = torch.stack([item["observation"]["proprio_sequence"] for item in batch], dim=0)
    actions = torch.stack([item["action"] for item in batch], dim=0)
    meta = {
        key: torch.tensor([item["meta"][key] for item in batch], dtype=torch.long)
        for key in ("variation", "episode", "step")
    }
    meta["task"] = [item["meta"]["task"] for item in batch]
    return {
        "observation": {
            "point_cloud_sequence": point_cloud_sequences,
            "proprio_sequence": proprio,
        },
        "action": actions,
        "meta": meta,
    }


def create_temporal_dataloader(
    dataset: RLBenchTemporalPointCloudDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Instantiate a DataLoader with the temporal collate helper."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=rlbench_temporal_collate,
    )


__all__ = [
    "RLBenchTemporalPointCloudDataset",
    "create_temporal_dataloader",
    "rlbench_temporal_collate",
]
