"""In-context imitation learning dataset utilities for RLBench exports."""
from __future__ import annotations

import pickle
import random
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
class _EpisodeRef:
    task: str
    variation: int
    episode: int
    demo_path: Path
    episode_dir: Path
    length: int


class RLBenchInContextDataset(Dataset):
    """Few-shot in-context imitation learning dataset."""

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
    ]
    MASK_LABEL_MAP_FILENAME = MASK_LABEL_MAP_FILENAME
    DEFAULT_MASK_IDS_TO_IGNORE: Tuple[int, ...] = []

    def __init__(
        self,
        root: Path | str,
        task: str,
        variations: Optional[Sequence[int] | Tuple[int, int]] = None,
        *,
        support_size_range: Tuple[int, int] = (1, 4),
        support_frames: int = 7,
        query_future_steps: int = 7,
        query_future_stride: int = 10,
        point_cloud_transform: Optional[PointCloudTransform] = None,
        proprio_keys: Sequence[str] = ("gripper_pose", "gripper_open"),
        action_keys: Sequence[str] = ("gripper_pose", "gripper_open"),
        match_variation: bool = False,
        cache_size: int = 16,
        seed: Optional[int] = None,
    ) -> None:
        if support_size_range[0] < 1 or support_size_range[0] > support_size_range[1]:
            raise ValueError("support_size_range must be a valid inclusive interval >= 1")
        if support_frames <= 0:
            raise ValueError("support_frames must be positive")
        if query_future_steps <= 0:
            raise ValueError("query_future_steps must be positive")
        if query_future_stride <= 0:
            raise ValueError("query_future_stride must be positive")
        if not proprio_keys:
            raise ValueError("proprio_keys must contain at least one attribute")
        if not action_keys:
            raise ValueError("action_keys must contain at least one attribute")

        self.root = Path(root)
        self.task = task
        self.task_root = self.root / task
        if not self.task_root.is_dir():
            raise FileNotFoundError(f"Task directory not found: {self.task_root}")

        self.support_size_range = support_size_range
        self.support_frames = support_frames
        self.query_future_steps = query_future_steps
        self.query_future_stride = query_future_stride
        self.point_cloud_transform = point_cloud_transform
        self.proprio_keys = tuple(proprio_keys)
        self.action_keys = tuple(action_keys)
        self.match_variation = match_variation
        self.cache_size = cache_size
        self._rng = random.Random(seed)

        self._mask_ignore_cache: Dict[Path, Tuple[int, ...]] = {}
        self._mask_missing_names: Dict[Path, Set[str]] = {}
        self._mask_error_logged: Set[Path] = set()

        min_required_length = max(support_frames, 1 + query_future_stride * query_future_steps)

        variation_ids = self._normalise_variations(variations)
        self._demo_cache: OrderedDict[str, object] = OrderedDict()
        self._episode_refs: List[_EpisodeRef] = []
        self._episodes_by_variation: Dict[int, List[int]] = {}
        self._build_episode_index(variation_ids, min_required_length)

        if not self._episode_refs:
            raise RuntimeError(
                "No episodes satisfy the length requirements for the chosen configuration."
            )

    def __len__(self) -> int:
        return len(self._episode_refs)

    def __getitem__(self, index: int) -> Dict[str, object]:
        query_ref = self._episode_refs[index]
        demo = self._load_demo(query_ref.demo_path)

        query_step = 0
        query_obs = demo[query_step]
        query_point_cloud = self._load_point_cloud(query_ref.episode_dir, query_step)
        query_proprio = self._build_vector(query_obs, self.proprio_keys).squeeze(0)
        query_gripper_pose = self._build_vector(query_obs, ("gripper_pose",)).squeeze(0)
        query_future = self._build_future_sequence(demo, query_step)

        support_refs = self._sample_support_refs(index)
        support_set = [self._build_support_item(ref) for ref in support_refs]

        return {
            "query": {
                "observation": {
                    "point_cloud": query_point_cloud,
                    "proprio": query_proprio,
                    "gripper_pose": query_gripper_pose,
                },
                "future_actions": query_future,
                "meta": {
                    "task": query_ref.task,
                    "variation": query_ref.variation,
                    "episode": query_ref.episode,
                    "initial_step": query_step,
                },
            },
            "support": support_set,
        }

    # ------------------------------------------------------------------
    # Index construction ------------------------------------------------

    def _build_episode_index(
        self,
        variation_ids: Sequence[int],
        min_required_length: int,
    ) -> None:
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
                demo = self._load_demo(demo_path)
                length = len(demo)
                if length < min_required_length:
                    continue
                ref = _EpisodeRef(
                    task=self.task,
                    variation=variation,
                    episode=int(episode_dir.name.replace("episode", "")),
                    demo_path=demo_path,
                    episode_dir=episode_dir,
                    length=length,
                )
                idx = len(self._episode_refs)
                self._episode_refs.append(ref)
                self._episodes_by_variation.setdefault(variation, []).append(idx)

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
    # Sampling helpers --------------------------------------------------

    def _sample_support_refs(self, query_index: int) -> List[_EpisodeRef]:
        min_support, max_support = self.support_size_range
        count = self._rng.randint(min_support, max_support)

        query_ref = self._episode_refs[query_index]
        if self.match_variation:
            candidate_indices = [
                idx for idx in self._episodes_by_variation.get(query_ref.variation, [])
                if idx != query_index
            ]
        else:
            candidate_indices = [idx for idx in range(len(self._episode_refs)) if idx != query_index]

        if not candidate_indices:
            raise RuntimeError("Unable to sample support episodes - not enough eligible episodes available.")

        actual_count = min(count, len(candidate_indices))
        selected_indices = self._rng.sample(candidate_indices, actual_count)
        return [self._episode_refs[idx] for idx in selected_indices]

    def _build_support_item(self, ref: _EpisodeRef) -> Dict[str, object]:
        demo = self._load_demo(ref.demo_path)
        indices = self._subsample_indices(ref.length)

        point_clouds = []
        proprio_vectors = []
        gripper_poses = []
        for idx in indices:
            obs = demo[idx]
            point_clouds.append(self._load_point_cloud(ref.episode_dir, idx))
            proprio_vectors.append(self._build_vector(obs, self.proprio_keys).squeeze(0))
            gripper_poses.append(self._build_vector(obs, ("gripper_pose",)).squeeze(0))

        return {
            "point_cloud_sequence": point_clouds,
            "proprio_sequence": torch.stack(proprio_vectors, dim=0),
            "gripper_pose_sequence": torch.stack(gripper_poses, dim=0),
            "meta": {
                "task": ref.task,
                "variation": ref.variation,
                "episode": ref.episode,
                "indices": indices,
            },
        }

    def _subsample_indices(self, length: int) -> List[int]:
        return list(np.linspace(0, length - 1, self.support_frames, dtype=int))

    def _build_future_sequence(self, demo, start_step: int) -> torch.Tensor:
        indices = [start_step + self.query_future_stride * (i + 1) for i in range(self.query_future_steps)]
        vectors = [self._build_vector(demo[idx], self.action_keys).squeeze(0) for idx in indices]
        return torch.stack(vectors, dim=0)

    # ------------------------------------------------------------------
    # Loading utilities -------------------------------------------------

    def _load_demo(self, path: Path):
        key = str(path)
        demo = self._demo_cache.get(key)
        if demo is None:
            with path.open("rb") as handle:
                demo = pickle.load(handle)
            if self.cache_size > 0:
                if len(self._demo_cache) >= self.cache_size:
                    self._demo_cache.popitem(last=False)
                self._demo_cache[key] = demo
        elif self.cache_size > 0:
            self._demo_cache.move_to_end(key)
        return demo

    def _load_point_cloud(self, episode_dir: Path, step: int) -> Dict[str, torch.Tensor]:
        npz_path = episode_dir / self.MERGED_POINT_CLOUD_FOLDER / f"{step}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing point cloud file: {npz_path}")
        with np.load(npz_path) as data:
            points = torch.from_numpy(data["points"].astype(np.float32))
            colors = torch.from_numpy(data["colors"].astype(np.float32)) / 255.0
            raw_masks = torch.from_numpy(data["masks"].astype(np.int64))
        ignore_ids = self._get_mask_ignore_ids(episode_dir)
        masks = self._process_masks(raw_masks, ignore_ids)
        cloud = {"points": points, "colors": colors, "masks": masks}
        if self.point_cloud_transform is not None:
            cloud = self.point_cloud_transform(cloud)
        return cloud

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

    def _build_vector(self, obs, keys: Sequence[str]) -> torch.Tensor:
        vector_parts: List[np.ndarray] = []
        for key in keys:
            value = getattr(obs, key)
            if value is None:
                raise ValueError(f"Observation attribute '{key}' is None")
            array = np.asarray(value, dtype=np.float32)
            vector_parts.append(array.reshape(1, -1))
        return torch.from_numpy(np.concatenate(vector_parts, axis=1))


def rlbench_in_context_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Collate helper that keeps per-sample support sets ragged."""

    queries = {
        "point_cloud": [item["query"]["observation"]["point_cloud"] for item in batch],
        "proprio": torch.stack([item["query"]["observation"]["proprio"] for item in batch], dim=0),
        "gripper_pose": torch.stack([item["query"]["observation"]["gripper_pose"] for item in batch], dim=0),
    }
    futures = torch.stack([item["query"]["future_actions"] for item in batch], dim=0)
    query_meta = {
        key: torch.tensor([item["query"]["meta"][key] for item in batch], dtype=torch.long)
        for key in ("variation", "episode", "initial_step")
    }
    query_meta["task"] = [item["query"]["meta"]["task"] for item in batch]

    support_sets = [item["support"] for item in batch]

    return {
        "query": {
            "observation": queries,
            "future_actions": futures,
            "meta": query_meta,
        },
        "support": support_sets,
    }


def create_in_context_dataloader(
    dataset: RLBenchInContextDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Instantiate a DataLoader with the in-context collate helper."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=rlbench_in_context_collate,
    )


__all__ = [
    "RLBenchInContextDataset",
    "create_in_context_dataloader",
    "rlbench_in_context_collate",
]
