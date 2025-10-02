"""Caching helpers for the RLBench in-context dataset."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from tools.datasets.common import (
    config_to_key,
    write_point_cloud,
    write_point_cloud_sequence,
)

STRING_DTYPE = h5py.string_dtype("utf-8")

IN_CONTEXT_CACHE_VERSION = 1


def _canonicalise_variations_key(variations) -> Dict[str, object]:
    if variations is None:
        return {"mode": "all"}
    if isinstance(variations, tuple):
        if len(variations) != 2:
            raise ValueError("variation tuple must contain two integers")
        start, end = variations
        return {"mode": "range", "start": int(start), "end": int(end)}
    return {"mode": "explicit", "values": [int(v) for v in variations]}


def _default_cache_dir(root: Path) -> Path:
    return Path(root).resolve() / ".rlbench_cache"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_cache_config(dataset_kwargs: Dict[str, object]) -> Dict[str, object]:
    from .dataset import RLBenchInContextDataset  # local import to avoid cycles

    return {
        "cache_version": IN_CONTEXT_CACHE_VERSION,
        "type": "in_context",
        "root": str(Path(dataset_kwargs.get("root", ".")).resolve()),
        "task": dataset_kwargs.get("task"),
        "variations": _canonicalise_variations_key(dataset_kwargs.get("variations")),
        "support_size_range": list(dataset_kwargs.get("support_size_range", (1, 4))),
        "support_frames": dataset_kwargs.get("support_frames", 7),
        "query_future_steps": dataset_kwargs.get("query_future_steps", 7),
        "query_future_stride": dataset_kwargs.get("query_future_stride", 10),
        "proprio_keys": list(dataset_kwargs.get("proprio_keys", ("gripper_pose", "gripper_open"))),
        "action_keys": list(dataset_kwargs.get("action_keys", ("gripper_pose", "gripper_open"))),
        "match_variation": bool(dataset_kwargs.get("match_variation", False)),
        "mask_names_to_ignore": RLBenchInContextDataset.MASK_NAMES_TO_IGNORE,
        "mask_label_map_filename": RLBenchInContextDataset.MASK_LABEL_MAP_FILENAME,
        "default_mask_ids": list(RLBenchInContextDataset.DEFAULT_MASK_IDS_TO_IGNORE),
        "seed": dataset_kwargs.get("seed"),
    }


def _collect_variation_metadata(dataset) -> Dict[int, Dict[str, object]]:
    info: Dict[int, Dict[str, object]] = {}
    for ref in getattr(dataset, "_episode_refs", []):
        variation = ref.variation
        if variation in info:
            continue
        variation_dir = ref.episode_dir.parent.parent
        data: Dict[str, object] = {
            "path": str(variation_dir.resolve()),
        }

        desc_path = variation_dir / dataset.VARIATION_DESCRIPTIONS_FILE
        if desc_path.is_file():
            try:
                with desc_path.open("rb") as f:
                    descriptions = pickle.load(f)
                if isinstance(descriptions, list):
                    data["descriptions"] = [str(item) for item in descriptions]
            except Exception:
                pass

        mask_path = variation_dir / dataset.MASK_LABEL_MAP_FILENAME
        if mask_path.is_file():
            try:
                with mask_path.open("r", encoding="utf-8") as f:
                    mask_map = json.load(f)
                if isinstance(mask_map, dict):
                    handles: List[int] = []
                    names: List[str] = []
                    for handle, name in mask_map.items():
                        try:
                            handles.append(int(handle))
                            names.append(str(name))
                        except Exception:
                            continue
                    data["mask_handles"] = handles
                    data["mask_names"] = names
            except Exception:
                pass

        info[variation] = data
    return info


def _write_variation_metadata(group: h5py.Group, metadata: Dict[int, Dict[str, object]]) -> None:
    group.attrs["count"] = len(metadata)
    for variation, data in metadata.items():
        var_grp = group.create_group(str(variation))
        var_grp.attrs["variation"] = int(variation)
        if "path" in data:
            var_grp.attrs["path"] = data["path"]
        if "descriptions" in data:
            var_grp.create_dataset(
                "descriptions",
                data=np.asarray(data["descriptions"], dtype=STRING_DTYPE),
            )
        if "mask_handles" in data and "mask_names" in data:
            handles = np.asarray(data["mask_handles"], dtype=np.int64)
            names = np.asarray(data["mask_names"], dtype=STRING_DTYPE)
            var_grp.create_dataset("mask_handles", data=handles)
            var_grp.create_dataset("mask_names", data=names)


def _write_in_context_cache(
    dataset,
    cache_path: Path,
    config: Dict[str, object],
    *,
    progress: bool = True,
) -> None:
    _ensure_parent(cache_path)
    total = len(dataset)
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["cache_version"] = IN_CONTEXT_CACHE_VERSION
        handle.attrs["length"] = total
        handle.attrs["config"] = json.dumps(config, sort_keys=True)
        variation_metadata = _collect_variation_metadata(dataset)
        _write_variation_metadata(handle.create_group("variation_metadata"), variation_metadata)
        samples_grp = handle.create_group("samples")

        for idx in range(total):
            sample = dataset[idx]
            sample_grp = samples_grp.create_group(str(idx))

            query_grp = sample_grp.create_group("query")
            q_obs_grp = query_grp.create_group("observation")
            write_point_cloud(q_obs_grp.create_group("point_cloud"), sample["query"]["observation"]["point_cloud"])
            q_obs_grp.create_dataset(
                "proprio",
                data=sample["query"]["observation"]["proprio"].numpy(),
                compression="gzip",
                compression_opts=4,
            )
            q_obs_grp.create_dataset(
                "gripper_pose",
                data=sample["query"]["observation"]["gripper_pose"].numpy(),
                compression="gzip",
                compression_opts=4,
            )

            query_grp.create_dataset(
                "future_actions",
                data=sample["query"]["future_actions"].numpy(),
                compression="gzip",
                compression_opts=4,
            )

            q_meta_grp = query_grp.create_group("meta")
            q_meta = sample["query"]["meta"]
            q_meta_grp.attrs["task"] = q_meta["task"]
            q_meta_grp.attrs["variation"] = int(q_meta["variation"])
            q_meta_grp.attrs["episode"] = int(q_meta["episode"])
            q_meta_grp.attrs["initial_step"] = int(q_meta["initial_step"])

            support_grp = sample_grp.create_group("support")
            support_list = sample["support"]
            support_grp.attrs["length"] = len(support_list)
            for sup_idx, support in enumerate(support_list):
                sup_grp = support_grp.create_group(str(sup_idx))
                write_point_cloud_sequence(sup_grp.create_group("point_cloud_sequence"), support["point_cloud_sequence"])
                sup_grp.create_dataset(
                    "proprio_sequence",
                    data=support["proprio_sequence"].numpy(),
                    compression="gzip",
                    compression_opts=4,
                )
                sup_grp.create_dataset(
                    "gripper_pose_sequence",
                    data=support["gripper_pose_sequence"].numpy(),
                    compression="gzip",
                    compression_opts=4,
                )
                sup_meta_grp = sup_grp.create_group("meta")
                sup_meta = support["meta"]
                sup_meta_grp.attrs["task"] = sup_meta["task"]
                sup_meta_grp.attrs["variation"] = int(sup_meta["variation"])
                sup_meta_grp.attrs["episode"] = int(sup_meta["episode"])
                sup_meta_grp.create_dataset(
                    "indices",
                    data=np.asarray(sup_meta["indices"], dtype=np.int64),
                    compression="gzip",
                    compression_opts=4,
                )

            if progress and idx % 25 == 0:
                print(f"[in-context-cache] processed {idx}/{total}")


def prepare_in_context_point_cloud_cache(
    *,
    cache_dir: Path | str | None = None,
    rebuild: bool = False,
    progress: bool = True,
    **dataset_kwargs,
) -> Path:
    """Generate or locate a cached in-context dataset on disk."""

    from .dataset import RLBenchInContextDataset  # local import to avoid cycles

    kwargs = dict(dataset_kwargs)
    kwargs.pop("point_cloud_transform", None)

    root = Path(kwargs.get("root", ".")).resolve()
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir(root)
    cache_dir = cache_dir.resolve()

    config = _build_cache_config(kwargs)
    cache_key = config_to_key(config)
    cache_path = cache_dir / "in_context" / kwargs.get("task", "unknown") / f"{cache_key}.h5"

    needs_build = rebuild or not cache_path.is_file()
    if needs_build:
        print(f"[in-context-cache] building cache at {cache_path}")
        dataset = RLBenchInContextDataset(point_cloud_transform=None, **kwargs)
        _write_in_context_cache(dataset, cache_path, config, progress=progress)
    else:
        with h5py.File(cache_path, "r") as handle:
            version = int(handle.attrs.get("cache_version", -1))
            if version != IN_CONTEXT_CACHE_VERSION:
                raise RuntimeError(
                    f"Cache version mismatch ({version} != {IN_CONTEXT_CACHE_VERSION}). "
                    "Pass rebuild=True to regenerate."
                )
    return cache_path


__all__ = [
    "IN_CONTEXT_CACHE_VERSION",
    "prepare_in_context_point_cloud_cache",
]
