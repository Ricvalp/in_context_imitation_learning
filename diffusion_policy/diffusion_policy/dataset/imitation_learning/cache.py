"""Caching helpers for the RLBench temporal point-cloud dataset."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from diffusion_policy.dataset.common import (
    config_to_key,
    write_point_cloud_sequence,
)

TEMPORAL_CACHE_VERSION = 1


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
    from .dataset import RLBenchTemporalPointCloudDataset  # local import to avoid cycles

    return {
        "cache_version": TEMPORAL_CACHE_VERSION,
        "type": "temporal",
        "root": str(Path(dataset_kwargs.get("root", ".")).resolve()),
        "task": dataset_kwargs.get("task"),
        "variations": _canonicalise_variations_key(dataset_kwargs.get("variations")),
        "point_cloud_history": dataset_kwargs.get("point_cloud_history", 1),
        "proprio_history": dataset_kwargs.get("proprio_history", 1),
        "future_action_window": dataset_kwargs.get("future_action_window", 4),
        "future_action_stride": dataset_kwargs.get("future_action_stride", 1),
        "proprio_keys": list(dataset_kwargs.get("proprio_keys", ("gripper_pose", "gripper_open"))),
        "action_keys": list(dataset_kwargs.get("action_keys", ("gripper_pose", "gripper_open"))),
        "mask_names_to_ignore": RLBenchTemporalPointCloudDataset.MASK_NAMES_TO_IGNORE,
        "mask_label_map_filename": RLBenchTemporalPointCloudDataset.MASK_LABEL_MAP_FILENAME,
        "default_mask_ids": list(RLBenchTemporalPointCloudDataset.DEFAULT_MASK_IDS_TO_IGNORE),
    }


STRING_DTYPE = h5py.string_dtype("utf-8")


def _collect_variation_metadata(dataset) -> Dict[int, Dict[str, object]]:
    info: Dict[int, Dict[str, object]] = {}
    for entry in getattr(dataset, "_indices", []):
        variation = entry.variation
        if variation in info:
            continue
        variation_dir = entry.episode_dir.parent.parent
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
                    # ensure numeric keys and string values
                    handles = []
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


def _write_temporal_cache(
    dataset,
    cache_path: Path,
    config: Dict[str, object],
    *,
    progress: bool = True,
) -> None:
    _ensure_parent(cache_path)
    total = len(dataset)
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["cache_version"] = TEMPORAL_CACHE_VERSION
        handle.attrs["length"] = total
        handle.attrs["config"] = json.dumps(config, sort_keys=True)
        variation_metadata = _collect_variation_metadata(dataset)
        _write_variation_metadata(handle.create_group("variation_metadata"), variation_metadata)
        samples_grp = handle.create_group("samples")

        for idx in range(total):
            sample = dataset[idx]
            sample_grp = samples_grp.create_group(str(idx))

            obs_grp = sample_grp.create_group("observation")
            pc_grp = obs_grp.create_group("point_cloud_sequence")
            write_point_cloud_sequence(pc_grp, sample["observation"]["point_cloud_sequence"])
            obs_grp.create_dataset(
                "proprio_sequence",
                data=sample["observation"]["proprio_sequence"].numpy(),
                compression="gzip",
                compression_opts=4,
            )

            action_grp = sample_grp.create_group("action")
            action_grp.create_dataset(
                "sequence",
                data=sample["action"].numpy(),
                compression="gzip",
                compression_opts=4,
            )

            meta_grp = sample_grp.create_group("meta")
            meta = sample["meta"]
            meta_grp.attrs["task"] = meta["task"]
            meta_grp.attrs["variation"] = int(meta["variation"])
            meta_grp.attrs["episode"] = int(meta["episode"])
            meta_grp.attrs["step"] = int(meta["step"])

            if progress and idx % 100 == 0:
                print(f"[temporal-cache] processed {idx}/{total}")


def prepare_temporal_point_cloud_cache(
    *,
    cache_dir: Path | str | None = None,
    rebuild: bool = False,
    progress: bool = True,
    **dataset_kwargs,
) -> Path:
    """Generate or locate a cached temporal dataset on disk."""

    from .dataset import RLBenchTemporalPointCloudDataset  # local import to avoid cycles

    kwargs = dict(dataset_kwargs)
    kwargs.pop("point_cloud_transform", None)

    root = Path(kwargs.get("root", ".")).resolve()
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir(root)
    cache_dir = cache_dir.resolve()

    config = _build_cache_config(kwargs)
    cache_key = config_to_key(config)
    cache_path = cache_dir / "temporal" / kwargs.get("task", "unknown") / f"{cache_key}.h5"

    needs_build = rebuild or not cache_path.is_file()
    if needs_build:
        print(f"[temporal-cache] building cache at {cache_path}")
        dataset = RLBenchTemporalPointCloudDataset(point_cloud_transform=None, **kwargs)
        _write_temporal_cache(dataset, cache_path, config, progress=progress)
    else:
        with h5py.File(cache_path, "r") as handle:
            version = int(handle.attrs.get("cache_version", -1))
            if version != TEMPORAL_CACHE_VERSION:
                raise RuntimeError(
                    f"Cache version mismatch ({version} != {TEMPORAL_CACHE_VERSION}). "
                    "Pass rebuild=True to regenerate."
                )
    return cache_path


__all__ = [
    "TEMPORAL_CACHE_VERSION",
    "prepare_temporal_point_cloud_cache",
]
