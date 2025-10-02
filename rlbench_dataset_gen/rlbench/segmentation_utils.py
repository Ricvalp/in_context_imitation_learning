"""Utilities for generating RLBench segmentation mask label mappings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import numpy as np

try:  # pragma: no cover - optional low-level fallback
    from pyrep.backend import sim as pr_sim
except Exception:  # pragma: no cover
    pr_sim = None

from pyrep.const import ObjectType

DEFAULT_MAP_FILENAME = "mask_to_label.json"


def collect_mask_handles(variation_dir: Path) -> Set[int]:
    """Return all non-background mask handles found in a variation directory."""
    handles: Set[int] = set()
    pattern = variation_dir.glob("episodes/episode*/merged_point_cloud/*.npz")
    for npz_path in pattern:
        try:
            with np.load(npz_path) as data:
                masks = data["masks"].astype(np.int64, copy=False)
        except (IOError, KeyError, ValueError):
            continue
        uniques = np.unique(masks)
        handles.update(int(v) for v in uniques if v != 0)
    return handles


def build_handle_label_map(task_env, handles: Iterable[int]) -> Tuple[Dict[int, str], Set[int]]:
    """Resolve object handles to names using the active task environment."""
    outstanding: Set[int] = set(int(h) for h in handles)
    mapping: Dict[int, str] = {}

    if not outstanding:
        return mapping, outstanding

    scene = task_env._scene  # pylint: disable=protected-access
    task = scene.task
    if task is not None:
        base = task.get_base()
        for obj in base.get_objects_in_tree(exclude_base=False):
            handle = int(obj.get_handle())
            if handle in outstanding:
                mapping[handle] = obj.get_name()
                outstanding.remove(handle)
            if not outstanding:
                break

    if outstanding:
        for obj in scene.pyrep.get_objects_in_tree(
            object_type=ObjectType.SHAPE, exclude_base=False
        ):
            handle = int(obj.get_handle())
            if handle in outstanding:
                mapping[handle] = obj.get_name()
                outstanding.remove(handle)
            if not outstanding:
                break

    if outstanding and pr_sim is not None:  # pragma: no cover
        for handle in list(outstanding):
            try:
                name = pr_sim.simGetObjectName(handle)
            except Exception:
                continue
            if name:
                mapping[handle] = name
                outstanding.remove(handle)

    return mapping, outstanding


def write_label_map(path: Path, mapping: Dict[int, str], overwrite: bool = True) -> None:
    """Serialise the mapping as JSON, optionally skipping if the file exists."""
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {str(k): v for k, v in sorted(mapping.items())}
    with path.open("w", encoding="utf-8") as file:
        json.dump(serialisable, file, indent=2, sort_keys=True)
        file.write("\n")

