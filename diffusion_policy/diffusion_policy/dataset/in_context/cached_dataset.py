"""HDF5-backed in-context dataset loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from tools.datasets.common import read_point_cloud, read_point_cloud_sequence

from .cache import IN_CONTEXT_CACHE_VERSION
from .dataset import PointCloudTransform


class InContextCachedDataset(Dataset):
    """Dataset that serves precomputed in-context samples from a cache file."""

    def __init__(
        self,
        cache_path: Path | str,
        *,
        point_cloud_transform: Optional[PointCloudTransform] = None,
    ) -> None:
        super().__init__()
        self.cache_path = Path(cache_path).expanduser().resolve()
        if not self.cache_path.is_file():
            raise FileNotFoundError(f"Cache file not found: {self.cache_path}")

        self._user_transform = point_cloud_transform
        with h5py.File(self.cache_path, "r") as handle:
            version = int(handle.attrs.get("cache_version", -1))
            if version != IN_CONTEXT_CACHE_VERSION:
                raise RuntimeError(
                    f"Cache version mismatch ({version} != {IN_CONTEXT_CACHE_VERSION})."
                )
            self._length = int(handle.attrs["length"])
            config_json = handle.attrs.get("config")
            self.config = json.loads(config_json) if config_json else None
            self.variation_metadata = self._load_variation_metadata(handle)

        self._h5: h5py.File | None = None

    def __len__(self) -> int:  # type: ignore[override]
        return self._length

    def _get_file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.cache_path, "r")
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):
        if getattr(self, "_h5", None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None

    def __getitem__(self, index: int) -> Dict[str, object]:  # type: ignore[override]
        if index < 0 or index >= self._length:
            raise IndexError(index)

        handle = self._get_file()
        sample_grp = handle["samples"][str(index)]

        query_grp = sample_grp["query"]
        q_obs_grp = query_grp["observation"]
        query_cloud = read_point_cloud(q_obs_grp["point_cloud"])
        if self._user_transform is not None:
            query_cloud = self._user_transform(dict(query_cloud))
        query_proprio = torch.from_numpy(q_obs_grp["proprio"][()]).to(torch.float32)
        query_gripper = torch.from_numpy(q_obs_grp["gripper_pose"][()]).to(torch.float32)
        future_actions = torch.from_numpy(query_grp["future_actions"][()]).to(torch.float32)

        q_meta_grp = query_grp["meta"]
        q_task = q_meta_grp.attrs["task"]
        if isinstance(q_task, bytes):
            q_task = q_task.decode("utf-8")
        query_meta = {
            "task": q_task,
            "variation": int(q_meta_grp.attrs["variation"]),
            "episode": int(q_meta_grp.attrs["episode"]),
            "initial_step": int(q_meta_grp.attrs["initial_step"]),
        }

        support_grp = sample_grp["support"]
        support_len = int(support_grp.attrs.get("length", len(support_grp)))
        support_items: List[Dict[str, object]] = []
        for sup_idx in range(support_len):
            sup_grp = support_grp[str(sup_idx)]
            pc_sequence = read_point_cloud_sequence(sup_grp["point_cloud_sequence"])
            if self._user_transform is not None:
                pc_sequence = [self._user_transform(dict(cloud)) for cloud in pc_sequence]
            proprio_seq = torch.from_numpy(sup_grp["proprio_sequence"][()]).to(torch.float32)
            gripper_seq = torch.from_numpy(sup_grp["gripper_pose_sequence"][()]).to(torch.float32)
            sup_meta_grp = sup_grp["meta"]
            s_task = sup_meta_grp.attrs["task"]
            if isinstance(s_task, bytes):
                s_task = s_task.decode("utf-8")
            support_items.append(
                {
                    "point_cloud_sequence": pc_sequence,
                    "proprio_sequence": proprio_seq,
                    "gripper_pose_sequence": gripper_seq,
                    "meta": {
                        "task": s_task,
                        "variation": int(sup_meta_grp.attrs["variation"]),
                        "episode": int(sup_meta_grp.attrs["episode"]),
                        "indices": sup_meta_grp["indices"][()].astype(np.int64).tolist(),
                    },
                }
            )

        return {
            "query": {
                "observation": {
                    "point_cloud": query_cloud,
                    "proprio": query_proprio,
                    "gripper_pose": query_gripper,
                },
                "future_actions": future_actions,
                "meta": query_meta,
            },
            "support": support_items,
        }

    @staticmethod
    def _load_variation_metadata(handle: h5py.File) -> Dict[int, Dict[str, object]]:
        metadata: Dict[int, Dict[str, object]] = {}
        if "variation_metadata" not in handle:
            return metadata
        group = handle["variation_metadata"]
        for key in group:
            var_grp = group[key]
            variation = int(var_grp.attrs.get("variation", int(key)))
            entry: Dict[str, object] = {
                "variation": variation,
            }
            if "path" in var_grp.attrs:
                entry["path"] = var_grp.attrs["path"]
            if "descriptions" in var_grp:
                raw_desc = var_grp["descriptions"][()]
                entry["descriptions"] = [d if isinstance(d, str) else d.decode("utf-8") for d in raw_desc]
            if "mask_handles" in var_grp and "mask_names" in var_grp:
                handles = var_grp["mask_handles"][()].astype(int).tolist()
                raw_names = var_grp["mask_names"][()]
                names = [n if isinstance(n, str) else n.decode("utf-8") for n in raw_names]
                entry["mask_to_label"] = dict(zip(handles, names))
            metadata[variation] = entry
        return metadata


__all__ = ["InContextCachedDataset"]
