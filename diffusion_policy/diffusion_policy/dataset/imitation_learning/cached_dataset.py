"""HDF5-backed temporal dataset loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from diffusion_policy.dataset.common import read_point_cloud_sequence
from diffusion_policy.model.common.normalizer import LinearNormalizer


from .cache import TEMPORAL_CACHE_VERSION
from .dataset import PointCloudTransform


class TemporalPointCloudCachedDataset(Dataset):
    """Dataset that serves precomputed temporal samples from a cache file."""

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
            if version != TEMPORAL_CACHE_VERSION:
                raise RuntimeError(
                    f"Cache version mismatch ({version} != {TEMPORAL_CACHE_VERSION})."
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

        obs_grp = sample_grp["observation"]
        pc_sequence = read_point_cloud_sequence(obs_grp["point_cloud_sequence"])
        if self._user_transform is not None:
            pc_sequence = [self._user_transform(dict(cloud)) for cloud in pc_sequence]

        proprio = torch.from_numpy(obs_grp["proprio_sequence"][()]).to(torch.float32)
        action_seq = torch.from_numpy(sample_grp["action"]["sequence"][()]).to(torch.float32)

        meta_grp = sample_grp["meta"]
        task = meta_grp.attrs["task"]
        if isinstance(task, bytes):
            task = task.decode("utf-8")
        meta = {
            "task": task,
            "variation": int(meta_grp.attrs["variation"]),
            "episode": int(meta_grp.attrs["episode"]),
            "step": int(meta_grp.attrs["step"]),
        }

        return {
            "observation": {
                "point_cloud_sequence": pc_sequence,
                "proprio_sequence": proprio,
            },
            "action": action_seq,
            "meta": meta,
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
                entry["descriptions"] = [desc if isinstance(desc, str) else desc.decode("utf-8") for desc in var_grp["descriptions"][()]]
            if "mask_handles" in var_grp and "mask_names" in var_grp:
                handles = var_grp["mask_handles"][()].astype(int).tolist()
                raw_names = var_grp["mask_names"][()]
                names = [name if isinstance(name, str) else name.decode("utf-8") for name in raw_names]
                entry["mask_to_label"] = dict(zip(handles, names))
            metadata[variation] = entry
        return metadata


__all__ = ["TemporalPointCloudCachedDataset"]
