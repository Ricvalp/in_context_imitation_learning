"""Dataset-driven linear normalisation helpers."""

from __future__ import annotations

from typing import Dict, Union

import torch
import torch.nn as nn


def _dict_apply(data: Dict[str, torch.Tensor], fn):
    return {key: fn(key, value) for key, value in data.items()}


class _StatsModule(nn.Module):
    def __init__(self, stats: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        for name, tensor in stats.items():
            self.register_buffer(name, tensor.clone())

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {name: buf.clone() for name, buf in self._buffers.items()}


class SingleFieldLinearNormalizer(nn.Module):
    """Normalises a single tensor field via an affine transform."""

    def __init__(
        self,
        scale: torch.Tensor,
        offset: torch.Tensor,
        input_stats: Dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        if scale.ndim != 1:
            raise ValueError("scale must be a 1D tensor")
        if offset.shape != scale.shape:
            raise ValueError("offset must have the same shape as scale")
        self.register_buffer("scale", scale.clone())
        self.register_buffer("offset", offset.clone())
        self.input_stats = _StatsModule(input_stats)

    @classmethod
    def create_manual(
        cls,
        scale: torch.Tensor,
        offset: torch.Tensor,
        input_stats: Dict[str, torch.Tensor],
    ) -> "SingleFieldLinearNormalizer":
        return cls(scale, offset, input_stats)

    @classmethod
    def create_identity(
        cls,
        dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> "SingleFieldLinearNormalizer":
        scale = torch.ones(dim, dtype=dtype)
        offset = torch.zeros(dim, dtype=dtype)
        stats = {
            "min": torch.full((dim,), -1.0, dtype=dtype),
            "max": torch.full((dim,), 1.0, dtype=dtype),
            "mean": torch.zeros(dim, dtype=dtype),
            "std": torch.ones(dim, dtype=dtype),
        }
        return cls(scale, offset, stats)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        dim = self.scale.shape[0]
        flat = tensor.reshape(-1, dim)
        flat = flat * self.scale + self.offset
        return flat.reshape_as(tensor)

    def unnormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        dim = self.scale.shape[0]
        flat = tensor.reshape(-1, dim)
        flat = (flat - self.offset) / self.scale
        return flat.reshape_as(tensor)

    def get_input_stats(self) -> Dict[str, torch.Tensor]:
        return self.input_stats.as_dict()

    def get_output_stats(self) -> Dict[str, torch.Tensor]:
        return {name: self.normalize(value) for name, value in self.input_stats.as_dict().items()}

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # pragma: no cover - nn.Module API
        return self.normalize(tensor)


class LinearNormalizer(nn.Module):
    """Dictionary-aware collection of affine normalisers."""

    def __init__(self) -> None:
        super().__init__()
        self.params_dict = nn.ModuleDict()

    def __getitem__(self, key: str) -> SingleFieldLinearNormalizer:
        if key not in self.params_dict:
            raise KeyError(key)
        return self.params_dict[key]

    def __setitem__(self, key: str, value: SingleFieldLinearNormalizer) -> None:
        if not isinstance(value, SingleFieldLinearNormalizer):
            raise TypeError("value must be a SingleFieldLinearNormalizer")
        self.params_dict[key] = value

    def normalize(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        if isinstance(x, dict):
            return _dict_apply(x, self._normalize_key)
        return self._normalize_default(x)

    def _normalize_key(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if key not in self.params_dict:
            raise KeyError(f"Normalizer key '{key}' missing")
        return self.params_dict[key].normalize(value)

    def _normalize_default(self, value: torch.Tensor) -> torch.Tensor:
        if "_default" not in self.params_dict:
            raise RuntimeError("Normalizer has no default entry")
        return self.params_dict["_default"].normalize(value)

    def unnormalize(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        if isinstance(x, dict):
            return {key: self[key].unnormalize(value) for key, value in x.items()}
        return self._unnormalize_default(x)

    def _unnormalize_default(self, value: torch.Tensor) -> torch.Tensor:
        if "_default" not in self.params_dict:
            raise RuntimeError("Normalizer has no default entry")
        return self.params_dict["_default"].unnormalize(value)

    def get_input_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {key: normalizer.get_input_stats() for key, normalizer in self.params_dict.items()}

    def get_output_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {key: normalizer.get_output_stats() for key, normalizer in self.params_dict.items()}

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        own_prefix = prefix + "params_dict."
        buckets: dict[str, dict[str, torch.Tensor | dict[str, torch.Tensor]]] = {}

        for key in list(state_dict.keys()):
            if not key.startswith(own_prefix):
                continue

            suffix = key[len(own_prefix):]
            parts = suffix.split(".")
            if len(parts) < 2:
                if strict:
                    unexpected_keys.append(key)
                # del state_dict[key]
                continue

            field = parts[0]
            entry = buckets.setdefault(field, {"input_stats": {}})
            slot = parts[1]

            if slot in ("scale", "offset"):
                entry[slot] = state_dict[key].clone()
            elif slot == "input_stats" and len(parts) == 3:
                stat_name = parts[2]
                entry["input_stats"][stat_name] = state_dict[key].clone()
            else:
                if strict:
                    unexpected_keys.append(key)
            # del state_dict[key]

        for field, payload in buckets.items():
            scale = payload.get("scale")
            offset = payload.get("offset")
            stats = payload.get("input_stats", {})
            if scale is None or offset is None:
                if strict:
                    missing_keys.append(f"{own_prefix}{field}.scale")
                    missing_keys.append(f"{own_prefix}{field}.offset")
                continue
            self.params_dict[field] = SingleFieldLinearNormalizer.create_manual(
                scale,
                offset,
                stats,  # type: ignore[arg-type]
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

__all__ = ["LinearNormalizer", "SingleFieldLinearNormalizer"]
