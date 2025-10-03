"""Dataset-driven linear normalisation helpers."""

from __future__ import annotations

from typing import Dict, Union

import torch
import torch.nn as nn

def _dict_apply(data: Dict[str, torch.Tensor], fn):
    return {key: fn(key, value) for key, value in data.items()}


class DictOfTensorMixin(nn.Module):
    """Lightweight mixin that holds a nested ParameterDict."""

    def __init__(self, params_dict: nn.ParameterDict | None = None) -> None:
        super().__init__()
        self.params_dict = params_dict or nn.ParameterDict()

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
        params = nn.ParameterDict()
        prefix = prefix + "params_dict"
        for name, value in list(state_dict.items()):
            if not name.startswith(prefix):
                continue
            key = name[len(prefix) + 1 :]
            parts = key.split(".")
            container = params
            for part in parts[:-1]:
                if part not in container:
                    container[part] = nn.ParameterDict()
                container = container[part]  # type: ignore[assignment]
            leaf = parts[-1]
            container[leaf] = value.clone()
            del state_dict[name]
        self.params_dict = params
        self.params_dict.requires_grad_(False)


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    """Normalises a single tensor field via an affine transform."""

    def __init__(self, params_dict: nn.ParameterDict | None = None) -> None:
        super().__init__(params_dict=params_dict)

    @classmethod
    def create_manual(
        cls,
        scale: torch.Tensor,
        offset: torch.Tensor,
        input_stats: Dict[str, torch.Tensor],
    ) -> "SingleFieldLinearNormalizer":
        params = nn.ParameterDict(
            {
                "scale": nn.Parameter(scale.clone(), requires_grad=False),
                "offset": nn.Parameter(offset.clone(), requires_grad=False),
                "input_stats": nn.ParameterDict(
                    {name: nn.Parameter(value.clone(), requires_grad=False) for name, value in input_stats.items()}
                ),
            }
        )
        return cls(params)

    @classmethod
    def create_identity(cls, dim: int, dtype: torch.dtype = torch.float32) -> "SingleFieldLinearNormalizer":
        scale = torch.ones(dim, dtype=dtype)
        offset = torch.zeros(dim, dtype=dtype)
        stats = {
            "min": torch.full((dim,), -1.0, dtype=dtype),
            "max": torch.full((dim,), 1.0, dtype=dtype),
            "mean": torch.zeros(dim, dtype=dtype),
            "std": torch.ones(dim, dtype=dtype),
        }
        return cls.create_manual(scale, offset, stats)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        params = self.params_dict
        scale = params["scale"]
        offset = params["offset"]
        flat = tensor.reshape(-1, scale.shape[0])
        flat = flat * scale + offset
        return flat.reshape_as(tensor)

    def unnormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        params = self.params_dict
        scale = params["scale"]
        offset = params["offset"]
        flat = tensor.reshape(-1, scale.shape[0])
        flat = (flat - offset) / scale
        return flat.reshape_as(tensor)

    def get_input_stats(self) -> Dict[str, torch.Tensor]:
        stats = self.params_dict["input_stats"]
        return {name: value.clone() for name, value in stats.items()}

    def get_output_stats(self) -> Dict[str, torch.Tensor]:
        return {name: self.normalize(value) for name, value in self.params_dict["input_stats"].items()}

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # pragma: no cover - nn.Module API
        return self.normalize(tensor)


class LinearNormalizer(DictOfTensorMixin):
    """Dictionary-aware collection of affine normalisers."""

    def __getitem__(self, key: str) -> SingleFieldLinearNormalizer:
        if key not in self.params_dict:
            raise KeyError(key)
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: SingleFieldLinearNormalizer) -> None:
        self.params_dict[key] = value.params_dict

    def normalize(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        if isinstance(x, dict):
            return _dict_apply(x, self._normalize_key)
        return self._normalize_default(x)

    def _normalize_key(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if key not in self.params_dict:
            raise KeyError(f"Normalizer key '{key}' missing")
        return SingleFieldLinearNormalizer(self.params_dict[key]).normalize(value)

    def _normalize_default(self, value: torch.Tensor) -> torch.Tensor:
        if "_default" not in self.params_dict:
            raise RuntimeError("Normalizer has no default entry")
        return SingleFieldLinearNormalizer(self.params_dict["_default"]).normalize(value)

    def unnormalize(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        if isinstance(x, dict):
            return {key: self[key].unnormalize(value) for key, value in x.items()}
        return self._unnormalize_default(x)

    def _unnormalize_default(self, value: torch.Tensor) -> torch.Tensor:
        if "_default" not in self.params_dict:
            raise RuntimeError("Normalizer has no default entry")
        return SingleFieldLinearNormalizer(self.params_dict["_default"]).unnormalize(value)

    def get_input_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {key: self[key].get_input_stats() for key in self.params_dict.keys()}

    def get_output_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {key: self[key].get_output_stats() for key in self.params_dict.keys()}


__all__ = ["LinearNormalizer", "SingleFieldLinearNormalizer"]
