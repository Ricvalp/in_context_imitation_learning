"""Reusable model components for policies."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "ObservationEncoder",
    "PointNetEncoder",
    "ConditionalUNet1D",
    "PlatonicTransformer",
]

_MODEL_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ObservationEncoder": ("equi_icil.models.pointnet", "ObservationEncoder"),
    "PointNetEncoder": ("equi_icil.models.pointnet", "PointNetEncoder"),
    "ConditionalUNet1D": ("equi_icil.models.unet1d", "ConditionalUNet1D"),
    "PlatonicTransformer": (
        "equi_icil.models.platonic_transformer",
        "PlatonicTransformer",
    ),
}

_CACHE: Dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    if name not in _MODEL_EXPORTS:
        raise AttributeError(f"module 'equi_icil.models' has no attribute '{name}'")

    module_name, attr_name = _MODEL_EXPORTS[name]
    module = _CACHE.get(module_name)
    if module is None:
        module = import_module(module_name)
        _CACHE[module_name] = module
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
