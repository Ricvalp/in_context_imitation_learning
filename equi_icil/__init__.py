"""Lightweight 3D diffusion policy training stack with lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "config",
    "platonic_config",
    "in_context_diffusion_config",
    "DiffusionPolicy",
    "DiffusionPolicyConfig",
    "PlatonicDiffusionPolicy",
    "PlatonicDiffusionPolicyConfig",
    "InContextDiffusionPolicy",
    "InContextDiffusionPolicyConfig",
    "InContextPlatonicDiffusionPolicy",
    "InContextPlatonicPolicyConfig",
    "DatasetConfig",
    "SparseDatasetConfig",
    "RLBenchTemporalH5Dataset",
    "RLBenchTemporalH5SparseDataset",
    "collate_temporal_batch",
    "collate_sparse_temporal_batch",
    "LinearNormalizer",
    "SingleFieldLinearNormalizer",
    "CheckpointManager",
    "ExponentialMovingAverage",
    "log_pointcloud_wandb",
    "mse",
    "set_seed",
    "visualize_trajectories",
]

_MODULE_EXPORTS: Dict[str, str] = {
    "config": "equi_icil.config",
    "platonic_config": "equi_icil.platonic_config",
    "in_context_diffusion_config": "equi_icil.in_context_diffusion_config",
    "DatasetConfig": "equi_icil.imitation_datasets",
    "SparseDatasetConfig": "equi_icil.imitation_datasets",
    "RLBenchTemporalH5Dataset": "equi_icil.imitation_datasets",
    "RLBenchTemporalH5SparseDataset": "equi_icil.imitation_datasets",
    "collate_temporal_batch": "equi_icil.imitation_datasets",
    "collate_sparse_temporal_batch": "equi_icil.imitation_datasets",
    "LinearNormalizer": "equi_icil.utils",
    "SingleFieldLinearNormalizer": "equi_icil.utils",
    "CheckpointManager": "equi_icil.utils",
    "ExponentialMovingAverage": "equi_icil.utils",
    "log_pointcloud_wandb": "equi_icil.utils",
    "mse": "equi_icil.utils",
    "set_seed": "equi_icil.utils",
    "visualize_trajectories": "equi_icil.utils",
    "DiffusionPolicy": "equi_icil.policies",
    "DiffusionPolicyConfig": "equi_icil.policies",
    "PlatonicDiffusionPolicy": "equi_icil.policies",
    "PlatonicDiffusionPolicyConfig": "equi_icil.policies",
    "InContextDiffusionPolicy": "equi_icil.policies",
    "InContextDiffusionPolicyConfig": "equi_icil.policies",
    "InContextPlatonicDiffusionPolicy": "equi_icil.policies",
    "InContextPlatonicPolicyConfig": "equi_icil.policies",
}

_MODULE_CACHE: Dict[str, Any] = {}


def _load_attribute(name: str) -> Tuple[Any, bool]:
    module_name = _MODULE_EXPORTS[name]
    module = _MODULE_CACHE.get(module_name)
    if module is None:
        module = import_module(module_name)
        _MODULE_CACHE[module_name] = module
    return getattr(module, name, module), hasattr(module, name)


def __getattr__(name: str) -> Any:
    if name not in _MODULE_EXPORTS:
        raise AttributeError(f"module 'equi_icil' has no attribute '{name}'")

    value, has_attr = _load_attribute(name)
    if not has_attr:
        globals()[name] = value
        return value

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
