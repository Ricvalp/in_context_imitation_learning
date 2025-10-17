"""Policy modules for diffusion-based imitation learning."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "DiffusionPolicy",
    "DiffusionPolicyConfig",
    "PlatonicDiffusionPolicy",
    "PlatonicDiffusionPolicyConfig",
    "InContextDiffusionPolicy",
    "InContextDiffusionPolicyConfig",
    "InContextPlatonicDiffusionPolicy",
    "InContextPlatonicPolicyConfig",
]

_POLICY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "DiffusionPolicy": ("equi_icil.policies.diffusion_policy", "DiffusionPolicy"),
    "DiffusionPolicyConfig": (
        "equi_icil.policies.diffusion_policy",
        "DiffusionPolicyConfig",
    ),
    "PlatonicDiffusionPolicy": (
        "equi_icil.policies.platonic_policy",
        "PlatonicDiffusionPolicy",
    ),
    "PlatonicDiffusionPolicyConfig": (
        "equi_icil.policies.platonic_policy",
        "PlatonicDiffusionPolicyConfig",
    ),
    "InContextDiffusionPolicy": (
        "equi_icil.policies.in_context_diffusion_policy",
        "InContextDiffusionPolicy",
    ),
    "InContextDiffusionPolicyConfig": (
        "equi_icil.policies.in_context_diffusion_policy",
        "InContextDiffusionPolicyConfig",
    ),
    "InContextPlatonicDiffusionPolicy": (
        "equi_icil.policies.in_context_platonic_policy",
        "InContextPlatonicDiffusionPolicy",
    ),
    "InContextPlatonicPolicyConfig": (
        "equi_icil.policies.in_context_platonic_policy",
        "InContextPlatonicPolicyConfig",
    ),
}

_CACHE: Dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    if name not in _POLICY_EXPORTS:
        raise AttributeError(f"module 'equi_icil.policies' has no attribute '{name}'")

    module_name, attr_name = _POLICY_EXPORTS[name]
    module = _CACHE.get(module_name)
    if module is None:
        module = import_module(module_name)
        _CACHE[module_name] = module
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
