"""Evaluate the diffusion policy inside an RLBench simulation."""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
import wandb
from pyrep.const import ObjectType

# Ensure the repository root is on the path for relative imports and RLBench
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml_collections import ConfigDict

from equi_icil.config import get_config as get_default_config
from equi_icil.platonic_config import get_config as get_platonic_config
from equi_icil.policies import (
    DiffusionPolicy,
    DiffusionPolicyConfig,
    PlatonicDiffusionPolicy,
    PlatonicDiffusionPolicyConfig,
    InContextPlatonicDiffusionPolicy,
    InContextPlatonicPolicyConfig,
)
from equi_icil.models.platonic_transformer.groups import PLATONIC_GROUPS
from equi_icil.rlbench_integration import (
    CAMERA_NAMES,
    ObservationHistory,
    ObservationProcessor,
    SimulationInputConfig,
    action_plan_to_command,
    create_action_mode,
    create_observation_config,
    instantiate_environment,
    resolve_task_class,
)
from equi_icil.utils import log_pointcloud_wandb, set_seed


LOG_POINTCLOUD_INTERVAL = 20


def _load_config(path: str | None) -> ConfigDict:
    """Load the training configuration, optionally from a Python module file.

    Args:
        path: Optional path to a Python module defining ``get_config``.

    Returns:
        Configuration dictionary merged with any overrides from ``path``.
    """
    if path is None:
        return get_default_config()

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    module_dir = config_path.parent
    module_name = config_path.stem
    sys.path.insert(0, str(module_dir))
    module = __import__(module_name)
    if not hasattr(module, "get_config"):
        raise AttributeError(f"Config module {module_name} does not define get_config")
    loaded_cfg = module.get_config()
    if not isinstance(loaded_cfg, ConfigDict):
        raise TypeError("get_config() must return an ml_collections.ConfigDict")

    base_cfg = get_default_config()
    if getattr(loaded_cfg, "model", None) is not None and hasattr(
        loaded_cfg.model, "transformer"
    ):
        base_cfg = get_platonic_config()

    base_cfg.update(loaded_cfg)
    return base_cfg


def _build_diffusion_model(cfg: ConfigDict) -> DiffusionPolicy:
    """Construct the PointNet-based diffusion policy."""

    model_cfg = DiffusionPolicyConfig(
        horizon=cfg.horizon,
        n_obs_steps=cfg.n_obs_steps,
        action_dim=cfg.action_dim,
        agent_dim=cfg.agent_dim,
        sample_points=cfg.sample_points,
        use_point_colors=cfg.use_point_colors,
        pointnet_in_channels=cfg.model.pointnet.in_channels,
        pointnet_hidden_dims=tuple(cfg.model.pointnet.hidden_dims),
        pointnet_out_dim=cfg.model.pointnet.out_dim,
        pointnet_use_layernorm=cfg.model.pointnet.use_layernorm,
        state_mlp_hidden=tuple(cfg.model.state_mlp_dims),
        unet_hidden_dims=tuple(cfg.model.unet.hidden_dims),
        unet_kernel_size=cfg.model.unet.kernel_size,
        unet_num_groups=cfg.model.unet.num_groups,
        num_inference_steps=cfg.model.num_inference_steps,
        noise_scheduler_cfg=dict(cfg.model.noise_scheduler),
    )
    return DiffusionPolicy(model_cfg)


def _build_platonic_model(cfg: ConfigDict) -> PlatonicDiffusionPolicy:
    """Construct the Platonic Transformer diffusion policy."""

    transformer_cfg = cfg.model.transformer
    dataset_mode = getattr(cfg, "dataset_mode", "dense")
    dense_mode = bool(getattr(transformer_cfg, "dense_mode", dataset_mode != "sparse"))
    solid_key = transformer_cfg.solid.lower()
    if solid_key not in PLATONIC_GROUPS:
        raise ValueError(f"Unknown platonic solid '{transformer_cfg.solid}'")
    group_size = PLATONIC_GROUPS[solid_key].G

    if transformer_cfg.hidden_dim % group_size != 0:
        raise ValueError(
            f"transformer.hidden_dim ({transformer_cfg.hidden_dim}) must be divisible by group size ({group_size})"
        )
    if transformer_cfg.num_heads % group_size != 0:
        raise ValueError(
            f"transformer.num_heads ({transformer_cfg.num_heads}) must be divisible by group size ({group_size})"
        )
    if transformer_cfg.input_vector_dim <= 0:
        raise ValueError(
            "transformer.input_vector_dim must be >= 1 to represent xyz coordinates"
        )

    input_scalar_dim = transformer_cfg.input_scalar_dim if cfg.use_point_colors else 0

    use_cls_token = getattr(transformer_cfg, "use_cls_token", False)
    if not dense_mode and use_cls_token:
        warnings.warn(
            "CLS token is not supported in sparse transformer mode; disabling it for evaluation.",
            RuntimeWarning,
        )
        use_cls_token = False

    model_cfg = PlatonicDiffusionPolicyConfig(
        horizon=cfg.horizon,
        n_obs_steps=cfg.n_obs_steps,
        action_dim=cfg.action_dim,
        agent_dim=cfg.agent_dim,
        sample_points=cfg.sample_points,
        use_point_colors=cfg.use_point_colors,
        transformer_input_scalar_dim=input_scalar_dim,
        transformer_input_vector_dim=transformer_cfg.input_vector_dim,
        transformer_hidden_dim=transformer_cfg.hidden_dim,
        transformer_output_dim=transformer_cfg.output_dim,
        transformer_num_layers=transformer_cfg.num_layers,
        transformer_num_heads=transformer_cfg.num_heads,
        transformer_solid=transformer_cfg.solid,
        transformer_dropout=transformer_cfg.dropout,
        transformer_drop_path_rate=transformer_cfg.drop_path_rate,
        transformer_ffn_dim_factor=transformer_cfg.ffn_dim_factor,
        transformer_rope_sigma=getattr(transformer_cfg, "rope_sigma", None),
        transformer_ape_sigma=getattr(transformer_cfg, "ape_sigma", None),
        transformer_learned_freqs=getattr(transformer_cfg, "learned_freqs", False),
        transformer_attention=getattr(transformer_cfg, "attention", False),
        transformer_use_key=getattr(transformer_cfg, "use_key", False),
        transformer_mean_aggregation=getattr(
            transformer_cfg, "mean_aggregation", False
        ),
        transformer_ffn_readout=getattr(transformer_cfg, "ffn_readout", False),
        transformer_norm_first=getattr(transformer_cfg, "norm_first", True),
        transformer_layer_scale_init_value=getattr(
            transformer_cfg, "layer_scale_init_value", None
        ),
        transformer_use_cls_token=use_cls_token,
        transformer_dense_mode=dense_mode,
        state_mlp_hidden=tuple(cfg.model.state_mlp_dims),
        unet_hidden_dims=tuple(cfg.model.unet.hidden_dims),
        unet_kernel_size=cfg.model.unet.kernel_size,
        unet_num_groups=cfg.model.unet.num_groups,
        num_inference_steps=cfg.model.num_inference_steps,
        noise_scheduler_cfg=dict(cfg.model.noise_scheduler),
    )
    return PlatonicDiffusionPolicy(model_cfg)


def _build_in_context_platonic_model(cfg: ConfigDict) -> InContextPlatonicDiffusionPolicy:
    """Construct the in-context Platonic Transformer diffusion policy."""

    transformer_cfg = cfg.model.transformer
    dataset_mode = getattr(cfg, "dataset_mode", "dense")
    if dataset_mode == "sparse":
        raise ValueError("In-context Platonic policy requires dense dataset inputs.")

    solid_key = transformer_cfg.solid.lower()
    if solid_key not in PLATONIC_GROUPS:
        raise ValueError(f"Unknown platonic solid '{transformer_cfg.solid}'")
    group_size = PLATONIC_GROUPS[solid_key].G

    if transformer_cfg.hidden_dim % group_size != 0:
        raise ValueError(
            f"transformer.hidden_dim ({transformer_cfg.hidden_dim}) must be divisible by group size ({group_size})"
        )
    if transformer_cfg.num_heads % group_size != 0:
        raise ValueError(
            f"transformer.num_heads ({transformer_cfg.num_heads}) must be divisible by group size ({group_size})"
        )
    if transformer_cfg.input_vector_dim <= 0:
        raise ValueError(
            "transformer.input_vector_dim must be >= 1 to represent xyz coordinates"
        )
    if not getattr(transformer_cfg, "attention", False):
        raise ValueError(
            "In-context Platonic policy requires transformer.attention to be enabled."
        )

    input_scalar_dim = transformer_cfg.input_scalar_dim if cfg.use_point_colors else 0

    use_cls_token = getattr(transformer_cfg, "use_cls_token", False)
    if not getattr(transformer_cfg, "dense_mode", True) and use_cls_token:
        warnings.warn(
            "CLS token is not supported in sparse transformer mode; disabling it for evaluation.",
            RuntimeWarning,
        )
        use_cls_token = False

    in_ctx_cfg = getattr(cfg.model, "in_context", None)
    if in_ctx_cfg is None:
        raise ValueError("Config missing model.in_context for in-context Platonic policy.")

    model_cfg = InContextPlatonicPolicyConfig(
        horizon=cfg.horizon,
        n_obs_steps=cfg.n_obs_steps,
        sample_points=cfg.sample_points,
        action_dim=cfg.action_dim,
        agent_dim=cfg.agent_dim,
        use_point_colors=cfg.use_point_colors,
        transformer_input_scalar_dim=input_scalar_dim,
        transformer_input_vector_dim=transformer_cfg.input_vector_dim,
        transformer_hidden_dim=transformer_cfg.hidden_dim,
        transformer_output_dim=transformer_cfg.output_dim,
        transformer_num_layers=transformer_cfg.num_layers,
        transformer_num_heads=transformer_cfg.num_heads,
        transformer_solid=transformer_cfg.solid,
        transformer_dropout=transformer_cfg.dropout,
        transformer_drop_path_rate=transformer_cfg.drop_path_rate,
        transformer_ffn_dim_factor=transformer_cfg.ffn_dim_factor,
        transformer_rope_sigma=getattr(transformer_cfg, "rope_sigma", None),
        transformer_ape_sigma=getattr(transformer_cfg, "ape_sigma", None),
        transformer_learned_freqs=getattr(transformer_cfg, "learned_freqs", False),
        transformer_attention=getattr(transformer_cfg, "attention", False),
        transformer_use_key=getattr(transformer_cfg, "use_key", False),
        transformer_mean_aggregation=getattr(
            transformer_cfg, "mean_aggregation", False
        ),
        transformer_ffn_readout=getattr(transformer_cfg, "ffn_readout", False),
        transformer_norm_first=getattr(transformer_cfg, "norm_first", True),
        transformer_layer_scale_init_value=getattr(
            transformer_cfg, "layer_scale_init_value", None
        ),
        transformer_use_cls_token=use_cls_token,
        transformer_dense_mode=True,
        time_embedding_base=getattr(in_ctx_cfg, "time_embedding_base", 10000.0),
        scalar_embedding_hidden_dim=getattr(
            in_ctx_cfg, "scalar_embedding_hidden_dim", transformer_cfg.hidden_dim
        ),
        num_inference_steps=cfg.model.num_inference_steps,
        noise_scheduler_cfg=dict(cfg.model.noise_scheduler),
    )
    return InContextPlatonicDiffusionPolicy(model_cfg)


def _infer_model_type(cfg: ConfigDict, payload: dict) -> str:
    """Infer which policy architecture should be instantiated."""

    model_cfg = getattr(cfg, "model", None)
    if model_cfg is not None:
        if getattr(model_cfg, "policy", "") == "in_context":
            return "platonic_in_context"
        if hasattr(model_cfg, "transformer"):
            return "platonic"

    state_dict = payload.get("model")
    if isinstance(state_dict, dict):
        if any(key.startswith("world_time_embedder") for key in state_dict.keys()):
            return "platonic_in_context"
        if any(key.startswith("transformer.") for key in state_dict.keys()):
            return "platonic_in_context"
        if any(key.startswith("encoder.transformer") for key in state_dict.keys()):
            return "platonic"

        if any(key.startswith("encoder.pointnet") for key in state_dict.keys()):
            return "diffusion"

    return "diffusion"


def _build_model(cfg: ConfigDict, model_type: str):
    """Construct the requested policy architecture."""

    if model_type == "platonic":
        return _build_platonic_model(cfg)
    if model_type == "platonic_in_context":
        return _build_in_context_platonic_model(cfg)
    if model_type != "diffusion":
        raise ValueError(f"Unsupported model type '{model_type}'")
    return _build_diffusion_model(cfg)


def _append_camera_frames(obs, buffers: dict) -> None:
    """Append RGB frames for each camera into ``buffers`` for logging.

    Args:
        obs: RLBench observation providing RGB frames.
        buffers: Mapping from camera name to list of frames.
    """
    for name in CAMERA_NAMES:
        frame = getattr(obs, f"{name}_rgb", None)
        if frame is None:
            continue
        buffers[name].append(np.asarray(frame, dtype=np.uint8))


class MaskFilterBuilder:
    """Constructs per-variation mask filters matching the training dataset."""

    def __init__(
        self,
        dataset_path: Optional[Path],
        *,
        ignore_names: Sequence[str],
        fallback_ids: Sequence[int],
    ) -> None:
        """Parse mask metadata and pre-compute name/id lookups.

        Args:
            dataset_path: Path to a dataset file containing mask metadata.
            ignore_names: Mask names to drop for every variation.
            fallback_ids: Mask ids to drop when metadata is unavailable.
        """
        self._fallback_ids = {int(v) for v in fallback_ids}
        self._ignore_names = {name.lower() for name in ignore_names}
        self._variation_mapping: Dict[int, Dict[int, str]] = {}
        self._name_to_handles: Dict[str, set[int]] = defaultdict(set)
        if dataset_path is None or not dataset_path.is_file():
            print(
                f"[eval] Dataset path '{dataset_path}' unavailable; mask filtering limited to fallback ids"
            )
            return

        try:
            with h5py.File(dataset_path, "r") as handle:
                if "variation_metadata" not in handle:
                    print(
                        "[eval] Variation metadata missing in dataset; mask filtering limited to fallback ids"
                    )
                    return
                metadata = handle["variation_metadata"]
                for key in metadata:
                    var_grp = metadata[key]
                    variation = int(var_grp.attrs.get("variation", int(key)))
                    handles_ds = var_grp.get("mask_handles")
                    names_ds = var_grp.get("mask_names")
                    if handles_ds is None or names_ds is None:
                        continue
                    handles = handles_ds[()].astype(np.int64)
                    raw_names = names_ds[()]
                    names: List[str] = []
                    for value in raw_names:
                        if isinstance(value, bytes):
                            names.append(value.decode("utf-8"))
                        else:
                            names.append(str(value))
                    mapping: Dict[int, str] = {}
                    for handle, name in zip(handles.tolist(), names):
                        mapping[int(handle)] = name
                        self._name_to_handles[name.lower()].add(int(handle))
                    if mapping:
                        self._variation_mapping[variation] = mapping
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[eval] Failed to load mask metadata from {dataset_path}: {exc}")

    def make_filter(
        self,
        variation: int,
        task_env,
        obs,
    ) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """Create a boolean mask filter for a given variation, if possible.

        Args:
            variation: RLBench variation identifier.
            task_env: RLBench task environment instance.
            obs: Latest observation for deriving fallback metadata.

        Returns:
            Callable converting a mask array into a boolean keep mask, or ``None``.
        """
        mapping = self._variation_mapping.get(variation)
        if mapping is None:
            mapping = self._mapping_from_scene(task_env, obs)
            if mapping:
                self._variation_mapping[variation] = mapping

        handles_to_ignore = set(self._fallback_ids)
        if mapping:
            for handle, name in mapping.items():
                if name and name.lower() in self._ignore_names:
                    handles_to_ignore.add(handle)
        else:
            for name in self._ignore_names:
                handles_to_ignore.update(self._name_to_handles.get(name, set()))

        if not handles_to_ignore:
            return None

        handles_array = np.array(sorted(handles_to_ignore), dtype=np.int64)

        def _filter(mask: np.ndarray) -> np.ndarray:
            mask_int = mask.astype(np.int64, copy=False)
            return ~np.isin(mask_int, handles_array)

        return _filter

    def _mapping_from_scene(self, task_env, obs) -> Dict[int, str]:
        """Derive a handle→name mapping directly from the RLBench scene.

        Args:
            task_env: RLBench task environment.
            obs: Observation containing segmentation masks.

        Returns:
            Mapping from object handles to resolved names.
        """
        if task_env is None:
            return {}

        handles = set()
        for name in CAMERA_NAMES:
            mask = getattr(obs, f"{name}_mask", None)
            if mask is None:
                continue
            mask_arr = np.asarray(mask).reshape(-1)
            handles.update(int(v) for v in np.unique(mask_arr) if v != 0)

        if not handles:
            return {}

        mapping: Dict[int, str] = {}
        scene = getattr(task_env, "_scene", None)
        if scene is None:
            return {}

        task = getattr(scene, "task", None)
        if task is not None:
            try:
                base = task.get_base()
                for obj in base.get_objects_in_tree(exclude_base=False):
                    handle = int(obj.get_handle())
                    if handle in handles:
                        name = obj.get_name()
                        mapping[handle] = name
                        self._name_to_handles[name.lower()].add(handle)
                        handles.remove(handle)
                    if not handles:
                        break
            except Exception:
                pass

        if handles:
            try:
                for obj in scene.pyrep.get_objects_in_tree(
                    object_type=ObjectType.SHAPE,
                    exclude_base=False,
                ):
                    handle = int(obj.get_handle())
                    if handle in handles:
                        name = obj.get_name()
                        mapping[handle] = name
                        self._name_to_handles[name.lower()].add(handle)
                        handles.remove(handle)
                    if not handles:
                        break
            except Exception:
                pass

        return mapping


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI arguments for the RLBench evaluation script.

    Args:
        argv: Sequence of command-line arguments (excluding script name).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the trained checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config file to override defaults",
    )
    parser.add_argument(
        "--tasks", nargs="+", required=True, help="List of RLBench task names"
    )
    parser.add_argument(
        "--variations",
        nargs="*",
        type=int,
        default=None,
        help="Task variations to evaluate",
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Episodes per task/variation"
    )
    parser.add_argument(
        "--max-steps", type=int, default=75, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for inference"
    )
    parser.add_argument(
        "--renderer",
        choices=["opengl", "opengl3"],
        default="opengl3",
        help="Renderer for RLBench cameras",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=(128, 128),
        help="Camera resolution width height",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA weights if present in checkpoint",
    )
    parser.add_argument(
        "--no-headless", action="store_true", help="Launch RLBench with a viewer"
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="Override wandb run name"
    )
    parser.add_argument(
        "--executed-actions",
        type=int,
        default=16,
        help="How many actions from each sampled plan to execute (default: 16)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> None:
    """Evaluate a trained diffusion policy on RLBench tasks.

    Args:
        argv: Command-line arguments provided by :mod:`argparse` or ``sys.argv``.
    """
    args = parse_args(argv)

    if args.executed_actions <= 0:
        raise ValueError("--executed-actions must be a positive integer")

    cfg = _load_config(args.config)
    cfg = cfg.copy_and_resolve_references()

    device = torch.device(args.device)
    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    model_type = _infer_model_type(cfg, payload)

    state_dict_key = (
        "ema_model" if (args.use_ema and "ema_model" in payload) else "model"
    )
    state_dict = payload.get(state_dict_key, {})

    if model_type == "platonic":
        transformer_cfg = getattr(cfg.model, "transformer", None)
        if transformer_cfg is None:
            raise ValueError(
                "Platonic model detected but cfg.model.transformer is missing"
            )
        has_cls_weights = any(
            key.startswith("encoder.transformer.cls_") for key in state_dict.keys()
        )
        if has_cls_weights and not getattr(transformer_cfg, "use_cls_token", False):
            print("Enabling CLS token in config based on checkpoint contents.")
            transformer_cfg.use_cls_token = True

    print(f"Detected model type: {model_type}")

    model = _build_model(cfg, model_type).to(device)

    if args.use_ema and "ema_model" in payload:
        model.load_state_dict(payload["ema_model"])
        print(f"Loaded EMA weights from {checkpoint_path}")
    else:
        model.load_state_dict(payload["model"])
        print(f"Loaded model weights from {checkpoint_path}")

    model.eval()

    obs_config = create_observation_config(
        image_size=args.image_size,
        renderer=args.renderer,
    )
    env = instantiate_environment(
        action_mode=create_action_mode(),
        obs_config=obs_config,
        headless=not args.no_headless,
    )

    sim_cfg = SimulationInputConfig(
        sample_points=cfg.sample_points,
        n_obs_steps=cfg.n_obs_steps,
        use_point_colors=cfg.use_point_colors,
        device=device,
    )
    processor = ObservationProcessor(sim_cfg)
    history = ObservationHistory(cfg.n_obs_steps)

    dataset_path = (
        Path(cfg.dataset_path).expanduser().resolve() if cfg.dataset_path else None
    )
    if dataset_path is not None and not dataset_path.is_file():
        print(
            f"[eval] Dataset path {dataset_path} not found; falling back to default mask ids only"
        )
        dataset_path = None
    mask_builder = MaskFilterBuilder(
        dataset_path,
        ignore_names=cfg.eval.mask_names_to_ignore,
        fallback_ids=cfg.eval.mask_ids_to_ignore,
    )

    tasks = []
    for name in args.tasks:
        task_cls = resolve_task_class(name)
        tasks.append((name, task_cls))

    variations: List[int]
    if args.variations:
        variations = list(dict.fromkeys(args.variations))
    else:
        variations = [0]

    wandb_run = None
    if cfg.logging.enable_wandb and not args.no_wandb:
        run_name = args.wandb_run_name or f"eval_{checkpoint_path.stem}"
        wandb_config = {
            "checkpoint": str(checkpoint_path),
            "tasks": args.tasks,
            "variations": variations,
            "episodes_per_variation": args.episodes,
            "max_steps": args.max_steps,
            "use_ema": args.use_ema,
            "executed_actions": args.executed_actions,
        }
        wandb_run = wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=run_name,
            config=wandb_config,
            reinit=True,
        )

    results = []
    episode_counter = 0
    video_logged = False
    try:
        for task_name, task_cls in tasks:
            task_env = env.get_task(task_cls)
            for variation in variations:
                try:
                    task_env.set_variation(variation)
                except Exception as exc:  # pragma: no cover - RLBench runtime errors
                    print(f"Failed to set variation {variation} for {task_name}: {exc}")
                    continue

                for episode in range(args.episodes):
                    descriptions, obs = task_env.reset()
                    processor.set_mask_filter(
                        mask_builder.make_filter(variation, task_env, obs)
                    )
                    history.reset()
                    feats, agent_state = processor.extract(obs)
                    history.append(feats, agent_state)

                    episode_tag: Optional[str] = None
                    video_buffers: Optional[dict] = None
                    if wandb_run is not None:
                        episode_tag = f"eval/rollout_episode_{episode_counter}"
                        episode_counter += 1
                        if not video_logged:
                            video_buffers = {name: [] for name in CAMERA_NAMES}
                            video_logged = True

                    if video_buffers is not None:
                        _append_camera_frames(obs, video_buffers)

                    success = False
                    steps_taken = 0

                    while steps_taken < args.max_steps:
                        pc_stack, agent_stack = history.stacked(device)
                        point_batch = pc_stack.unsqueeze(0)
                        agent_batch = agent_stack.unsqueeze(0)

                        with torch.no_grad():
                            plan = model.sample(point_batch, agent_batch)

                        if (
                            episode_tag is not None
                            and steps_taken % LOG_POINTCLOUD_INTERVAL == 0
                        ):
                            log_pointcloud_wandb(
                                wandb_run=wandb_run,
                                point_cloud=pc_stack.cpu(),
                                gt_actions=None,
                                pred_actions=plan[0].cpu(),
                                tag=episode_tag,
                            )

                        actions_to_execute = min(
                            args.executed_actions,
                            plan.shape[1],
                            args.max_steps - steps_taken,
                        )
                        if actions_to_execute <= 0:
                            break

                        episode_done = False
                        for action_index in range(actions_to_execute):
                            command = action_plan_to_command(
                                plan[0, action_index],
                                last_agent_state=history.latest_agent_state(),
                            )

                            try:
                                obs, reward, terminate = task_env.step(command)
                            except (
                                Exception
                            ) as exc:  # pragma: no cover - runtime safety
                                print(
                                    f"Step failed on {task_name} variation {variation}: {exc}"
                                )
                                episode_done = True
                                break

                            feats, agent_state = processor.extract(obs)
                            history.append(feats, agent_state)
                            if video_buffers is not None:
                                _append_camera_frames(obs, video_buffers)
                            steps_taken += 1

                            if terminate:
                                success = True
                                episode_done = True
                                break

                            if steps_taken >= args.max_steps:
                                episode_done = True
                                break

                        if episode_done:
                            break

                    results.append(
                        {
                            "task": task_name,
                            "variation": variation,
                            "episode": episode,
                            "success": success,
                            "steps": steps_taken,
                        }
                    )
                    status = "success" if success else "failure"
                    print(
                        f"[{task_name} | var {variation} | ep {episode}] {status} in {steps_taken} steps"
                    )

                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                f"{episode_tag}/success": float(success),
                                f"{episode_tag}/steps": steps_taken,
                            }
                        )

                    if video_buffers is not None:
                        video_logs = {}
                        for camera, frames in video_buffers.items():
                            if not frames:
                                continue
                            video_array = np.stack(frames, axis=0).astype(np.uint8)
                            video_array = np.transpose(video_array, (0, 3, 1, 2))
                            video_logs[f"eval/videos/{camera}"] = wandb.Video(
                                video_array, fps=10
                            )
                        if video_logs:
                            wandb_run.log(video_logs)
    finally:
        env.shutdown()

    if not results:
        print("No episodes evaluated.")
        if wandb_run is not None:
            wandb_run.finish()
        return

    total = len(results)
    successes = sum(1 for r in results if r["success"])
    avg_steps = sum(r["steps"] for r in results if r["steps"] > 0)
    avg_steps = avg_steps / max(1, successes)
    print("\n=== Evaluation Summary ===")
    print(f"Episodes: {total}")
    print(f"Successes: {successes} ({successes / total * 100:.1f}%)")
    if successes > 0:
        print(f"Average steps (successful episodes): {avg_steps:.2f}")

    if wandb_run is not None:
        wandb_run.log(
            {
                "eval/summary/episodes": total,
                "eval/summary/success_rate": successes / total if total > 0 else 0.0,
                "eval/summary/avg_steps_success": avg_steps if successes > 0 else 0.0,
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
