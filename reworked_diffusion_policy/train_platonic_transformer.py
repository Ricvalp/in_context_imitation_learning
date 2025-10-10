"""Entry point for training the Platonic Transformer diffusion policy."""

from __future__ import annotations

import os
from typing import Dict

from absl import app, flags, logging
import tqdm
import wandb
from ml_collections import ConfigDict
from ml_collections import config_flags
import torch
from torch.utils.data import DataLoader

from .platonic_config import get_config
from .checkpoint import CheckpointManager
from .dataset import (
    DatasetConfig,
    SparseDatasetConfig,
    RLBenchTemporalH5Dataset,
    RLBenchTemporalH5SparseDataset,
    collate_temporal_batch,
    collate_sparse_temporal_batch,
)
from .models.platonic_policy import PlatonicDiffusionPolicy, PlatonicDiffusionPolicyConfig
from .models.platonic_transformer.groups import PLATONIC_GROUPS
from .utils import (
    ExponentialMovingAverage,
    log_pointcloud_wandb,
    mse,
    set_seed,
    visualize_trajectories,
)


FLAGS = flags.FLAGS

default_config_path = os.path.join(os.path.dirname(__file__), "platonic_config.py")
config_flags.DEFINE_config_file(
    "config",
    default=default_config_path,
)

flags.DEFINE_string("dataset_path", None, "Override dataset path")
flags.DEFINE_integer("batch_size", None, "Override batch size")
flags.DEFINE_integer("epochs", None, "Override number of epochs")
flags.DEFINE_float("learning_rate", None, "Override learning rate")
flags.DEFINE_float("weight_decay", None, "Override weight decay")
flags.DEFINE_float("ema_decay", None, "Override EMA decay")
flags.DEFINE_integer("sample_points", None, "Number of points to sample per cloud")
flags.DEFINE_integer("num_workers", None, "Number of dataloader workers")
flags.DEFINE_integer("num_inference_steps", None, "Override sampling steps")
flags.DEFINE_bool("enable_viser", None, "Enable viser visualization during eval")
flags.DEFINE_string("device", None, "Device to train on (cpu or cuda)" )
flags.DEFINE_string("checkpoint_dir", None, "Override checkpoint directory")
flags.DEFINE_integer("checkpoint_interval", None, "Override checkpoint save interval")
flags.DEFINE_multi_string("task", [], "RLBench task names to include when reading datasets")
flags.DEFINE_bool("debug_dataset", False, "Load only a subset of each task for faster debugging")
flags.DEFINE_integer(
    "debug_max_samples",
    16,
    "Number of samples to load per task when --debug_dataset is enabled",
)
flags.DEFINE_enum(
    "dataset_mode",
    None,
    ["dense", "sparse"],
    "Point-cloud batching strategy to use (dense or sparse).",
)
flags.DEFINE_integer(
    "sparse_max_points",
    None,
    "Optional cap on the number of points per frame when using sparse mode.",
)


def _apply_overrides(cfg: ConfigDict) -> ConfigDict:
    """Apply CLI flag overrides on top of the resolved configuration.

    Args:
        cfg: Configuration produced by :func:`get_config`.

    Returns:
        Updated configuration reflecting any CLI overrides.
    """
    if FLAGS.dataset_path:
        cfg.dataset_path = FLAGS.dataset_path
    if FLAGS.batch_size:
        cfg.batch_size = FLAGS.batch_size
    if FLAGS.epochs:
        cfg.training.num_epochs = FLAGS.epochs
    if FLAGS.learning_rate:
        cfg.training.lr = FLAGS.learning_rate
    if FLAGS.weight_decay:
        cfg.training.weight_decay = FLAGS.weight_decay
    if FLAGS.sample_points:
        cfg.sample_points = FLAGS.sample_points
    if FLAGS.num_workers is not None:
        cfg.num_workers = FLAGS.num_workers
    if FLAGS.num_inference_steps:
        cfg.model.num_inference_steps = FLAGS.num_inference_steps
    if FLAGS.ema_decay:
        cfg.ema.decay = FLAGS.ema_decay
    if FLAGS.enable_viser is not None:
        cfg.eval.enable_viser = FLAGS.enable_viser
    if FLAGS.device:
        cfg.device = FLAGS.device
    if FLAGS.checkpoint_dir:
        cfg.checkpoint.dir = FLAGS.checkpoint_dir
    if FLAGS.checkpoint_interval is not None:
        cfg.checkpoint.save_every = max(0, FLAGS.checkpoint_interval)
    if FLAGS.task:
        cfg.tasks = tuple(FLAGS.task)
    if FLAGS.debug_dataset:
        cfg.debug.limit_dataset = True
    if FLAGS.debug_max_samples is not None:
        cfg.debug.max_samples_per_task = max(1, FLAGS.debug_max_samples)
    if FLAGS.dataset_mode:
        cfg.dataset_mode = FLAGS.dataset_mode
    if FLAGS.sparse_max_points is not None:
        cfg.sparse_max_points_per_frame = max(1, FLAGS.sparse_max_points)
    cfg.model.transformer.dense_mode = cfg.dataset_mode != "sparse"
    return cfg


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move every tensor in ``batch`` to ``device`` using non-blocking copies.

    Args:
        batch: Mini-batch mapping field names to tensors.
        device: Destination device.

    Returns:
        Dictionary containing tensors located on ``device``.
    """
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_dataloaders(cfg: ConfigDict):
    """Construct dataset and data loaders based on ``cfg``.

    Args:
        cfg: Training configuration.

    Returns:
        Tuple ``(dataset, train_loader, eval_loader)``.
    """
    max_samples = None
    if hasattr(cfg, "debug") and cfg.debug.limit_dataset:
        max_samples = max(1, int(cfg.debug.max_samples_per_task))

    dataset_mode = getattr(cfg, "dataset_mode", "dense")

    if dataset_mode == "sparse":
        dataset_cfg = SparseDatasetConfig(
            path=cfg.dataset_path,
            n_obs_steps=cfg.n_obs_steps,
            action_horizon=cfg.horizon,
            use_point_colors=cfg.use_point_colors,
            task_names=tuple(cfg.tasks or ()),
            max_samples_per_file=max_samples,
            max_points_per_frame=cfg.sparse_max_points_per_frame,
        )
        dataset = RLBenchTemporalH5SparseDataset(dataset_cfg)
        collate_fn = collate_sparse_temporal_batch
    else:
        dataset_cfg = DatasetConfig(
            path=cfg.dataset_path,
            sample_points=cfg.sample_points,
            n_obs_steps=cfg.n_obs_steps,
            action_horizon=cfg.horizon,
            use_point_colors=cfg.use_point_colors,
            task_names=tuple(cfg.tasks or ()),
            max_samples_per_file=max_samples,
        )
        dataset = RLBenchTemporalH5Dataset(dataset_cfg)
        collate_fn = collate_temporal_batch

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        collate_fn=collate_fn,
    )

    eval_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=max(0, cfg.num_workers // 2),
        pin_memory=cfg.pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return dataset, train_loader, eval_loader


def build_model(cfg: ConfigDict) -> PlatonicDiffusionPolicy:
    """Instantiate the Platonic Transformer diffusion policy from ``cfg``."""

    transformer_cfg = cfg.model.transformer
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
        raise ValueError("transformer.input_vector_dim must be >= 1 to represent xyz coordinates")

    if not cfg.use_point_colors and transformer_cfg.input_scalar_dim > 0:
        logging.warning(
            "use_point_colors=False, overriding transformer.input_scalar_dim to zero scalars."
        )
    input_scalar_dim = transformer_cfg.input_scalar_dim if cfg.use_point_colors else 0

    use_cls_token = transformer_cfg.use_cls_token
    if not transformer_cfg.dense_mode and use_cls_token:
        logging.warning("CLS token is not supported in sparse transformer mode; disabling it.")
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
        transformer_rope_sigma=transformer_cfg.rope_sigma,
        transformer_ape_sigma=transformer_cfg.ape_sigma,
        transformer_learned_freqs=transformer_cfg.learned_freqs,
        transformer_attention=transformer_cfg.attention,
        transformer_use_key=transformer_cfg.use_key,
        transformer_mean_aggregation=transformer_cfg.mean_aggregation,
        transformer_ffn_readout=transformer_cfg.ffn_readout,
        transformer_norm_first=transformer_cfg.norm_first,
        transformer_layer_scale_init_value=transformer_cfg.layer_scale_init_value,
        transformer_use_cls_token=use_cls_token,
        transformer_dense_mode=transformer_cfg.dense_mode,
        state_mlp_hidden=tuple(cfg.model.state_mlp_dims),
        unet_hidden_dims=tuple(cfg.model.unet.hidden_dims),
        unet_kernel_size=cfg.model.unet.kernel_size,
        unet_num_groups=cfg.model.unet.num_groups,
        num_inference_steps=cfg.model.num_inference_steps,
        noise_scheduler_cfg=dict(cfg.model.noise_scheduler),
    )
    return PlatonicDiffusionPolicy(model_cfg)


def evaluate(
    model: PlatonicDiffusionPolicy,
    dataloader: DataLoader,
    device: torch.device,
    cfg: ConfigDict,
    *,
    wandb_run=None,
    epoch: int | None = None,
) -> float:
    """Evaluate ``model`` over ``dataloader`` and return the mean MSE.

    Args:
        model: Policy to evaluate.
        dataloader: Data loader providing evaluation batches.
        device: Target device for tensors.
        cfg: Configuration controlling evaluation side-effects.
        wandb_run: Optional active Weights & Biases run.
        epoch: Optional epoch index for logging.

    Returns:
        Mean squared error aggregated over the evaluation set.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    vis_sample = None

    def _gather_first_pointcloud(batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "point_cloud_sample_idx" not in batch_dict:
            return batch_dict["point_clouds"][0].cpu()

        sample_idx = 0
        sample_mask = batch_dict["point_cloud_sample_idx"] == sample_idx
        points = batch_dict["point_clouds"][sample_mask].cpu()
        obs_ids = batch_dict["point_cloud_obs_idx"][sample_mask].cpu()
        lengths = batch_dict.get("frame_lengths")
        if lengths is None:
            raise ValueError("frame_lengths missing from sparse batch")
        lengths = lengths[sample_idx].cpu()
        max_len = int(lengths.max().item())
        feat_dim = points.shape[-1]
        dense = torch.zeros(cfg.n_obs_steps, max_len, feat_dim)
        for obs_step in range(cfg.n_obs_steps):
            mask = obs_ids == obs_step
            step_points = points[mask]
            count = step_points.shape[0]
            if count > 0:
                dense[obs_step, :count] = step_points
        return dense

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = _to_device(batch, device)
            preds = model.sample(
                batch["point_clouds"],
                batch["agent_pos"],
                point_sample_idx=batch.get("point_cloud_sample_idx"),
                point_obs_idx=batch.get("point_cloud_obs_idx"),
            )
            target = batch["action"]
            batch_mse = mse(preds, target)
            bs = preds.shape[0]
            total_loss += float(batch_mse.detach().cpu().item()) * bs
            total_samples += bs

            if vis_sample is None:
                vis_sample = (
                    _gather_first_pointcloud(batch),
                    target[0].cpu(),
                    preds[0].cpu(),
                )

            if cfg.eval.max_batches is not None and batch_idx + 1 >= cfg.eval.max_batches:
                break

    if vis_sample is not None:
        pc, gt_act, pred_act = vis_sample
        if cfg.eval.enable_viser:
            visualize_trajectories(
                pc,
                gt_act,
                pred_act,
                point_size=cfg.eval.point_size,
                axes_length=cfg.eval.axes_length,
                axes_radius=cfg.eval.axes_radius,
            )
        if wandb_run is not None and cfg.logging.enable_wandb and cfg.logging.log_pointcloud_eval:
            log_pointcloud_wandb(
                wandb_run=wandb_run,
                point_cloud=pc,
                gt_actions=gt_act,
                pred_actions=pred_act,
                tag="eval/pointcloud",
            )

    return total_loss / max(1, total_samples)


def train(argv) -> None:
    """Entry point used by both ``absl.app`` and :mod:`sphinx` automation.

    Args:
        argv: Command-line arguments supplied by :mod:`absl`.
    """
    del argv
    cfg = get_config()
    cfg = cfg.copy_and_resolve_references()
    cfg.update(FLAGS.config)
    cfg = _apply_overrides(cfg)

    if not cfg.dataset_path:
        raise ValueError("dataset_path must be provided (via config or --dataset_path)")

    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    logging.info("Loading dataset from %s", cfg.dataset_path)
    dataset, train_loader, eval_loader = build_dataloaders(cfg)

    logging.info(
        "Loaded %d samples from %d files", len(dataset), len(dataset.source_files)
    )
    if dataset.task_names:
        logging.info("Tasks: %s", ", ".join(dataset.task_names))

    run_id = wandb.util.generate_id()
    logging.info("Run ID: %s", run_id)

    model = build_model(cfg)
    model.set_normalizer(dataset.normalizer)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        betas=(cfg.training.beta1, cfg.training.beta2),
        eps=cfg.training.eps,
        weight_decay=cfg.training.weight_decay,
    )

    ema_helper = None
    if cfg.ema.use_ema:
        ema_helper = ExponentialMovingAverage(model, decay=cfg.ema.decay).to(device)

    checkpoint_manager = CheckpointManager(
        directory=cfg.checkpoint.dir,
        prefix=cfg.checkpoint.prefix,
        run_id=run_id,
        top_k=cfg.checkpoint.top_k,
        maximize_metric=cfg.checkpoint.maximize_metric,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logging.info("Model parameters: %.2fM", total_params / 1e6)
    wandb_run = None
    full_run_name = cfg.logging.run_name
    if full_run_name:
        full_run_name = f"{full_run_name}_{run_id}"
    else:
        full_run_name = run_id
    if cfg.logging.enable_wandb:
        wandb_cfg = cfg.to_dict()
        wandb_cfg.setdefault("logging", {})
        wandb_cfg["logging"]["run_id"] = run_id
        wandb_run = wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=full_run_name,
            id=run_id,
            config=wandb_cfg,
            reinit=True,
        )

    logging.info("Starting training for %d epochs", cfg.training.num_epochs)
    global_step = 0
    last_eval_metric: float | None = None

    epoch_iter = tqdm.trange(cfg.training.num_epochs, desc="Epoch", leave=True)
    for epoch in epoch_iter:
        model.train()
        step_iter = tqdm.tqdm(train_loader, desc=f"Train {epoch}", leave=False)
        for batch_idx, batch in enumerate(step_iter):
            batch = _to_device(batch, device)
            loss, metrics = model.compute_loss(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            optimizer.step()

            if ema_helper is not None:
                ema_helper.update(model)

            train_mse = float(metrics.get("train_mse", loss.item()))
            step_iter.set_postfix(loss=f"{train_mse:.4f}")

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/mse": train_mse,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

            global_step += 1

        if (epoch + 1) % cfg.training.eval_interval == 0:
            eval_model = ema_helper.ema_model if ema_helper is not None else model
            eval_loss = evaluate(
                eval_model,
                eval_loader,
                device,
                cfg,
                wandb_run=wandb_run,
                epoch=epoch,
            )
            checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                ema_model=ema_helper.ema_model if ema_helper is not None else None,
                metric=eval_loss,
                update_latest=False,
            )
            last_eval_metric = eval_loss
            epoch_iter.set_postfix(eval_mse=f"{eval_loss:.6f}")
            if wandb_run is not None:
                wandb_run.log({"eval/mse": eval_loss, "train/epoch": epoch}, step=global_step)

        if cfg.checkpoint.save_every > 0 and (epoch + 1) % cfg.checkpoint.save_every == 0:
            ckpt_path = checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                ema_model=ema_helper.ema_model if ema_helper is not None else None,
            )
            logging.info("Saved checkpoint to %s", ckpt_path)

    if wandb_run is not None:
        wandb_run.finish()

    final_ckpt = checkpoint_manager.save(
        model=model,
        optimizer=optimizer,
        epoch=cfg.training.num_epochs,
        global_step=global_step,
        ema_model=ema_helper.ema_model if ema_helper is not None else None,
    )
    logging.info("Saved final checkpoint to %s", final_ckpt)


def main(argv) -> None:
    """Wrapper that mirrors ``absl.app`` expectations.

    Args:
        argv: Raw CLI arguments from :mod:`absl.app`.
    """
    train(argv)


if __name__ == "__main__":
    app.run(main)
