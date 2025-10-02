"""Entry point for training the lightweight 3D diffusion policy."""

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

from .config import get_config
from .dataset import DatasetConfig, RLBenchTemporalH5Dataset, collate_temporal_batch
from .models.diffusion_policy import DiffusionPolicy, DiffusionPolicyConfig
from .utils import (
    ExponentialMovingAverage,
    log_pointcloud_wandb,
    mse,
    set_seed,
    visualize_trajectories,
)


FLAGS = flags.FLAGS

default_config_path = os.path.join(os.path.dirname(__file__), "config.py")
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


def _apply_overrides(cfg: ConfigDict) -> ConfigDict:
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
    return cfg


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_dataloaders(cfg: ConfigDict):
    dataset_cfg = DatasetConfig(
        path=cfg.dataset_path,
        sample_points=cfg.sample_points,
        n_obs_steps=cfg.n_obs_steps,
        action_horizon=cfg.horizon,
        use_point_colors=cfg.use_point_colors,
    )

    train_dataset = RLBenchTemporalH5Dataset(dataset_cfg)
    eval_dataset = RLBenchTemporalH5Dataset(dataset_cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        collate_fn=collate_temporal_batch,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(0, cfg.num_workers // 2),
        pin_memory=cfg.pin_memory,
        drop_last=False,
        collate_fn=collate_temporal_batch,
    )

    return train_loader, eval_loader


def build_model(cfg: ConfigDict) -> DiffusionPolicy:
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


def evaluate(
    model: DiffusionPolicy,
    dataloader: DataLoader,
    device: torch.device,
    cfg: ConfigDict,
    *,
    wandb_run=None,
    epoch: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    vis_sample = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = _to_device(batch, device)
            preds = model.sample(batch["point_clouds"], batch["agent_pos"])
            target = batch["action"]
            batch_mse = mse(preds, target)
            bs = preds.shape[0]
            total_loss += float(batch_mse.detach().cpu().item()) * bs
            total_samples += bs

            if vis_sample is None:
                vis_sample = (
                    batch["point_clouds"][0].cpu(),
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
            prefix = "eval" if epoch is None else f"eval/epoch_{epoch}"
            log_pointcloud_wandb(
                wandb_run=wandb_run,
                point_cloud=pc,
                gt_actions=gt_act,
                pred_actions=pred_act,
                tag=prefix,
            )

    return total_loss / max(1, total_samples)


def train(argv) -> None:
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
    train_loader, eval_loader = build_dataloaders(cfg)

    model = build_model(cfg).to(device)

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

    total_params = sum(p.numel() for p in model.parameters())
    logging.info("Model parameters: %.2fM", total_params / 1e6)
    wandb_run = None
    if cfg.logging.enable_wandb:
        wandb_cfg = cfg.to_dict()
        wandb_run = wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=cfg.logging.run_name,
            config=wandb_cfg,
            reinit=True,
        )

    logging.info("Starting training for %d epochs", cfg.training.num_epochs)
    global_step = 0

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
            epoch_iter.set_postfix(eval_mse=f"{eval_loss:.6f}")
            if wandb_run is not None:
                wandb_run.log({"eval/mse": eval_loss, "train/epoch": epoch}, step=global_step)

    if wandb_run is not None:
        wandb_run.finish()


def main(argv) -> None:
    train(argv)


if __name__ == "__main__":
    app.run(main)
