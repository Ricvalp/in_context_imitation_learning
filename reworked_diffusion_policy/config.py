"""Default configuration for the lightweight 3D diffusion policy trainer."""

from __future__ import annotations

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()

    # Data -----------------------------------------------------------------
    cfg.dataset_path = ""  # must be provided via flag
    cfg.tasks = ()
    cfg.batch_size = 64
    cfg.num_workers = 0
    cfg.pin_memory = True
    cfg.shuffle = True
    cfg.drop_last = True

    # Task geometry --------------------------------------------------------
    cfg.horizon = 16
    cfg.n_obs_steps = 2
    cfg.action_dim = 8
    cfg.agent_dim = 8
    cfg.sample_points = 4096
    cfg.use_point_colors = True

    # Model ----------------------------------------------------------------
    cfg.model = ConfigDict()
    cfg.model.pointnet = ConfigDict()
    cfg.model.pointnet.in_channels = 6
    cfg.model.pointnet.hidden_dims = (128, 256, 512)
    cfg.model.pointnet.out_dim = 256
    cfg.model.pointnet.use_layernorm = True

    cfg.model.state_mlp_dims = (128, 256)

    cfg.model.unet = ConfigDict()
    cfg.model.unet.hidden_dims = (256, 512, 1024)
    cfg.model.unet.kernel_size = 3
    cfg.model.unet.num_groups = 8
    # cfg.model.unet.num_res_blocks = 2

    cfg.model.noise_scheduler = ConfigDict()
    cfg.model.noise_scheduler.num_train_timesteps = 1000
    cfg.model.noise_scheduler.beta_start = 1e-4
    cfg.model.noise_scheduler.beta_end = 0.02
    cfg.model.noise_scheduler.beta_schedule = "squaredcos_cap_v2"
    cfg.model.num_inference_steps = 50

    # Training -------------------------------------------------------------
    cfg.training = ConfigDict()
    cfg.training.num_epochs = 1000
    cfg.training.lr = 1e-4
    cfg.training.weight_decay = 1e-6
    cfg.training.beta1 = 0.95
    cfg.training.beta2 = 0.999
    cfg.training.eps = 1e-8
    cfg.training.grad_clip_norm = 1.0
    cfg.training.log_interval = 20
    cfg.training.eval_interval = 50

    # EMA ------------------------------------------------------------------
    cfg.ema = ConfigDict()
    cfg.ema.use_ema = True
    cfg.ema.decay = 0.995

    # Evaluation -----------------------------------------------------------
    cfg.eval = ConfigDict()
    cfg.eval.max_batches = 50
    cfg.eval.enable_viser = False
    cfg.eval.point_size = 0.002
    cfg.eval.axes_length = 0.1
    cfg.eval.axes_radius = 0.004
    cfg.eval.mask_names_to_ignore = (
        "Floor",
        "Wall1",
        "Wall2",
        "Wall3",
        "Wall4",
        "Roof",
        "workspace",
        "diningTable_visible",
        "ResizableFloor_5_25_visibleElement",
    )
    cfg.eval.mask_ids_to_ignore = ()

    # Logging --------------------------------------------------------------
    cfg.logging = ConfigDict()
    cfg.logging.enable_wandb = True
    cfg.logging.project = "equi_poli_debug"
    cfg.logging.entity = "equivariance"
    cfg.logging.run_name = "debug_run"
    cfg.logging.log_pointcloud_eval = True

    # Checkpointing ---------------------------------------------------------
    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "./checkpoints"
    cfg.checkpoint.prefix = "diffusion_policy"
    cfg.checkpoint.save_every = 50
    cfg.checkpoint.top_k = 3
    cfg.checkpoint.maximize_metric = False

    # Runtime --------------------------------------------------------------
    cfg.device = "cuda"
    cfg.seed = 42

    return cfg


__all__ = ["get_config"]
