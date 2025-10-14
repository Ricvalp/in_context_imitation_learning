"""Default configuration for the Transformer-based in-context diffusion policy."""

from __future__ import annotations

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    """Return the default configuration for the in-context diffusion transformer."""
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
    cfg.sample_points = 1024
    cfg.use_point_colors = True
    cfg.dataset_mode = "dense"

    # Model ----------------------------------------------------------------
    cfg.model = ConfigDict()
    cfg.model.policy = "in_context_diffusion"

    cfg.model.transformer = ConfigDict()
    cfg.model.transformer.hidden_dim = 512
    cfg.model.transformer.num_layers = 6
    cfg.model.transformer.num_heads = 8
    cfg.model.transformer.mlp_dim = 2048
    cfg.model.transformer.dropout = 0.1
    cfg.model.transformer.attention_dropout = 0.1
    cfg.model.transformer.activation = "gelu"
    cfg.model.transformer.norm_first = True
    cfg.model.transformer.layer_norm_eps = 1e-5
    cfg.model.transformer.mask_action_to_obs = False

    cfg.model.in_context = ConfigDict()
    cfg.model.in_context.scalar_embedding_hidden_dim = 256
    cfg.model.in_context.time_embedding_base = 10000.0
    cfg.model.in_context.diffusion_embedding_base = 10000.0

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
    cfg.logging.project = "equi_poli_diffusion_transformer"
    cfg.logging.entity = "equivariance"
    cfg.logging.run_name = "diffusion_transformer_run"
    cfg.logging.log_pointcloud_eval = True

    # Debug ----------------------------------------------------------------
    cfg.debug = ConfigDict()
    cfg.debug.limit_dataset = False
    cfg.debug.max_samples_per_task = 16

    # Checkpointing ---------------------------------------------------------
    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "./checkpoints"
    cfg.checkpoint.prefix = "in_context_diffusion_policy"
    cfg.checkpoint.save_every = 50
    cfg.checkpoint.top_k = 3
    cfg.checkpoint.maximize_metric = False

    # Runtime --------------------------------------------------------------
    cfg.device = "cuda"
    cfg.seed = 42

    return cfg


__all__ = ["get_config"]

