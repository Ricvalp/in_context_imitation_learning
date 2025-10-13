"""Default configuration for training with the Platonic Transformer policy."""

from __future__ import annotations

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    """Return the default configuration for the Platonic Transformer trainer."""
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
    cfg.dataset_mode = "dense"  # options: "dense", "sparse"
    cfg.sparse_max_points_per_frame = None

    # Model ----------------------------------------------------------------
    cfg.model = ConfigDict()
    cfg.model.transformer = ConfigDict()
    cfg.model.transformer.input_scalar_dim = 3  # RGB values appended to xyz
    cfg.model.transformer.input_vector_dim = (
        1  # xyz interpreted as a single vector channel
    )
    cfg.model.transformer.hidden_dim = (
        384  # divisible by |G| for tetrahedral group (12)
    )
    cfg.model.transformer.output_dim = 256
    cfg.model.transformer.num_layers = 4
    cfg.model.transformer.num_heads = 12  # multiples of |G|
    cfg.model.transformer.solid = "tetrahedron"
    cfg.model.transformer.dropout = 0.1
    cfg.model.transformer.drop_path_rate = 0.1
    cfg.model.transformer.ffn_dim_factor = 4
    cfg.model.transformer.rope_sigma = 1.0
    cfg.model.transformer.ape_sigma = 0.3
    cfg.model.transformer.learned_freqs = True
    cfg.model.transformer.attention = False
    cfg.model.transformer.use_key = False
    cfg.model.transformer.mean_aggregation = True
    cfg.model.transformer.ffn_readout = True
    cfg.model.transformer.norm_first = True
    cfg.model.transformer.layer_scale_init_value = 1e-3
    cfg.model.transformer.dense_mode = True
    cfg.model.transformer.use_cls_token = False

    cfg.model.state_mlp_dims = (128, 256)

    cfg.model.unet = ConfigDict()
    cfg.model.unet.hidden_dims = (256, 512, 1024)
    cfg.model.unet.kernel_size = 3
    cfg.model.unet.num_groups = 8

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
    cfg.logging.project = "equi_poli_platonic"
    cfg.logging.entity = "equivariance"
    cfg.logging.run_name = "platonic_transformer_run"
    cfg.logging.log_pointcloud_eval = True

    # Debug ----------------------------------------------------------------
    cfg.debug = ConfigDict()
    cfg.debug.limit_dataset = False
    cfg.debug.max_samples_per_task = 16

    # Checkpointing ---------------------------------------------------------
    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "./checkpoints"
    cfg.checkpoint.prefix = "platonic_diffusion_policy"
    cfg.checkpoint.save_every = 50
    cfg.checkpoint.top_k = 3
    cfg.checkpoint.maximize_metric = False

    # Runtime --------------------------------------------------------------
    cfg.device = "cuda"
    cfg.seed = 42

    return cfg


__all__ = ["get_config"]
