import os
import argparse

import torch
import torchmetrics
import numpy as np
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import gc

from datasets.omol import get_omol_loaders
from models.platoformer.platoformer import PlatonicTransformer
from models.platoformer.groups import PLATONIC_GROUPS

from utils import CosineWarmupScheduler, MemoryMonitorCallback, RandomSOd, TimerCallback
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.strategies import DDPStrategy

# Performance optimizations
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


class OMolModel(pl.LightningModule):
    """Lightning module for OMol energy and force prediction."""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)

        # Calculate total input channels
        in_channels_scalar = (
            92  # base atom features onehot
            + 3
            * ("coords" in self.hparams.scalar_features)  # x,y,z coordinates as scalars
            + 1 * ("charges" in self.hparams.scalar_features)  # charges as scalars
        )
        in_channels_vector = 0  # No vector features used in this setup

        # --- Dynamically configure model outputs based on force prediction mode ---
        if self.hparams.predict_forces:
            # Direct force prediction: 1 scalar (energy) and 1 vector (force)
            out_channels_scalar = 1
            out_channels_vec = 1
            scalar_task_level = "graph"
            vector_task_level = "node"
        else:
            # Energy prediction only (forces from gradient)
            out_channels_scalar = 1
            out_channels_vec = 0
            scalar_task_level = "graph"
            vector_task_level = "graph"  # Not used for output, but required
        # --- End of dynamic configuration ---

        # Model specification
        solid_name = self.hparams.solid_name.lower()
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(
                f"Unsupported solid_name '{solid_name}'. Supported: {list(PLATONIC_GROUPS.keys())}"
            )

        if self.hparams.head_dim is not None:
            num_heads = self.hparams.hidden_dim // (
                self.hparams.head_dim * PLATONIC_GROUPS[solid_name].G
            )
            if (self.hparams.num_heads is not None) and (
                num_heads != self.hparams.num_heads
            ):
                raise ValueError(
                    f"head_dim {self.hparams.head_dim} does not match num_heads {self.hparams.num_heads}"
                )
            self.hparams.num_heads = num_heads

        self.net = PlatonicTransformer(
            input_dim=in_channels_scalar,
            input_dim_vec=in_channels_vector,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=out_channels_scalar,
            output_dim_vec=out_channels_vec,
            nhead=self.hparams.num_heads,
            num_layers=self.hparams.layers,
            solid_name=solid_name,
            spatial_dim=3,
            dense_mode=self.hparams.dense_mode,
            scalar_task_level=scalar_task_level,
            vector_task_level=vector_task_level,
            ffn_readout=self.hparams.ffn_readout,
            mean_aggregation=self.hparams.mean_aggregation,
            dropout=self.hparams.dropout,
            norm_first=self.hparams.norm_first,
            drop_path_rate=self.hparams.drop_path_rate,
            layer_scale_init_value=self.hparams.layer_scale,
            attention=self.hparams.attention,
            ffn_dim_factor=4,
            rope_sigma=self.hparams.rope_sigma,
            ape_sigma=self.hparams.ape_sigma,
            learned_freqs=self.hparams.learned_freqs,
            freq_init=self.hparams.freq_init,
            use_key=self.hparams.use_key,
        )

        # Initialize normalization parameters
        self.register_buffer("shift", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("avg_num_nodes", torch.tensor(1.0, dtype=torch.float32))

        # Setup metrics
        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.train_metric_force = torchmetrics.MeanAbsoluteError()
        self.train_metric_energy_per_atom = torchmetrics.MeanAbsoluteError()

        self.valid_metric = torchmetrics.MeanAbsoluteError()
        self.valid_metric_force = torchmetrics.MeanAbsoluteError()
        self.valid_metric_energy_per_atom = torchmetrics.MeanAbsoluteError()

        self.test_metrics_energy = torchmetrics.MeanAbsoluteError()
        self.test_metrics_force = torchmetrics.MeanAbsoluteError()
        self.test_metrics_energy_per_atom = torchmetrics.MeanAbsoluteError()

    def forward(self, graph):
        graph = graph.to(self.device)
        # Prepare input features
        x = [graph.x]
        if "coords" in self.hparams.scalar_features:
            x.append(graph.pos)
        if "charges" in self.hparams.scalar_features:
            x.append(graph.charges[:, None])

        x = torch.cat(x, dim=-1)

        # Forward pass
        pred_scalar, pred_vec = self.net(
            x,
            graph.pos,
            graph.batch,
            vec=None,
            avg_num_nodes=self.avg_num_nodes.to(graph.pos.device),
        )

        pred_energy = pred_scalar.view(-1)

        if self.hparams.predict_forces:
            # Squeeze the middle dimension: [N, 1, 3] -> [N, 3]
            pred_force = pred_vec.squeeze(1)
            return pred_energy, pred_force
        else:
            return pred_energy

    def pred_energy_and_force(self, graph):
        if self.hparams.predict_forces:
            # Model directly outputs energy and forces
            pred_energy, pred_force = self(graph)
            return pred_energy, pred_force
        else:
            # Calculate forces from energy gradient (autograd)
            with torch.enable_grad():
                graph.pos = graph.pos.clone().requires_grad_(True)
                pred_energy = self(graph)
                sign = -1.0
                pred_force = (
                    sign
                    * torch.autograd.grad(
                        pred_energy,
                        graph.pos,
                        grad_outputs=torch.ones_like(pred_energy),
                        create_graph=self.training,
                        retain_graph=self.training,
                    )[0]
                )

            if not self.training:
                pred_energy = pred_energy.detach()
                pred_force = pred_force.detach()

            return pred_energy, pred_force

    def training_step(self, graph, batch_idx):
        if self.hparams.train_augm:
            batch_size = graph.batch.max().item() + 1
            rots = self.rotation_generator(n=batch_size).type_as(graph.pos)
            rot_per_sample = rots[graph.batch]
            graph.pos = torch.einsum("bij,bj->bi", rot_per_sample, graph.pos)
            graph.forces = torch.einsum("bij,bj->bi", rot_per_sample, graph.forces)

        pred_energy, pred_force = self.pred_energy_and_force(graph)

        # Loss calculation
        energy_loss = torch.mean(
            (pred_energy - ((graph.energy - self.shift) / self.scale)) ** 2
        )
        force_loss = torch.mean(
            torch.sqrt(torch.sum((pred_force - graph.forces / self.scale) ** 2, -1))
        )
        loss = energy_loss + self.hparams.lambda_F * force_loss

        # Logging metrics (converted to meV and meV/Å)
        pred_energy_mev = (pred_energy.detach() * self.scale + self.shift) * 1000
        true_energy_mev = graph.energy * 1000
        pred_force_mev_ang = pred_force.detach() * self.scale * 1000
        true_force_mev_ang = graph.forces * 1000

        pred_energy_per_atom_mev = pred_energy_mev / graph.num_atoms
        true_energy_per_atom_mev = true_energy_mev / graph.num_atoms

        self.train_metric(pred_energy_mev, true_energy_mev)
        self.train_metric_force(pred_force_mev_ang, true_force_mev_ang)
        self.train_metric_energy_per_atom(
            pred_energy_per_atom_mev, true_energy_per_atom_mev
        )

        self.log(
            "train MAE (energy) [meV]",
            self.train_metric,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            "train MAE (force) [meV/Å]",
            self.train_metric_force,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            "train MAE (energy/atom) [meV]",
            self.train_metric_energy_per_atom,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        if batch_idx % 250 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, graph, batch_idx):
        pred_energy, pred_force = self.pred_energy_and_force(graph)

        pred_energy_mev = (pred_energy * self.scale + self.shift) * 1000
        true_energy_mev = graph.energy * 1000
        pred_force_mev_ang = pred_force * self.scale * 1000
        true_force_mev_ang = graph.forces * 1000

        pred_energy_per_atom_mev = pred_energy_mev / graph.num_atoms
        true_energy_per_atom_mev = true_energy_mev / graph.num_atoms

        self.valid_metric(pred_energy_mev, true_energy_mev)
        self.valid_metric_force(pred_force_mev_ang, true_force_mev_ang)
        self.valid_metric_energy_per_atom(
            pred_energy_per_atom_mev, true_energy_per_atom_mev
        )

    def on_validation_epoch_end(self):
        self.log(
            "valid MAE (energy) [meV]", self.valid_metric, prog_bar=True, sync_dist=True
        )
        self.log(
            "valid MAE (force) [meV/Å]",
            self.valid_metric_force,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "valid MAE (energy/atom) [meV]",
            self.valid_metric_energy_per_atom,
            prog_bar=True,
            sync_dist=True,
        )

    def test_step(self, graph, batch_idx):
        pred_energy, pred_force = self.pred_energy_and_force(graph)

        pred_energy_mev = (pred_energy * self.scale + self.shift) * 1000
        true_energy_mev = graph.energy * 1000
        pred_force_mev_ang = pred_force * self.scale * 1000
        true_force_mev_ang = graph.forces * 1000

        pred_energy_per_atom_mev = pred_energy_mev / graph.num_atoms
        true_energy_per_atom_mev = true_energy_mev / graph.num_atoms

        self.test_metrics_energy(pred_energy_mev, true_energy_mev)
        self.test_metrics_force(pred_force_mev_ang, true_force_mev_ang)
        self.test_metrics_energy_per_atom(
            pred_energy_per_atom_mev, true_energy_per_atom_mev
        )

    def on_test_epoch_end(self):
        self.log(
            "test MAE (energy) [meV]",
            self.test_metrics_energy,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "test MAE (force) [meV/Å]",
            self.test_metrics_force,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "test MAE (energy/atom) [meV]",
            self.test_metrics_energy_per_atom,
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn == "freqs":
                    no_decay.add(fpn)
                elif pn.endswith("bias") or ("layer_scale" in pn):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("kernel"):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        missing_params = param_dict.keys() - (decay | no_decay)
        if missing_params:
            print(
                f"Warning: Parameters {missing_params} were not explicitly assigned. Adding to no_decay."
            )
            no_decay.update(missing_params)

        assert (
            len(decay & no_decay) == 0
        ), f"Parameters in both decay and no_decay sets: {decay & no_decay}"

        optim_groups = [
            {
                "params": [
                    param_dict[p_name]
                    for p_name in sorted(list(decay))
                    if p_name in param_dict
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    param_dict[p_name]
                    for p_name in sorted(list(no_decay))
                    if p_name in param_dict
                ],
                "weight_decay": 0.0,
            },
        ]

        optim_groups = [group for group in optim_groups if group["params"]]

        optimizer = torch.optim.Adam(optim_groups, lr=self.hparams.lr)
        if self.hparams.cosine_scheduler:
            scheduler = CosineWarmupScheduler(
                optimizer, self.hparams.warmup, self.trainer.max_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "valid MAE (energy/atom) [meV]",
            }
        else:
            return {"optimizer": optimizer, "monitor": "valid MAE (energy/atom) [meV]"}


def main(args):
    pl.seed_everything(args.seed)

    train_loader, val_loader, test_loader, _, _ = get_omol_loaders(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_charges=False,
        seed=args.seed,
        debug_subset=args.debug_subset,
        referencing=args.referencing,
        include_hof=args.include_hof,
        scale_shift=args.scale_shift,
        recalculate=args.recalculate_stats,
        use_k_hot=args.use_khot_encoding,
    )

    accelerator = "gpu" if args.gpus > 0 and torch.cuda.is_available() else "cpu"
    devices = args.gpus if accelerator == "gpu" else "auto"

    logger = (
        pl.loggers.WandbLogger(
            project=args.wandb_project_name,
            name=None,
            config=vars(args),
            save_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs"),
            entity=args.wandb_identity,
        )
        if args.log
        else None
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="valid MAE (energy) [meV]",
            mode="min",
            filename="best-energy-{epoch:02d}",
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="valid MAE (force) [meV/Å]",
            mode="min",
            filename="best-force-{epoch:02d}",
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="valid MAE (energy/atom) [meV]",
            mode="min",
            filename="best-energy-per-atom-{epoch:02d}",
        ),
        pl.callbacks.ModelCheckpoint(save_last=True, filename="last"),
        TimerCallback(),
        MemoryMonitorCallback(log_frequency=20),
    ]
    if args.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))
    if args.timer:
        callbacks.append(Timer(duration=args.timer))

    if args.load_weights:
        model = OMolModel.load_from_checkpoint(
            checkpoint_path=args.load_weights, args=args
        )
    else:
        model = OMolModel(args)

    if hasattr(train_loader.dataset, "scale"):
        model.scale = torch.tensor(train_loader.dataset.scale).to(model.device)
        model.shift = torch.tensor(train_loader.dataset.shift).to(model.device)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=args.enable_progress_bar,
        precision=args.precision,
        inference_mode=False,
        strategy=DDPStrategy(find_unused_parameters=True) if args.gpus > 1 else "auto",
    )

    if not args.test_ckpt:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)
        best_ckpt_path = (
            callbacks[2].best_model_path if callbacks[2].best_model_path else "last"
        )
        trainer.test(model, test_loader, ckpt_path=best_ckpt_path)
    else:
        model = OMolModel.load_from_checkpoint(
            args.test_ckpt,
            hparams_file=os.path.join(os.path.dirname(args.test_ckpt), "hparams.yaml"),
            args=args,
        )
        trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OMol Energy and Force Prediction Training"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=40, help="Number of training epochs"
    )
    parser.add_argument(
        "--timer",
        type=str,
        default=None,
        help='Timer for training, e.g., "00:08:00:00"',
    )
    parser.add_argument("--warmup", type=int, default=0, help="Number of warmup epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--cosine_scheduler",
        type=eval,
        default=True,
        help="Use cosine annealing scheduler",
    )
    parser.add_argument(
        "--lambda_F", type=float, default=5.0, help="Weight for force loss"
    )

    # Model architecture
    parser.add_argument(
        "--hidden_dim", type=int, default=1152, help="Hidden dimension of the model"
    )
    parser.add_argument(
        "--layers", type=int, default=14, help="Number of layers in the model"
    )
    parser.add_argument(
        "--drop_path_rate", type=float, default=0.0, help="Stochastic depth rate"
    )
    parser.add_argument(
        "--predict_forces",
        type=eval,
        default=True,
        help="Enable direct force prediction instead of using gradients.",
    )

    # Platonic Transformer specific parameters
    parser.add_argument(
        "--solid_name",
        type=str,
        default="tetrahedron",
        help="Group name for Platonic solids (Platoformer only)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=72,
        help="Number of attention heads (Platoformer only)",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=None,
        help="Implicitly defines number of heads (Platoformer only)",
    )
    parser.add_argument(
        "--rope_sigma", type=float, default=4, help="Sigma for RoPE (Platoformer only)"
    )
    parser.add_argument(
        "--ape_sigma", type=eval, default=None, help="Sigma for APE (Platoformer only)"
    )
    parser.add_argument(
        "--freq_init",
        type=str,
        default="spiral",
        choices=["random", "spiral"],
        help="Frequency init for RoPE (Platoformer only)",
    )
    parser.add_argument(
        "--use_key",
        type=eval,
        default=False,
        help="Use key projection with RoPE (Platoformer only)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout rate (Platoformer only)"
    )
    parser.add_argument(
        "--norm_first",
        type=eval,
        default=True,
        help="Use LayerNorm before attention (Platoformer only)",
    )
    parser.add_argument(
        "--learned_freqs",
        type=eval,
        default=True,
        help="Learn frequencies for RoPE (Platoformer only)",
    )
    parser.add_argument(
        "--dense_mode",
        type=eval,
        default=False,
        help="Enable dense attention blocks (Platoformer only)",
    )
    parser.add_argument(
        "--mean_aggregation",
        type=eval,
        default=False,
        help="Use mean aggregation instead of sum (Platoformer only)",
    )
    parser.add_argument(
        "--attention",
        type=eval,
        default=True,
        help="Use attention in PlatonicConv (Platoformer only)",
    )
    parser.add_argument(
        "--ffn_readout",
        type=eval,
        default=True,
        help="Feed-forward readout (Platoformer only)",
    )

    parser.add_argument(
        "--layer_scale",
        type=eval,
        default=None,
        help="Layer scaling factor for PlatonicTransformer",
    )

    # Training features
    parser.add_argument(
        "--train_augm", type=eval, default=True, help="Use rotation augmentation"
    )

    # Input features
    parser.add_argument(
        "--scalar_features",
        type=eval,
        default=[],
        help='Additional scalar features, e.g., ["coords", "charges"]',
    )
    parser.add_argument(
        "--vector_features",
        type=eval,
        default=[],
        help='Additional vector features, e.g., ["coords"]',
    )

    # Data and logging
    # parser.add_argument('--data_dir', type=str, default='datasets/omol/', help='Data directory')
    # parser.add_argument('--data_dir', type=str, default='/scratch/islam_omol/ivi_omol/', help='Data directory')
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/opt/datasets/ivi/ivi_omol",
        help="Data directory",
    )
    parser.add_argument(
        "--debug_subset",
        type=int,
        default=None,
        help="Use a subset of the dataset for debugging",
    )
    parser.add_argument(
        "--recalculate_stats",
        type=eval,
        default=False,
        help="Recalculate dataset statistics",
    )
    parser.add_argument(
        "--referencing",
        type=eval,
        default=True,
        help="Use per-atom referencing for the target energy",
    )
    parser.add_argument(
        "--include_hof",
        type=eval,
        default=False,
        help="Normalize target using HOF values",
    )
    parser.add_argument(
        "--scale_shift",
        type=eval,
        default=False,
        help="Use scale and shift normalization",
    )
    parser.add_argument(
        "--use_khot_encoding",
        type=eval,
        default=True,
        help="Use k-hot encoding for atom types",
    )

    parser.add_argument(
        "--config", type=eval, default=None, help="Sweep configuration dictionary"
    )
    parser.add_argument(
        "--model_id",
        type=int,
        default=None,
        help="Model ID in case you would want to label the configuration",
    )

    # System and checkpointing
    parser.add_argument("--log", type=eval, default=True, help="Enable logging")
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="Platonic-OMol",
        help="WandB project name",
    )
    parser.add_argument("--wandb_identity", type=str, default=None, help="WandB entity")
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs to use (0 for CPU)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["16-mixed", "bf16-mixed", "32"],
        help="Training precision",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of data loading workers"
    )
    parser.add_argument(
        "--enable_progress_bar", type=eval, default=True, help="Show progress bar"
    )
    parser.add_argument(
        "--test_ckpt", type=str, default=None, help="Path to a checkpoint for testing"
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    parser.add_argument(
        "--load_weights",
        type=str,
        default=None,
        help="Path to load model weights from for a new run",
    )

    args = parser.parse_args()

    if args.config is not None:
        for key, value in args.config.items():
            setattr(args, key, value)

    main(args)
