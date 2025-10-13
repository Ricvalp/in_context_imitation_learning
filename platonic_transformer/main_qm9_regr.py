import os
import argparse

import torch
import torchmetrics
import numpy as np
import pytorch_lightning as pl
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from datasets.k_hot_encoding import KHOT_EMBEDDINGS

from models.platoformer.platoformer import PlatonicTransformer
from models.platoformer.groups import PLATONIC_GROUPS

from utils import (
    CosineWarmupScheduler,
    RandomSOd,
    TimerCallback,
    StopOnPersistentDivergence,
)
from pytorch_lightning.callbacks import Timer

from models.platoformer.utils import scatter_add

# Performance optimizations
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


class QM9Model(pl.LightningModule):
    """Lightning module for QM9 molecular property prediction using PlatonicTransformer."""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)  # Saves all args passed to __init__

        # Initialize k-hot embedding tensor if needed
        if self.hparams.use_k_hot_encoding:
            self._init_khot_embeddings()

        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)

        # Standard input dimension from QM9 graph.x (atom type embeddings)
        # For QM9, graph.x typically has 11 features.
        input_feature_dimensionality = 97 if self.hparams.use_k_hot_encoding else 11

        solid_name = self.hparams.solid_name.lower()
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(
                f"Unsupported solid_name '{solid_name}'. Supported: {list(PLATONIC_GROUPS.keys())}"
            )

        # This sets the number of heads in case head_dim is specified.
        if self.hparams.head_dim is not None:
            num_heads = self.hparams.hidden_dim // (
                self.hparams.head_dim * PLATONIC_GROUPS[solid_name.lower()].G
            )
            if (self.hparams.num_heads is not None) and (
                num_heads != self.hparams.num_heads
            ):
                raise ValueError(
                    f"head_dim {self.hparams.head_dim} does not match num_heads {self.hparams.num_heads} "
                )
            self.hparams.num_heads = num_heads

        self.net = PlatonicTransformer(
            # Basic/essential specification:
            input_dim=input_feature_dimensionality,
            input_dim_vec=0,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=1,  # Single target property prediction
            output_dim_vec=0,
            nhead=self.hparams.num_heads,
            num_layers=self.hparams.layers,  # Using 'layers' arg for num_layers
            solid_name=solid_name,
            spatial_dim=3,
            dense_mode=self.hparams.dense_mode,
            # Pooling and readout specification:
            scalar_task_level="graph",
            vector_task_level="graph",
            ffn_readout=self.hparams.ffn_readout,
            # Attention block specification:
            mean_aggregation=self.hparams.mean_aggregation,
            dropout=self.hparams.dropout,
            norm_first=self.hparams.norm_first,
            drop_path_rate=self.hparams.drop_path_rate,
            layer_scale_init_value=self.hparams.layer_scale_init_value,
            attention=self.hparams.attention,
            ffn_dim_factor=4,
            # RoPE and APE specification:
            rope_sigma=self.hparams.rope_sigma,
            ape_sigma=self.hparams.ape_sigma,
            learned_freqs=self.hparams.learned_freqs,
            freq_init=self.hparams.freq_init,
            use_key=self.hparams.use_key,
        )
        # self.net = torch.compile(self.net)

        # Initialize normalization parameters
        self.register_buffer("shift", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))

        # Setup metrics
        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.valid_metric = torchmetrics.MeanAbsoluteError()
        self.test_metric = torchmetrics.MeanAbsoluteError()

    def _init_khot_embeddings(self):
        """Initialize k-hot embedding tensor for fast lookup."""
        # QM9 has atoms with atomic numbers 1, 6, 7, 8, 9 (H, C, N, O, F)
        embedding_dim = len(next(iter(KHOT_EMBEDDINGS.values())))
        embedding_tensor = torch.zeros(5, embedding_dim, dtype=torch.float32)
        qm9_to_atomic = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}
        for qm9_idx, atomic_num in qm9_to_atomic.items():
            if atomic_num in KHOT_EMBEDDINGS:
                embedding_tensor[qm9_idx] = torch.tensor(KHOT_EMBEDDINGS[atomic_num])
        self.register_buffer("khot_embedding_tensor", embedding_tensor)

    def include_k_hot_encoding(self, x):
        if not self.hparams.use_k_hot_encoding:
            return x
        atom_onehot = x[:, :5]
        atom_indices = torch.argmax(atom_onehot, dim=-1)
        embeddings = self.khot_embedding_tensor[atom_indices]
        return torch.cat([embeddings, x[:, -5:]], dim=-1)

    def forward(self, graph):
        # Use the class method instead of the global function
        node_features = self.include_k_hot_encoding(graph.x)

        positions = graph.pos  # [N, 3]
        positions_mean = scatter_add(
            graph.pos, graph.batch, dim_size=graph.batch.max().item() + 1
        ) / scatter_add(
            torch.ones_like(graph.pos[:, :1]),
            graph.batch,
            dim_size=graph.batch.max().item() + 1,
        )
        positions = (
            positions - positions_mean[graph.batch]
        )  # Center positions per graph
        batch_idx = graph.batch  # [N]

        # PlatonicTransformer expects x, pos, batch
        pred, _ = self.net(
            node_features,
            positions,
            batch_idx,
            vec=None,
            avg_num_nodes=self.avg_num_nodes,
        )

        return pred.squeeze(-1)  # Assuming output_dim is 1

    def set_dataset_statistics(self, dataloader):
        """
        Compute and cache or load the mean and standard deviation of the target property.

        The statistics are saved to a file named 'stats_{target_name}.npz' in the
        dataset's root directory to avoid re-computation on subsequent runs.
        """
        stats_file = os.path.join(
            self.hparams.data_dir, f"stats_{self.hparams.target}.npz"
        )

        if os.path.exists(stats_file):
            print(f"Loading dataset statistics from cached file: {stats_file}")
            stats = np.load(stats_file)
            self.shift = torch.tensor(stats["shift"])
            self.scale = torch.tensor(stats["scale"])
            self.avg_num_nodes = torch.tensor(stats["avg_num_nodes"])
        else:
            print("Computing dataset statistics...")
            ys = []
            total_num_nodes = 0
            for data in dataloader:
                ys.append(data.y)
                total_num_nodes += data.num_nodes
            ys = np.concatenate(ys)

            self.shift = torch.tensor(np.mean(ys))
            self.scale = torch.tensor(np.std(ys))
            self.avg_num_nodes = torch.tensor(total_num_nodes / len(dataloader.dataset))

            print(f"Saving dataset statistics to {stats_file}")
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            np.savez(
                stats_file,
                shift=self.shift,
                scale=self.scale,
                avg_num_nodes=self.avg_num_nodes,
            )

        print(f"Target statistics - Mean: {self.shift:.4f}, Std: {self.scale:.4f}")

    def training_step(self, graph, batch_idx):  #
        # Apply rotation augmentation if enabled
        if self.hparams.train_augm:
            batch_size = graph.batch.max().item() + 1
            rots = self.rotation_generator(n=batch_size).type_as(graph.pos)
            rot_per_sample = rots[graph.batch]
            graph.pos = torch.einsum("bij,bj->bi", rot_per_sample, graph.pos)

        pred = self(graph)
        loss = torch.mean(torch.abs(pred - (graph.y - self.shift) / self.scale))
        self.train_metric(pred * self.scale + self.shift, graph.y)
        return loss

    def validation_step(self, graph, batch_idx):  #
        pred = self(graph)
        self.valid_metric(pred * self.scale + self.shift, graph.y)

    def test_step(self, graph, batch_idx):  #
        pred = self(graph)
        self.test_metric(pred * self.scale + self.shift, graph.y)

    def on_train_epoch_end(self):  #
        self.log("train MAE", self.train_metric, prog_bar=True)

    def on_validation_epoch_end(self):  #
        self.log("valid MAE", self.valid_metric, prog_bar=True)

    def on_test_epoch_end(self):  #
        self.log("test MAE", self.test_metric, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer with weight decay and learning rate schedule."""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():  # mn is module name, m is module instance
            for (
                pn,
                p,
            ) in (
                m.named_parameters()
            ):  # pn is parameter name (e.g., 'weight', 'bias', 'freqs')
                fpn = f"{mn}.{pn}" if mn else pn  # fpn is the full parameter name

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
                # Parameters not matching any rule will be caught later and added to no_decay by default.

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        current_params_in_groups = decay | no_decay
        missing_params = param_dict.keys() - current_params_in_groups
        if missing_params:
            print(
                f"Warning: Parameters {missing_params} were not explicitly assigned to decay/no_decay by specific rules. Adding to no_decay by default."
            )
            no_decay.update(missing_params)  # Add missing parameters to no_decay group

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"Parameters {inter_params} found in both decay and no_decay sets!"

        # Ensure all learnable parameters are covered
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"Parameters {param_dict.keys() - union_params} not assigned to any optimizer group!"

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

        # Filter out empty groups (e.g., if 'decay' set is empty)
        optim_groups = [group for group in optim_groups if group["params"]]

        if not optim_groups and list(
            param_dict.keys()
        ):  # Should not happen if there are learnable params
            raise ValueError(
                "No optimizer groups were created, but there are learnable parameters."
            )
        elif not optim_groups and not list(param_dict.keys()):  # No learnable params
            print("Warning: No learnable parameters found for the optimizer.")

        optimizer = torch.optim.Adam(optim_groups, lr=self.hparams.lr)
        if self.hparams.cosine_scheduler:
            scheduler = CosineWarmupScheduler(
                optimizer, self.hparams.warmup, self.trainer.max_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "valid MAE",
            }
        else:
            return {"optimizer": optimizer, "monitor": "valid MAE"}


def load_data(args):
    """Load and preprocess QM9 dataset."""
    # Load dataset
    dataset = QM9(root=args.data_dir)

    # Create train/val/test split (same as DimeNet)
    random_state = np.random.RandomState(seed=42)
    perm = torch.from_numpy(random_state.permutation(np.arange(130831)))
    train_idx, val_idx, test_idx = perm[:110000], perm[110000:120000], perm[120000:]
    datasets = {
        "train": dataset[train_idx],
        "val": dataset[val_idx],
        "test": dataset[test_idx],
    }

    # Select target property
    targets = [
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "U0",
        "U",
        "H",
        "G",
        "Cv",
        "U0_atom",
        "U_atom",
        "H_atom",
        "G_atom",
        "A",
        "B",
        "C",
    ]
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11, 12, 13, 14, 15])
    dataset.data.y = dataset.data.y[:, idx]
    dataset.data.y = dataset.data.y[:, targets.index(args.target)]

    # Create dataloaders
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
        )
        for split, dataset in datasets.items()
    }

    return dataloaders["train"], dataloaders["val"], dataloaders["test"]


def main(args):  #
    pl.seed_everything(args.seed)

    train_loader, val_loader, test_loader = load_data(args)

    if args.gpus > 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = "cpu"
        devices = "auto"

    if args.log:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(
            project=f"Platonic-QM9-Regr",
            name=None,
            config=vars(args),  # Log all hyperparameters
            save_dir=save_dir,
            entity=args.wandb_identity,
        )
    else:
        logger = None

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="valid MAE", mode="min", every_n_epochs=1, save_last=True
        ),
        TimerCallback(),
    ]
    if args.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))
    if args.timer is not None:
        callbacks.append(Timer(duration=args.timer))
    callbacks.append(
        StopOnPersistentDivergence(
            monitor="valid MAE",
            threshold=1.0,
            patience=10,
            grace_epochs=20,
            verbose=False,
        )
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=args.enable_progress_bar,
        precision=args.precision,
    )

    if args.test_ckpt is None:
        model = QM9Model(args)  # Pass all args
        model.set_dataset_statistics(train_loader)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)
        # Test with the best checkpoint from training
        best_ckpt_path = (
            callbacks[0].best_model_path if callbacks[0].best_model_path else "last"
        )
        trainer.test(model, test_loader, ckpt_path=best_ckpt_path)

    else:
        # When loading from checkpoint, ensure hparams are available or pass args
        model = QM9Model.load_from_checkpoint(
            args.test_ckpt,
            hparams_file=os.path.join(os.path.dirname(args.test_ckpt), "hparams.yaml"),
            args=args,
        )
        model.set_dataset_statistics(
            train_loader
        )  # Recompute stats or ensure they are loaded
        trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QM9 Property Prediction Training")

    # General training parameters
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of training epochs."
    )  # Adjusted default
    parser.add_argument(
        "--timer",
        type=str,
        default=None,
        help='Timer for training, e.g., "00:08:00:00".',
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup epochs for cosine scheduler.",
    )
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="Weight decay."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )  # Changed default
    parser.add_argument(
        "--cosine_scheduler",
        type=eval,
        default=True,
        help="Use cosine annealing scheduler.",
    )

    # Common model architecture parameters
    parser.add_argument(
        "--hidden_dim", type=int, default=1152, help="Hidden dimension for the model."
    )
    parser.add_argument(
        "--layers", type=int, default=14, help="Number of layers in the model."
    )

    # Platonic Transformer specific parameters
    parser.add_argument(
        "--num_heads",
        type=int,
        default=72,
        help="Number of attention heads (Transformer only).",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=None,
        help="Implicitly defines number of heads (Transformer only).",
    )
    parser.add_argument(
        "--rope_sigma",
        type=eval,
        default=1.0,
        help="Sigma for RFF positional encoding (Transformer only).",
    )  # set to 4 if using spiral init
    parser.add_argument(
        "--use_key",
        type=eval,
        default=False,
        help="Use key projection when using RoPE (Transformer only).",
    )
    parser.add_argument(
        "--ape_sigma",
        type=eval,
        default=0.5,
        help="Sigma for RFF positional encoding (Transformer only).",
    )
    parser.add_argument(
        "--freq_init",
        type=str,
        default="random",
        choices=["random", "spiral"],
        help="Frequency initialization method for RoPE (Transformer only).",
    )  # combo spiral-rope_sigma=4 or random-rope_sigma=1
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout rate (Transformer only)."
    )
    parser.add_argument(
        "--norm_first",
        type=eval,
        default=True,
        help="Use LayerNorm before attention in Transformer.",
    )
    parser.add_argument(
        "--use_rope",
        type=eval,
        default=True,
        help="Use Rotary Position Embedding (RoPE) in Transformer.",
    )
    parser.add_argument(
        "--learned_freqs",
        type=eval,
        default=True,
        help="Use Rotary Position Embedding (RoPE) in Transformer.",
    )
    parser.add_argument(
        "--dense_mode", type=eval, default=True, help="Enable dense attention blocks."
    )
    parser.add_argument(
        "--solid_name",
        type=str,
        default="octahedron",
        help="tetrahedron, trivial_3, octahedron, icosahedron",
    )
    parser.add_argument(
        "--mean_aggregation",
        type=eval,
        default=False,
        help="Use mean aggregation instead of sum.",
    )
    parser.add_argument(
        "--attention",
        type=eval,
        default=True,
        help="Use attention in PlatonicConv (Transformer only).",
    )
    parser.add_argument(
        "--ffn_readout",
        type=eval,
        default=True,
        help="Feed-forward readout (Transformer only).",
    )
    parser.add_argument(
        "--layer_scale_init_value",
        type=float,
        default=None,
        help="Initial value for LayerScale in Platonic Transformer (default: disabled)",
    )
    parser.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.0,
        help="Stochastic depth rate (uniform).",
    )

    # Training features
    parser.add_argument(
        "--train_augm",
        type=eval,
        default=True,
        help="Use rotation augmentation during training.",
    )  #
    parser.add_argument(
        "--use_k_hot_encoding",
        type=eval,
        default=True,
        help="Use k-hot encoding for atom types instead of one-hot.",
    )  #
    # Data and logging
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets/qm9",
        help="Directory for QM9 dataset.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="mu",
        help="Target molecular property to predict from QM9.",
    )
    parser.add_argument(
        "--log", type=eval, default=True, help="Enable logging (e.g., to WandB)."
    )
    parser.add_argument(
        "--wandb_identity",
        type=str,
        default="m-m-islam-university-of-amsterdam",
        help="use <m-m-islam-university-of-amsterdam> for teamspace, or your own identity",
    )

    # Sweep configuration (remains for hyperparameter optimization)
    parser.add_argument(
        "--config",
        type=eval,
        default=None,
        help="Sweep configuration dictionary (e.g., from WandB).",
    )
    parser.add_argument(
        "--model_id",
        type=int,
        default=None,
        help="Model ID for labeling configurations.",
    )

    # System and checkpointing
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs to use (0 for CPU)."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["16-mixed", "bf16-mixed", "32"],
        help="Precision for training: 16 or 32.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of data loading workers."
    )
    parser.add_argument(
        "--enable_progress_bar",
        type=eval,
        default=True,
        help="Show progress bar during training.",
    )
    parser.add_argument(
        "--test_ckpt", type=str, default=None, help="Path to a checkpoint for testing."
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )

    args = parser.parse_args()

    if args.config is not None:  #
        for key, value in args.config.items():
            setattr(args, key, value)

    main(args)
