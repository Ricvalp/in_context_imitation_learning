from functools import partial
import os
import argparse
import sys

import torch
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_cluster import knn_graph
import numpy as np

# Performance optimizations
torch.set_float32_matmul_precision("medium")

try:
    import wandb
    from matplotlib import cm

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ==================================================================================================
# shapenet_car.py
# Modified from https://github.com/ml-jku/UPT/blob/main/src/datasets/shapenet_car.py.
# ==================================================================================================


def num_nodes_to_batch_idx(num_nodes):
    return (
        torch.arange(len(num_nodes)).to(num_nodes.device).repeat_interleave(num_nodes)
    )


class ShapenetCarDataset(torch.utils.data.Dataset):
    # generated with torch.randperm(889, generator=torch.Generator().manual_seed(0))[:189]
    TEST_INDICES = {
        550,
        592,
        229,
        547,
        62,
        464,
        798,
        836,
        5,
        732,
        876,
        843,
        367,
        496,
        142,
        87,
        88,
        101,
        303,
        352,
        517,
        8,
        462,
        123,
        348,
        714,
        384,
        190,
        505,
        349,
        174,
        805,
        156,
        417,
        764,
        788,
        645,
        108,
        829,
        227,
        555,
        412,
        854,
        21,
        55,
        210,
        188,
        274,
        646,
        320,
        4,
        344,
        525,
        118,
        385,
        669,
        113,
        387,
        222,
        786,
        515,
        407,
        14,
        821,
        239,
        773,
        474,
        725,
        620,
        401,
        546,
        512,
        837,
        353,
        537,
        770,
        41,
        81,
        664,
        699,
        373,
        632,
        411,
        212,
        678,
        528,
        120,
        644,
        500,
        767,
        790,
        16,
        316,
        259,
        134,
        531,
        479,
        356,
        641,
        98,
        294,
        96,
        318,
        808,
        663,
        447,
        445,
        758,
        656,
        177,
        734,
        623,
        216,
        189,
        133,
        427,
        745,
        72,
        257,
        73,
        341,
        584,
        346,
        840,
        182,
        333,
        218,
        602,
        99,
        140,
        809,
        878,
        658,
        779,
        65,
        708,
        84,
        653,
        542,
        111,
        129,
        676,
        163,
        203,
        250,
        209,
        11,
        508,
        671,
        628,
        112,
        317,
        114,
        15,
        723,
        746,
        765,
        720,
        828,
        662,
        665,
        399,
        162,
        495,
        135,
        121,
        181,
        615,
        518,
        749,
        155,
        363,
        195,
        551,
        650,
        877,
        116,
        38,
        338,
        849,
        334,
        109,
        580,
        523,
        631,
        713,
        607,
        651,
        168,
    }

    def __init__(
        self,
        data_path,
        split,
        knn=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.split = split
        self.knn = knn
        self.domain_min = torch.tensor([-2.0, -1.0, -4.5])
        self.domain_max = torch.tensor([2.0, 4.5, 6.0])
        self.mean = torch.tensor(-36.3099)
        self.std = torch.tensor(48.5743)

        # discover uris
        self.uris = []
        for i in range(9):
            param_uri = f"{self.data_path}/param{i}"
            if not os.path.exists(param_uri):
                continue
            for name in sorted(os.listdir(param_uri)):
                sample_uri = f"{param_uri}/{name}"
                if os.path.isdir(sample_uri):
                    self.uris.append(sample_uri)

        assert len(self.uris) == 889, f"found {len(self.uris)} uris instead of 889."
        # split into train/test uris
        if split == "train":
            train_idxs = [
                i for i in range(len(self.uris)) if i not in self.TEST_INDICES
            ]
            self.uris = [self.uris[train_idx] for train_idx in train_idxs]
            assert len(self.uris) == 700
        elif split == "test":
            self.uris = [self.uris[test_idx] for test_idx in self.TEST_INDICES]
            assert len(self.uris) == 189
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.uris)

    def getitem_pressure(self, idx):
        p = torch.load(f"{self.uris[idx]}/pressure.th", weights_only=True)
        p -= self.mean
        p /= self.std
        return p

    def getitem_all_pos(self, idx):
        all_pos = torch.load(f"{self.uris[idx]}/mesh_points.th", weights_only=True)
        all_pos.sub_(self.domain_min).div_(self.domain_max - self.domain_min)
        all_pos = (all_pos - 0.4063) / 0.1531  # normalize
        return all_pos

    def __getitem__(self, idx):
        all_pos = self.getitem_all_pos(idx)
        pressure = self.getitem_pressure(idx)

        # Apply the same permutation to both tensors
        num_points = len(all_pos)
        perm_indices = torch.randperm(num_points)

        all_pos = all_pos[perm_indices]
        pressure = pressure[perm_indices]

        output = {
            "node_positions": all_pos,
            "target": pressure,
            "num_nodes": torch.LongTensor([num_points]),
        }

        return output

    def collate_fn(self, batch):
        batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}
        batch["batch_idx"] = num_nodes_to_batch_idx(batch["num_nodes"])
        if self.knn > 0:
            batch["edge_index"] = knn_graph(
                batch["node_positions"], k=self.knn, batch=batch["batch_idx"], loop=True
            )
        return batch


# ==================================================================================================
# Main script for pressure prediction
# ==================================================================================================
from utils import CosineWarmupScheduler, TimerCallback
from pytorch_lightning.callbacks import Timer

from models.platoformer.platoformer import PlatonicTransformer
from models.platoformer.groups import PLATONIC_GROUPS


class PressureModel(pl.LightningModule):
    """Lightning module for ShapenetCar pressure prediction."""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # This task predicts a scalar, so no vector features are outputted.
        self.lifted = False

        # Calculate total input channels
        in_channels_scalar = 1 + 3 * (  # base node features (ones)
            "coords" in self.hparams.scalar_features
        )
        in_channels_vector = 1 * ("coords" in self.hparams.vector_features)

        # Calculate output dimensions for scalar pressure prediction
        out_channels_scalar = 1
        out_channels_vec = 0

        # Model specification
        if self.hparams.equivariance == "Tn":
            solid_name = "trivial"
        elif self.hparams.equivariance == "SEn":
            solid_name = "tetrahedron"
        else:
            raise ValueError(
                f"Unsupported equivariance type: {self.hparams.equivariance}. "
                "Supported types are 'Tn' (trivial) and 'SEn' (tetrahedron)."
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
            input_dim=in_channels_scalar,
            input_dim_vec=in_channels_vector,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=out_channels_scalar,
            output_dim_vec=out_channels_vec,
            nhead=self.hparams.num_heads,
            num_layers=self.hparams.layers,
            solid_name=solid_name,
            ffn_dim_factor=4,
            task_level="node",
            dropout=self.hparams.dropout,
            norm_first=self.hparams.norm_first,
            freq_sigma=self.hparams.freq_sigma,
            learned_freqs=self.hparams.learned_freqs,
            spatial_dim=3,
            dense_mode=self.hparams.dense_mode,
            mean_aggregation=self.hparams.mean_aggregation,
            attention=self.hparams.attention,
            ffn_readout=self.hparams.ffn_readout,
        )

        # Setup metrics
        self.train_metric = torchmetrics.MeanSquaredError()
        self.valid_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()

        # Store all test batches and predictions for visualization
        self.all_test_batches = []
        self.all_test_preds = []

    def set_dataset_statistics(self, train_loader):
        # Calculate average number of nodes for Platoformer
        num_nodes_list = [d["num_nodes"].item() for d in train_loader.dataset]
        self.avg_num_nodes = sum(num_nodes_list) / len(num_nodes_list)

        # Store dataset normalization constants
        self.dataset_mean = train_loader.dataset.mean
        self.dataset_std = train_loader.dataset.std

    def forward(self, graph):
        # Compatibility with model inputs
        pos = graph["node_positions"]
        batch = graph["batch_idx"]
        edge_index = graph["edge_index"]

        # Prepare input features
        x = []
        vec = []

        # Add scalar features
        if "coords" in self.hparams.scalar_features:
            x.append(pos)

        # Add vector features
        if "coords" in self.hparams.vector_features:
            vec.append(pos[:, None, :])

        # Combine features
        x = torch.cat(x, dim=-1) if x else None
        vec = torch.cat(vec, dim=1) if vec else None

        # Create a dummy scalar feature if no other scalar features are provided
        if x is None and vec is not None:
            x = torch.ones(pos.size(0), 1).type_as(pos)

        # Forward pass
        pred_scalar, _ = self.net(
            x, pos, batch, vec=vec, avg_num_nodes=self.avg_num_nodes
        )

        return pred_scalar.squeeze(-1)

    def training_step(self, graph, batch_idx):
        y_pred = self(graph)
        loss = torch.nn.functional.mse_loss(y_pred, graph["target"])
        self.train_metric(y_pred, graph["target"])
        return loss

    def validation_step(self, graph, batch_idx):
        y_pred = self(graph)
        self.valid_metric(y_pred, graph["target"])

    def test_step(self, graph, batch_idx):
        y_pred = self(graph)
        self.test_metric.update(y_pred, graph["target"])

        # Store all batches and predictions for visualization at the end of the epoch
        self.all_test_batches.append({k: v.cpu() for k, v in graph.items()})
        self.all_test_preds.append(y_pred.cpu())

    def on_train_epoch_end(self):
        self.log("train_mse", self.train_metric, prog_bar=True)
        self.train_metric.reset()

    def on_validation_epoch_end(self):
        valid_mse = self.valid_metric.compute()
        unnormalized_valid_rmse = torch.sqrt(valid_mse) * self.dataset_std
        self.log("valid_rmse_unnormalized", unnormalized_valid_rmse, prog_bar=True)
        self.log("valid_mse", valid_mse, prog_bar=True)
        self.valid_metric.reset()

    def on_test_epoch_end(self):
        # First, log the summary metrics for the entire test set
        test_mse = self.test_metric.compute()
        unnormalized_test_rmse = torch.sqrt(test_mse) * self.dataset_std
        self.log("test_rmse_unnormalized", unnormalized_test_rmse, prog_bar=True)
        self.log("test_mse", test_mse, prog_bar=True)
        self.test_metric.reset()

        # Then, if batches were saved, log the 3D visualizations iteratively
        if self.all_test_batches and self.logger and WANDB_AVAILABLE:
            # Iterate over each saved batch
            for i, batch in enumerate(self.all_test_batches):
                predictions = self.all_test_preds[i]
                # Determine the number of individual samples in the batch
                num_samples_in_batch = batch["batch_idx"].max().item() + 1

                # Iterate over each sample within the batch
                for sample_idx_in_batch in range(num_samples_in_batch):
                    # Find the indices corresponding to the current sample
                    sample_indices = torch.where(
                        batch["batch_idx"] == sample_idx_in_batch
                    )[0]

                    if len(sample_indices) == 0:
                        continue

                    # Extract data for this single sample
                    points = batch["node_positions"][sample_indices]
                    gt_pressures = batch["target"][sample_indices]
                    pred_pressures = predictions[sample_indices]

                    # Un-normalize pressures for intuitive coloring
                    gt_pressures = gt_pressures * self.dataset_std + self.dataset_mean
                    pred_pressures = (
                        pred_pressures * self.dataset_std + self.dataset_mean
                    )

                    # Create a shared color scale for direct comparison
                    vmin = min(gt_pressures.min(), pred_pressures.min())
                    vmax = max(gt_pressures.max(), pred_pressures.max())

                    # Log this single sample. The logger handles the step increment,
                    # creating a slider in the UI for these keys.
                    self.logger.experiment.log(
                        {
                            "test_visualizations/Ground_Truth": self._create_point_cloud_log(
                                points, gt_pressures, vmin, vmax
                            ),
                            "test_visualizations/Prediction": self._create_point_cloud_log(
                                points, pred_pressures, vmin, vmax
                            ),
                        }
                    )

        # Clear the stored batches and predictions for the next potential test run
        self.all_test_batches.clear()
        self.all_test_preds.clear()

    def _create_point_cloud_log(self, points, pressures, vmin, vmax):
        """Helper function to create a wandb.Object3D for a point cloud."""
        # Normalize pressures to [0, 1] for colormapping, with epsilon for stability
        normalized_pressures = (pressures - vmin) / (vmax - vmin + 1e-9)

        # Get colors from a matplotlib colormap
        colors = cm.viridis(normalized_pressures.numpy())[:, :3] * 255

        # Combine points and colors
        points_np = points.numpy()
        colored_points = np.concatenate((points_np, colors), axis=1)

        return wandb.Object3D(colored_points)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn == "freqs" or pn.endswith("bias") or ("layer_scale" in pn):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("kernel"):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        no_decay.update(param_dict.keys() - (decay | no_decay))
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"Parameters {inter_params} found in both decay and no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"Parameters {param_dict.keys() - union_params} not assigned to any optimizer group!"

        optim_groups = [
            {
                "params": [param_dict[p_name] for p_name in sorted(list(decay))],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [param_dict[p_name] for p_name in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.Adam(optim_groups, lr=self.hparams.lr)
        if self.hparams.cosine_scheduler:
            scheduler = CosineWarmupScheduler(
                optimizer, self.hparams.warmup, self.trainer.max_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "valid_mse",
            }
        else:
            return {"optimizer": optimizer, "monitor": "valid_mse"}


def load_data(args):
    """Load and preprocess ShapenetCar dataset."""
    train_dataset = ShapenetCarDataset(
        data_path=args.data_dir, split="train", knn=args.knn
    )
    # Use the test set for both validation and testing
    test_dataset = ShapenetCarDataset(
        data_path=args.data_dir, split="test", knn=args.knn
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
    )

    return train_loader, val_loader, test_loader


def main(args):

    pl.seed_everything(args.seed)

    train_loader, val_loader, test_loader = load_data(args)

    if args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = "cpu"
        devices = "auto"

    if args.log:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(
            project="Shapenet-Car-Pressure",
            name=f"platoformer-equiv_{args.equivariance}",
            config=args,
            save_dir=save_dir,
            entity="erikjbekkers",
        )
    else:
        logger = None

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="valid_mse", mode="min", every_n_epochs=1, save_last=True
        ),
        TimerCallback(),
    ]
    if args.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))
    if args.timer is not None:
        callbacks.append(Timer(duration=args.timer))

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=args.enable_progress_bar,
    )

    if args.test_ckpt is None:
        model = PressureModel(args)
        model.set_dataset_statistics(train_loader)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)
        trainer.test(model, test_loader, ckpt_path="best")
    else:
        model = PressureModel.load_from_checkpoint(args.test_ckpt)
        model.set_dataset_statistics(train_loader)
        trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shapenet-Car Pressure Prediction Training"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--timer",
        type=str,
        default=None,
        help='Timer for training, e.g., "00:08:00:00"',
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cosine_scheduler",
        type=eval,
        default=True,
        help="Use cosine annealing scheduler",
    )

    # Model architecture
    parser.add_argument(
        "--hidden_dim", type=eval, default=768, help="Hidden dimension(s)"
    )
    parser.add_argument("--layers", type=eval, default=2, help="Layers per scale")
    parser.add_argument(
        "--equivariance",
        type=str,
        default="SEn",
        help="Type of equivariance (SEn or Tn)",
    )

    # Platonic Transformer specific parameters
    parser.add_argument(
        "--num_heads", type=int, default=None, help="Number of attention heads."
    )
    parser.add_argument(
        "--head_dim", type=int, default=16, help="Implicitly defines number of heads."
    )
    parser.add_argument(
        "--freq_sigma",
        type=float,
        default=2.0,
        help="Sigma for RFF positional encoding.",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument(
        "--norm_first", type=eval, default=True, help="Use LayerNorm before attention."
    )
    parser.add_argument(
        "--learned_freqs", type=eval, default=True, help="Learn frequencies for RFF."
    )
    parser.add_argument(
        "--dense_mode", type=eval, default=True, help="Use dense mode for attention."
    )
    parser.add_argument(
        "--mean_aggregation",
        type=eval,
        default=False,
        help="Use mean aggregation in attention.",
    )
    parser.add_argument(
        "--attention", type=eval, default=False, help="Use attention in the model."
    )
    parser.add_argument(
        "--ffn_readout", type=eval, default=True, help="Use FFN readout after pooling."
    )

    # Input features
    parser.add_argument(
        "--scalar_features",
        type=eval,
        default=[],
        help='Features to use as scalars (e.g., ["coords"])',
    )
    parser.add_argument(
        "--vector_features",
        type=eval,
        default=["coords"],
        help='Features to use as vectors (e.g., ["coords"])',
    )

    # Data and logging
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets/shapenet_car/preprocessed",
        help="Data directory for ShapenetCar",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=16,
        help="k for k-NN graph construction in the dataset loader",
    )
    parser.add_argument(
        "--log", type=eval, default=True, help="Enable logging with wandb"
    )

    # System and checkpointing
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of data loading workers"
    )
    parser.add_argument(
        "--enable_progress_bar", type=eval, default=True, help="Show progress bar"
    )
    parser.add_argument(
        "--test_ckpt", type=str, default=None, help="Checkpoint for testing"
    )
    parser.add_argument(
        "--resume_ckpt", type=str, default=None, help="Checkpoint to resume from"
    )

    args = parser.parse_args()
    main(args)
