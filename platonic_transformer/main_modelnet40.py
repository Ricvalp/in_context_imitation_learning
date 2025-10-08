import os
import argparse

import torch
import torchmetrics
import pytorch_lightning as pl
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader

from models.platoformer.platoformer import PlatonicTransformer
from models.platoformer.groups import PLATONIC_GROUPS

from utils import (CosineWarmupScheduler, NormalizeCoord, RandomJitter,
                   RandomRotatePerturbation, RandomShift, RandomSOd,
                   SamplePoints, TimerCallback)

# Performance optimization
torch.set_float32_matmul_precision('medium')

# Some augmentation functions
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, RandomScale


class ModelNet40Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)

        self.avg_num_nodes = args.num_points
        
        # Calculate total input channels
        in_channels_scalar = (
            3 * ("coords" in self.hparams.scalar_features) +  # x,y,z coordinates as scalars
            3 * ("normals" in self.hparams.scalar_features)    # normal components as scalars
        )
        in_channels_vector = (
            1 * ("coords" in self.hparams.vector_features) +  # position as vector
            1 * ("normals" in self.hparams.vector_features) +  # normal as vector
            3 * ("pose" in self.hparams.vector_features)       # pose matrix (3 vectors)
        )
        in_channels = in_channels_scalar + in_channels_vector

        # Ensure at least one input channel if none are specified
        if in_channels == 0:
            in_channels_scalar = 1  # will use constant ones as input
            in_channels = 1

        # Initialize model
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
            num_heads = self.hparams.hidden_dim // (self.hparams.head_dim * PLATONIC_GROUPS[solid_name.lower()].G)
            if (self.hparams.num_heads is not None) and (num_heads != self.hparams.num_heads):
                raise ValueError(f"head_dim {self.hparams.head_dim} does not match num_heads {self.hparams.num_heads} ")
            self.hparams.num_heads = num_heads

        self.net = PlatonicTransformer(
            input_dim=in_channels_scalar,
            input_dim_vec=in_channels_vector,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=40,
            output_dim_vec=0,
            nhead=self.hparams.num_heads,
            num_layers=self.hparams.layers,
            solid_name=solid_name,
            ffn_dim_factor=4,
            task_level="graph",
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
        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=40)
        self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=40)
        self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=40)

    def forward(self, data):
        # Apply rotation augmentation if enabled (during training)
        if (self.training and self.hparams.train_augm) or (not self.training and self.hparams.test_augm):
            rot = self.rotation_generator().type_as(data.pos)
            data.pos = torch.einsum('ij,bj->bi', rot, data.pos)
            if hasattr(data, 'normal'):
                data.normal = torch.einsum('ij,bj->bi', rot, data.normal)
        else:
            rot = torch.eye(3, device=data.pos.device)

        # Prepare input features
        x = []  # scalar features
        vec = []  # vector features

        # Add scalar features
        if "coords" in self.hparams.scalar_features:
            x.append(data.pos)
        if "normals" in self.hparams.scalar_features and hasattr(data, 'normal'):
            x.append(data.normal)

        # Add vector features
        if "coords" in self.hparams.vector_features:
            vec.append(data.pos[:,None,:])
        if "normals" in self.hparams.vector_features and hasattr(data, 'normal'):
            vec.append(data.normal[:,None,:])
        if "pose" in self.hparams.vector_features:
            vec.append(rot.transpose(-2,-1).unsqueeze(0).expand(data.pos.shape[0], -1, -1))

        # Combine features
        if not x and not vec:  # Only add constant ones if both x and vec are empty
            x = torch.ones(data.pos.size(0), 1).type_as(data.pos)
        else:
            x = torch.cat(x, dim=-1) if x else None
        vec = torch.cat(vec, dim=1) if vec else None

        # Forward pass
        pred, _ = self.net(x, data.pos, data.batch, vec=vec, avg_num_nodes=self.avg_num_nodes)
        return pred

    def training_step(self, data, batch_idx):
        pred = self(data)
        loss = torch.nn.functional.cross_entropy(pred, data.y)
        self.train_metric(pred, data.y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, data, batch_idx):
        pred = self(data)
        self.valid_metric(pred, data.y)

    def test_step(self, data, batch_idx):
        pred = self(data)
        self.test_metric(pred, data.y)

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_metric, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("valid_acc", self.valid_metric, prog_bar=True)

    def on_test_epoch_end(self):
        suffix = "_rotated" if self.hparams.test_augm else ""
        self.log(f"test_acc{suffix}", self.test_metric, prog_bar=True)
        # self.log("test_acc", self.test_metric, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, self.hparams.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def load_data(args):
    """Load and preprocess ModelNet40 dataset using PyG."""
    
    # Define transforms
    train_transform = T.Compose([
        NormalizeCoord(),
        SamplePoints(num=args.num_points, remove_faces=True, include_normals=True),
        RandomRotatePerturbation(angle_sigma=0.06, angle_clip=0.18),
        RandomScale((0.8, 1.25)),
        RandomShift(shift_range=0.1),
        RandomJitter(sigma=0.01, clip=0.05),
    ])

    test_transform = T.Compose([
        NormalizeCoord(),
        SamplePoints(num=args.num_points, remove_faces=True, include_normals=True),
    ])
    
    # Create datasets
    train_dataset = ModelNet(
        args.data_dir,
        name='40',
        train=True,
        transform=train_transform,
    )
    
    test_dataset = ModelNet(
        args.data_dir,
        name='40',
        train=False,
        transform=test_transform,
    )
   
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    return train_loader, test_loader

def main(args):
    # Set random seed
    pl.seed_everything(args.seed)

    # Load data
    train_loader, test_loader = load_data(args)

    # Setup hardware configuration
    if args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = "cpu"
        devices = "auto"
        
    # Configure logging
    if args.log:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(
            project="PlatonicTransformer-ModelNet40",
            config=args,
            save_dir=save_dir
        )
    else:
        logger = None

    # Setup callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor='valid_acc',
            mode='max',
            save_last=True
        ),
        TimerCallback()
    ]
    if args.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))

    # Initialize trainer
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=args.enable_progress_bar
    )

    # Train or test
    if args.test_ckpt is None:
        model = ModelNet40Model(args)
        trainer.fit(model, train_loader, test_loader)
        # Test without augmentation
        trainer.test(model, test_loader, ckpt_path = callbacks[0].last_model_path)
        # Test with augmentation
        model.hparams.test_augm = True
        trainer.test(model, test_loader, ckpt_path = callbacks[0].last_model_path)
    else:
        model = ModelNet40Model.load_from_checkpoint(args.test_ckpt)
        # Test without augmentation
        trainer.test(model, test_loader)
        # Test with augmentation
        model.hparams.test_augm = True
        trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ModelNet40 Classification Training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-12, help='Weight decay')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Model architecture
    parser.add_argument('--hidden_dim', type=eval, default=768, help='Hidden dimension(s)')
    # parser.add_argument('--hidden_dim', type=eval, default=[128,128,128,128,128], help='Hidden dimension(s)')
    parser.add_argument('--layers', type=eval, default=7, help='Layers per scale')
    # parser.add_argument('--layers', type=eval, default=[1,1,1,1,1], help='Layers per scale')
    parser.add_argument('--equivariance', type=str, default="Tn", help='Type of equivariance')
    
    # Platonic Transformer specific parameters
    parser.add_argument('--num_heads', type=int, default=None, help='Number of attention heads (Transformer only).')
    parser.add_argument('--head_dim', type=int, default=16, help='Implicitly defines number of heads (Transformer only).')
    parser.add_argument('--freq_sigma', type=float, default=1.0, help='Sigma for RFF positional encoding (Transformer only).')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (Transformer only).')
    parser.add_argument('--norm_first', type=eval, default=True, help='Use LayerNorm before attention in Transformer.')
    parser.add_argument('--use_rope', type=eval, default=True, help='Use Rotary Position Embedding (RoPE) in Transformer.')
    parser.add_argument('--learned_freqs', type=eval, default=True, help='Use Rotary Position Embedding (RoPE) in Transformer.')
    parser.add_argument('--dense_mode', type=eval, default=True, help='Enable dense attention blocks.')
    parser.add_argument('--mean_aggregation', type=eval, default=False, help='Use mean aggregation instead of sum.')
    parser.add_argument('--attention', type=eval, default=False, help='Use attention in the model.')
    parser.add_argument('--ffn_readout', type=eval, default=True, help='Use FFN readout after pooling.')
    
    # Input features
    parser.add_argument('--scalar_features', type=eval, default=["normals"], help='Features to use as scalars: ["coords", "normals"]')
    # parser.add_argument('--vector_features', type=eval, default=["normal","pose"], help='Features to use as vectors: ["coords", "normals", "pose"]')
    parser.add_argument('--vector_features', type=eval, default=[], help='Features to use as vectors: ["coords", "normals", "pose"]')
    
    # Training features
    parser.add_argument('--train_augm', type=eval, default=True, help='Use rotation augmentation during training')
    parser.add_argument('--test_augm', type=eval, default=False, help='Use rotation augmentation during testing')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points to sample')

    # Sweep configuration
    parser.add_argument('--config', type=eval, default=None, help='Sweep configuration dictionary')
    parser.add_argument('--model_id', type=int, default=None, help='Model ID in case you would want to label the configuration')
    
    # Data and logging
    parser.add_argument('--data_dir', type=str, default="./datasets/modelnet", help='Data directory')
    parser.add_argument('--log', type=eval, default=True, help='Enable logging')
    
    # System and checkpointing
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--enable_progress_bar', type=eval, default=True, help='Show progress bar')
    parser.add_argument('--test_ckpt', type=str, default=None, help='Checkpoint for testing')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Checkpoint to resume from')
    
    args = parser.parse_args()

    # Overwrite default settings with values from config if provided
    if args.config is not None:
        for key, value in args.config.items():
            setattr(args, key, value)
    
    main(args)
