from functools import partial
import os
import argparse
import sys

import torch
import torchmetrics
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader


from datasets.protein_md import MDAnalysisDataset
from utils import CosineWarmupScheduler, RandomSOd, TimerCallback
from pytorch_lightning.callbacks import Timer

from models.platoformer.platoformer import PlatonicTransformer
from models.platoformer.groups import PLATONIC_GROUPS



# Performance optimizations
torch.set_float32_matmul_precision('medium')


class MDModel(pl.LightningModule):
    """Lightning module for QM9 molecular property prediction."""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)
        
        # Calculate total input channels
        in_channels_scalar = (
            2 + # base atom features
            3 * ("coords" in self.hparams.scalar_features) +  # x,y,z coordinates as scalars
            3 * ("velocity" in self.hparams.scalar_features)  # x,y,z velocity as scalars
        )
        in_channels_vector = (
            1 * ("coords" in self.hparams.vector_features) +   # position as vector
            1 * ("velocity" in self.hparams.vector_features)      # velocity as vector
        )
        
        # Calculate output dimensions
        self.lifted = self.hparams.orientations != 0
        out_channels_scalar = 3 if not self.lifted else 0
        out_channels_vec = 0 if not self.lifted else 1

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
            num_heads = self.hparams.hidden_dim // (self.hparams.head_dim * PLATONIC_GROUPS[solid_name.lower()].G)
            if (self.hparams.num_heads is not None) and (num_heads != self.hparams.num_heads):
                raise ValueError(f"head_dim {self.hparams.head_dim} does not match num_heads {self.hparams.num_heads} ")
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
    
    def set_dataset_statistics(self, train_loader):
        self.avg_num_nodes = sum(data.num_nodes for data in train_loader) / len(train_loader.dataset)

    def forward(self, graph):
        # Prepare input features
        x = []
        vec = []
        
        # Add base atomic features
        x.append(graph.x)
        
        # Add scalar features
        if "coords" in self.hparams.scalar_features:
            x.append(graph.pos)
        if "velocity" in self.hparams.scalar_features:
            x.append(graph.vel_0)
        
        # Add vector features
        if "coords" in self.hparams.vector_features:
            vec.append(graph.pos[:,None,:])
        if "velocity" in self.hparams.vector_features:
            vec.append(graph.vel_0[:,None,:])

        # Combine features
        x = torch.cat(x, dim=-1) if x else None
        vec = torch.cat(vec, dim=1) if vec else None
        x = torch.ones(graph.pos.size(0), 1).type_as(graph.pos) if (x is None and vec is None) else x

        # Forward pass
        pred_scalar, pred_vec = self.net(x, graph.pos, graph.batch, vec=vec, avg_num_nodes=self.avg_num_nodes)

        # Process output
        if not self.lifted:
            delta = pred_scalar
        else:
            delta = pred_vec[:, 0, :]

        return graph.pos + delta
    

    def training_step(self, graph, batch_idx):
        # Apply rotation augmentation if enabled
        if self.hparams.train_augm:
            batch_size = graph.batch.max().item() + 1
            rots = self.rotation_generator(n=batch_size).type_as(graph.pos)
            rot_per_sample = rots[graph.batch]
            graph.pos = torch.einsum('bij,bj->bi', rot_per_sample, graph.pos)
            graph.vel_0 = torch.einsum('bij,bj->bi', rot_per_sample, graph.vel_0)
            graph.y = torch.einsum('bij,bj->bi', rot_per_sample, graph.y)
        
        # Forward pass and loss computation
        y_pred = self(graph)
        loss = torch.nn.functional.mse_loss(y_pred, graph.y)
        self.train_metric(y_pred.contiguous(), graph.y.contiguous())
        return loss

    def validation_step(self, graph, batch_idx):
        y_pred = self(graph)
        self.valid_metric(y_pred.contiguous(), graph.y.contiguous())
        
    def test_step(self, graph, batch_idx):
        y_pred = self(graph)
        self.test_metric(y_pred.contiguous(), graph.y.contiguous())

    def on_train_epoch_end(self):
        self.log("train MAE", self.train_metric, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("valid MAE", self.valid_metric, prog_bar=True)

    def on_test_epoch_end(self):
        self.log("test MAE", self.test_metric, prog_bar=True)
    
    def configure_optimizers(self):
        """Configure optimizer with weight decay and learning rate schedule."""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():  # mn is module name, m is module instance
            for pn, p in m.named_parameters():  # pn is parameter name (e.g., 'weight', 'bias', 'freqs')
                fpn = f'{mn}.{pn}' if mn else pn  # fpn is the full parameter name

                if pn == 'freqs':
                    no_decay.add(fpn)
                elif pn.endswith('bias') or ('layer_scale' in pn):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('kernel'):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                # Parameters not matching any rule will be caught later and added to no_decay by default.

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        current_params_in_groups = decay | no_decay
        missing_params = param_dict.keys() - current_params_in_groups
        if missing_params:
            print(f"Warning: Parameters {missing_params} were not explicitly assigned to decay/no_decay by specific rules. Adding to no_decay by default.")
            no_decay.update(missing_params) # Add missing parameters to no_decay group

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} found in both decay and no_decay sets!"
        
        # Ensure all learnable parameters are covered
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} not assigned to any optimizer group!"

        optim_groups = [
            {"params": [param_dict[p_name] for p_name in sorted(list(decay)) if p_name in param_dict], "weight_decay": self.hparams.weight_decay},
            {"params": [param_dict[p_name] for p_name in sorted(list(no_decay)) if p_name in param_dict], "weight_decay": 0.0},
        ]
        
        # Filter out empty groups (e.g., if 'decay' set is empty)
        optim_groups = [group for group in optim_groups if group["params"]]

        if not optim_groups and list(param_dict.keys()): # Should not happen if there are learnable params
            raise ValueError("No optimizer groups were created, but there are learnable parameters.")
        elif not optim_groups and not list(param_dict.keys()): # No learnable params
             print("Warning: No learnable parameters found for the optimizer.")

        optimizer = torch.optim.Adam(optim_groups, lr=self.hparams.lr)
        if self.hparams.cosine_scheduler:
            scheduler = CosineWarmupScheduler(optimizer, self.hparams.warmup, self.trainer.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid MAE"}
        else:
            return {"optimizer": optimizer, "monitor": "valid MAE"}

def load_data(args):
    """Load and preprocess QM9 dataset."""
    # Load dataset
    dataset = partial(MDAnalysisDataset, dataset_name='protein_md', 
                      data_dir=args.data_dir, cutoff_rate=0.25,
                      virtual_channels=8, load_cached=True, 
                      delta_frame=15, backbone=args.backbone_atoms_only, 
                      test_rot=args.test_rot_trans, 
                      test_trans=args.test_rot_trans)
    
    dataset_train = dataset(max_samples=args.max_samples, partition='train')
    dataset_valid = dataset(max_samples=args.max_samples,  partition='valid')
    dataset_test  = dataset(max_samples=args.max_samples,  partition='test')
    
    # Create train/val/test split (same as FastEGNN)
    datasets = {'train': dataset_train, 'val': dataset_valid, 'test': dataset_test}
      # Create dataloaders
    dataloaders = {
        split: DataLoader(dataset.get_data(), batch_size=args.batch_size, 
                         shuffle=(split == 'train'), 
                         num_workers=args.num_workers)
        for split, dataset in datasets.items()
    }
    
    return dataloaders['train'], dataloaders['val'], dataloaders['test']


def main(args):
    
    # Set random seed
    pl.seed_everything(args.seed)

    # Load data
    train_loader, val_loader, test_loader = load_data(args)

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
            project="PlatonicTransformer-MDProtein",
            name=None,  # Let wandb auto-generate the run name
            config=args, 
            save_dir=save_dir,
            entity='m-m-islam-university-of-amsterdam'  # The team entity for shared logging
        )
    else:
        logger = None

    # Setup callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor='valid MAE', mode='min', 
                                   every_n_epochs=1, save_last=True),
        TimerCallback()
    ]
    if args.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    if args.timer is not None:
        callbacks.append(Timer(
            duration=args.timer
        ))

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
        model = MDModel(args)
        model.set_dataset_statistics(train_loader)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)
        trainer.test(model, test_loader, ckpt_path=callbacks[0].best_model_path)
    else:
        model = MDModel.load_from_checkpoint(args.test_ckpt)
        model.set_dataset_statistics(train_loader)
        trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QM9 Property Prediction Training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--timer', type=str, default=None, help='Timer for training in string format, e.g., \"00:08:00:00\" for 8 hours')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='Weight decay')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cosine_scheduler', type=eval, default=True, help='Use cosine annealing scheduler')
    parser.add_argument('--max_samples', type=int, default=1e8, help='Maximum number of samples to use from the dataset')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=eval, default=384, help='Hidden dimension(s)')
    parser.add_argument('--layers', type=eval, default=4, help='Layers per scale')
    parser.add_argument('--equivariance', type=str, default="SEn", help='Type of equivariance')
    parser.add_argument('--orientations', type=int, default=8, help='Number of orientations controlling vector outputs')

    # Platonic Transformer specific parameters
    parser.add_argument('--num_heads', type=int, default=None, help='Number of attention heads (Transformer only).')
    parser.add_argument('--head_dim', type=int, default=16, help='Implicitly defines number of heads (Transformer only).')
    parser.add_argument('--freq_sigma', type=float, default=0.1, help='Sigma for RFF positional encoding (Transformer only).')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (Transformer only).')
    parser.add_argument('--norm_first', type=eval, default=True, help='Use LayerNorm before attention in Transformer.')
    parser.add_argument('--use_rope', type=eval, default=True, help='Use Rotary Position Embedding (RoPE) in Transformer.')
    parser.add_argument('--learned_freqs', type=eval, default=True, help='Use Rotary Position Embedding (RoPE) in Transformer.')
    parser.add_argument('--dense_mode', type=eval, default=True, help='Enable dense attention blocks.')
    parser.add_argument('--mean_aggregation', type=eval, default=False, help='Use mean aggregation instead of sum.')
    parser.add_argument('--attention', type=eval, default=False, help='Use attention in the model.')
    parser.add_argument('--ffn_readout', type=eval, default=True, help='Use FFN readout after pooling.')

    # Training features
    parser.add_argument('--train_augm', type=eval, default=True, help='Use rotation augmentation')
    
    # Input features
    parser.add_argument('--scalar_features', type=eval, default=[], help='Features to use as scalars: ["coords"]')
    parser.add_argument('--vector_features', type=eval, default=["velocity"], help='Features to use as vectors: ["coords"]')
    parser.add_argument('--test_rot_trans', type=bool, default=False, help='Rotate and translate the test set for evaluation')
   
    # Data and logging
    parser.add_argument('--data_dir', type=str, default='./datasets/protein_md/', help='Data directory')
    parser.add_argument('--log', type=eval, default=True, help='Enable logging')
    parser.add_argument('--backbone_atoms_only', type=bool, default=False, help='Use oonly backbone or all atoms of the protein')

    # Sweep configuration
    parser.add_argument('--config', type=eval, default=None, help='Sweep configuration dictionary')
    parser.add_argument('--model_id', type=int, default=None, help='Model ID in case you would want to label the configuration')
    
    # System and checkpointing
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--enable_progress_bar', type=eval, default=True, help='Show progress bar')
    parser.add_argument('--test_ckpt', type=str, default=None, help='Checkpoint for testing')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Checkpoint to resume from')
    
    args = parser.parse_args()

    # Overwrite default settings with values from config if provided
    if args.config is not None:
        for key, value in args.config.items():
            setattr(args, key, value)

    main(args)
