import os
import argparse
import random

import torch
import torchmetrics
import pytorch_lightning as pl
import torchvision
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
# DEIT-III CHANGE: Import the LAMB optimizer from timm. You may need to run: pip install timm
try:
    from timm.optim import Lamb
except ImportError:
    print("timm is not installed. LAMB optimizer will not be available. Run 'pip install timm'")
    Lamb = None


# Import model and necessary utilities
from models.platoformer.platoformer import PlatonicTransformer
from models.platoformer.groups import PLATONIC_GROUPS
from utils import CosineWarmupScheduler, RandomSOd, TimerCallback

# In order to be able to download cifar on the server
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Performance optimization
torch.set_float32_matmul_precision('medium')


class CIFAR10Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        # Setup 2D rotation augmentation for the point cloud
        self.rotation_generator = RandomSOd(2)

        # CIFAR10 point cloud has 3 scalar features (RGB) and 0 vector features
        in_channels_scalar = args.patch_size * args.patch_size * 3
        in_channels_vector = 2
        
        # The number of "points" is now the number of patches (e.g., (32/8)^2 = 16)
        self.avg_num_nodes = (32 // args.patch_size)**2
        
        # --- Model Initialization ---

        solid_name = self.hparams.solid_name
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(f"Invalid solid_name '{solid_name}'. Must be one of: {list(PLATONIC_GROUPS.keys())}")

        if self.hparams.head_dim is not None:
            group = PLATONIC_GROUPS[solid_name.lower()]
            num_heads = self.hparams.hidden_dim // (self.hparams.head_dim * group.G)
            if (self.hparams.num_heads is not None) and (num_heads != self.hparams.num_heads):
                raise ValueError(f"head_dim {self.hparams.head_dim} does not match num_heads {self.hparams.num_heads} ")
            self.hparams.num_heads = num_heads

        self.net = PlatonicTransformer(
            # Basic/essential specification:
            input_dim=in_channels_scalar,
            input_dim_vec=in_channels_vector,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=10,  # 10 classes for CIFAR10
            output_dim_vec=0,
            nhead=self.hparams.num_heads,
            num_layers=self.hparams.layers,
            solid_name=solid_name,
            spatial_dim=2,  # CIFAR10 is treated as a 2D point cloud
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
            conditioning_dim=None,
            conditioning_mlp_dim=None,
        )

        # Setup metrics
        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, data):
        # Apply rotation augmentation during training if enabled
        if self.training and self.hparams.train_augm:
            rot = self.rotation_generator().type_as(data.pos)
            data.pos = torch.einsum('ij,bj->bi', rot, data.pos)
        else:
            rot = torch.eye(2, device=data.pos.device)

        vec = rot.transpose(-2,-1).unsqueeze(0).expand(data.pos.shape[0], -1, -1)

        # Forward pass through the network (batch_mask and edge_index can be None)
        pred, _ = self.net(data.x, data.pos, data.batch, vec=vec, avg_num_nodes=self.avg_num_nodes)
        return pred

    def _calculate_loss(self, pred, y):
        """Helper function to calculate loss based on configuration."""
        # DEIT-III CHANGE: Use Binary Cross-Entropy loss if specified
        if self.hparams.loss_fn == "bce":
            y_one_hot = torch.nn.functional.one_hot(y, num_classes=10).float()
            return torch.nn.functional.binary_cross_entropy_with_logits(pred, y_one_hot)
        else:
            return torch.nn.functional.cross_entropy(pred, y)

    def training_step(self, data, batch_idx):
        pred = self(data)
        loss = self._calculate_loss(pred, data.y)
        self.train_metric(pred, data.y)
        self.log("train_loss", loss, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, data, batch_idx):
        pred = self(data)
        loss = self._calculate_loss(pred, data.y)
        self.valid_metric(pred, data.y)
        self.log("valid_loss", loss, batch_size=self.hparams.batch_size)

    def test_step(self, data, batch_idx):
        pred = self(data)
        self.test_metric(pred, data.y)

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_metric, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("valid_acc", self.valid_metric, prog_bar=True)

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_metric, prog_bar=True)

    def configure_optimizers(self):
        # DEIT-III CHANGE: Use LAMB optimizer if specified
        if self.hparams.optimizer == "lamb":
            if Lamb is None:
                raise ImportError("timm is not installed. Cannot use LAMB optimizer.")
            optimizer = Lamb(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else: # Default to AdamW
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            
        scheduler = CosineWarmupScheduler(optimizer, self.hparams.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def load_data(args):
    """Load and preprocess CIFAR10 dataset by patching images into point clouds."""
    
    if 32 % args.patch_size != 0:
        raise ValueError("Image dimension (32) must be divisible by patch_size.")

    # DEIT-III CHANGE: Implement the "3-Augment" strategy + Color Jitter for training
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        # --- Start of DeiT-III Augmentations ---
        torchvision.transforms.RandomChoice([
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.RandomSolarize(threshold=128.0),
            torchvision.transforms.GaussianBlur(kernel_size=(3, 3))
        ]),
        torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        # --- End of DeiT-III Augmentations ---
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    num_patches_1d = 32 // args.patch_size
    grid = torch.linspace(0.0, 1.0, num_patches_1d)
    grid_x, grid_y = torch.meshgrid(grid, grid, indexing='xy')
    patch_pos = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    # Zero center:
    patch_pos = patch_pos - 0.5  # is only relevant for global RoPE

    def collate_fn(batch):
        data_list = []
        p = args.patch_size
        for image_tensor, label in batch:
            patches = image_tensor.unfold(1, p, p).unfold(2, p, p)
            patches = patches.permute(1, 2, 0, 3, 4).contiguous()
            x = patches.view(-1, 3 * p * p)
            data = Data(x=x, pos=patch_pos.clone(), y=torch.tensor([label]))
            data_list.append(data)
        return Batch.from_data_list(data_list)

    full_train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, transform=transform_train, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_test, download=True)
    
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
   
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def main(args):
    pl.seed_everything(args.seed)
    train_loader, val_loader, test_loader = load_data(args)

    if args.gpus > 0:
        accelerator, devices = "gpu", args.gpus
    else:
        accelerator, devices = "cpu", "auto"
        
    if args.log:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(project="Platonic-CIFAR10", config=args, save_dir=save_dir)
    else:
        logger = None

    callbacks = [pl.callbacks.ModelCheckpoint(monitor='valid_acc', mode='max', save_last=True), TimerCallback()]
    if args.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))

    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, callbacks=callbacks, gradient_clip_val=0.5,
                         accelerator=accelerator, devices=devices, enable_progress_bar=args.enable_progress_bar)

    if args.test_ckpt is None:
        model = CIFAR10Model(args)
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader, ckpt_path='best')
    else:
        model = CIFAR10Model.load_from_checkpoint(args.test_ckpt)
        trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR10 Point Cloud Classification Training')
    
    # --- General Training Parameters (DEIT-III CHANGE: Updated defaults) ---
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--warmup', type=int, default=20, help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=8e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--optimizer', type=str, default='lamb', choices=['adamw', 'lamb'], help='Optimizer to use')
    parser.add_argument('--loss_fn', type=str, default='bce', choices=['cross_entropy', 'bce'], help='Loss function to use')

    # --- Model Architecture ---
    parser.add_argument('--hidden_dim', type=eval, default=768, help='Hidden dimension(s)')
    parser.add_argument('--layers', type=eval, default=12, help='Number of layers or layers per scale')
    parser.add_argument('--solid_name', type=str, default="cyclic_4", help='Type of symmetry group ("trivial_2", "cyclic_#", "dihedral_#", "flop") (Tn, SEn)')
    parser.add_argument('--num_ori', type=int, default=8, help='Number of orientations')
    
    # --- Platonic Transformer Specific Parameters ---
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--head_dim', type=int, default=None, help='Implicitly defines number of heads')
    parser.add_argument('--rope_sigma', type=eval, default=16.0, help='Sigma for RFF positional encoding')
    parser.add_argument('--use_key', type=eval, default=False, help='Use key projection when using RoPE')
    parser.add_argument('--freq_init', type=str, default='spiral', choices=['random', 'spiral'], help='Frequency initialization method for RoPE')
    parser.add_argument('--ape_sigma', type=eval, default=16.0, help='Sigma for RFF positional encoding')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--norm_first', type=eval, default=True, help='Use LayerNorm before attention')
    parser.add_argument('--learned_freqs', type=eval, default=True, help='Learnable frequencies for RFF')
    parser.add_argument('--dense_mode', type=eval, default=True, help='Use dense attention')
    parser.add_argument('--mean_aggregation', type=eval, default=False, help='Use mean aggregation instead of sum')
    parser.add_argument('--attention', type=eval, default=True, help='Use attention in the model')
    parser.add_argument('--ffn_readout', type=eval, default=False, help='Use FFN readout after pooling')
    # --- NEW ARGUMENT FOR PLATOFORMER LAYER SCALE ---
    parser.add_argument('--layer_scale_init_value', type=float, default=None, help='Initial value for LayerScale in Platonic Transformer (default: disabled)')

    # --- NEW, RENAMED GENERIC ARGUMENT FOR DROP PATH / STOCHASTIC DEPTH ---
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='Stochastic depth rate (uniform) for both models')

    # --- Data and Augmentation ---
    parser.add_argument('--train_augm', type=eval, default=False, help='Use rotation augmentation during training')
    parser.add_argument('--patch_size', type=int, default=4, help='Side length of the square image patches')
    parser.add_argument('--data_dir', type=str, default="./datasets/cifar10", help='Data directory')

    # --- System and Checkpointing ---
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--log', type=eval, default=True, help='Enable logging')
    parser.add_argument('--enable_progress_bar', type=eval, default=True, help='Show progress bar')
    parser.add_argument('--test_ckpt', type=str, default=None, help='Checkpoint for testing')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--config', type=eval, default=None, help='Sweep configuration dictionary')
    
    args = parser.parse_args()

    if args.config is not None:
        for key, value in args.config.items():
            setattr(args, key, value)
    
    main(args)
