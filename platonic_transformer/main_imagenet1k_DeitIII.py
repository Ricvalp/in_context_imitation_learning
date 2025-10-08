import os
import argparse

import torch
import torchvision
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.data import Data, Batch
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
# DEIT-III CHANGE: Import the LAMB optimizer from timm. You may need to run: pip install timm
try:
    from timm.optim import Lamb
except ImportError:
    print("timm is not installed. LAMB optimizer will not be available. Run 'pip install timm'")
    Lamb = None

from models.platoformer.platoformer import PlatonicTransformer
from models.baseline.vit import VisionTransformer
from models.platoformer.groups import PLATONIC_GROUPS
from utils import CosineWarmupScheduler, RandomSOd, TimerCallback
torch.set_float32_matmul_precision('medium')


class ImageNetModel(pl.LightningModule): 
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.rotation_generator = RandomSOd(2)
        in_channels_scalar = args.patch_size * args.patch_size * 3
        in_channels_vector = 0
        num_patches = (224 // args.patch_size) ** 2
        
        self.avg_num_nodes = (224 // args.patch_size)**2  # ImageNet images are 224x224
        

        if self.hparams.model_type == "platoformer":
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
                output_dim=1000,  # 1000 classes for ImageNet
                output_dim_vec=0,
                nhead=self.hparams.num_heads,
                num_layers=self.hparams.layers,
                solid_name=solid_name,
                spatial_dim=2,  # ImageNet is treated as a 2D point cloud
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
            )
        elif self.hparams.model_type == "vit":
            self.net = VisionTransformer(
                in_channels=in_channels_scalar,
                num_patches=num_patches,
                num_classes=1000,
                hidden_dim=self.hparams.hidden_dim,
                num_layers=self.hparams.layers,
                num_heads=self.hparams.num_heads,
                mlp_dim=self.hparams.mlp_dim,
                dropout=self.hparams.dropout,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.hparams.model_type}")

        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=1000)
        self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=1000)
        self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=1000)

    def forward(self, data):
        # Apply rotation augmentation during training if enabled
        if self.training and self.hparams.train_augm:
            rot = self.rotation_generator().type_as(data.pos)
            data.pos = torch.einsum('ij,bj->bi', rot, data.pos)

        # Forward pass through the selected network
        if self.hparams.model_type == "platoformer":
            pred, _ = self.net(data.x, data.pos, data.batch, vec=None, avg_num_nodes=self.avg_num_nodes)
        elif self.hparams.model_type == "vit":
            pred = self.net(data[0])
        else:
            raise ValueError(f"Unsupported model_type: {self.hparams.model_type}")
        return pred

    def _calculate_loss(self, pred, y):
        """Helper function to calculate loss based on configuration."""
        # DEIT-III CHANGE: Use Binary Cross-Entropy loss if specified
        if self.hparams.loss_fn == "bce":
            y_one_hot = torch.nn.functional.one_hot(y, num_classes=1000).float()
            return torch.nn.functional.binary_cross_entropy_with_logits(pred, y_one_hot)
        else:
            return torch.nn.functional.cross_entropy(pred, y)

    def training_step(self, data, batch_idx):
        pred = self(data)
        loss = self._calculate_loss(pred, data.y)
        self.train_metric(pred, data.y)
        self.log("train_loss", loss, batch_size=data.num_graphs, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, data, batch_idx):
        pred = self(data)
        loss = self._calculate_loss(pred, data.y)
        self.valid_metric(pred, data.y)
        self.log("valid_loss", loss, batch_size=data.num_graphs, prog_bar=True, sync_dist=True)

    def test_step(self, data, batch_idx):
        pred = self(data)
        self.test_metric(pred, data.y)

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_metric, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log("valid_acc", self.valid_metric, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_metric, prog_bar=True, sync_dist=True)

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
    """Load and preprocess ImageNet dataset by patching images into point clouds."""
    
    if 224 % args.patch_size != 0:  # ImageNet images are 224x224
        raise ValueError("Image dimension (224) must be divisible by patch_size.")

    # DEIT-III CHANGE: Implement the "3-Augment" strategy + Color Jitter for training
    transform_train = torchvision.transforms.Compose([
        # torchvision.transforms.RandomCrop(224, padding=4),
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
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
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(256),
    #     torchvision.transforms.CenterCrop(224),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    
 
    num_patches_1d = 224 // args.patch_size  
    grid = torch.linspace(0.0, 1.0, num_patches_1d)
    grid_x, grid_y = torch.meshgrid(grid, grid, indexing='xy')
    patch_pos = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    def collate_fn(batch):
        """
        Custom collate function to convert a batch of images into a batch of patch-based point clouds.
        """
        data_list = []
        p = args.patch_size
        
        for image_tensor, label in batch:
            # image_tensor shape: [C, H, W] = [3, 224, 224]
            # 1. Unfold image into patches: [C, H, W] -> [C, num_patches_h, num_patches_w, p, p]
            patches = image_tensor.unfold(1, p, p).unfold(2, p, p)
            # 2. Permute and flatten: -> [num_patches_total, C*p*p]
            patches = patches.permute(1, 2, 0, 3, 4).contiguous()
            # x shape will be [num_patches, patch_size*patch_size*C]
            x = patches.view(-1, 3 * p * p)
            data = Data(x=x, pos=patch_pos.clone(), y=torch.tensor([label]))
            data_list.append(data)
        
        return Batch.from_data_list(data_list)

    # --- Create datasets and DataLoaders ---
    # full_train_dataset = torchvision.datasets.ImageNet(root=args.data_dir, split='train', transform=transform_train)
    # test_dataset = torchvision.datasets.ImageNet(root=args.data_dir, split='val', transform=transform)
    from datasets.imagenet.dataset import ImageNet
    full_train_dataset = ImageNet(data_dir="/scratch-nvme/ml-datasets/imagenet/ILSVRC/Data/CLS-LOC/", split='train', transform=transform_train)
    test_dataset = ImageNet(data_dir="/scratch-nvme/ml-datasets/imagenet/ILSVRC/Data/CLS-LOC/", split='val', transform=transform)

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    
    # Subsample the training dataset on percentage for data scaling experiments
    if args.train_percentage < 100:
        subset_size = int((args.train_percentage / 100) * len(train_dataset))
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset, [subset_size, len(train_dataset) - subset_size], generator
            =torch.Generator().manual_seed(args.seed))
    print(f"Training on {len(train_dataset)} samples ({100*len(train_dataset)/len(full_train_dataset):.2f}% of full training set)")
   
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
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
        logger = pl.loggers.WandbLogger(project="Platonic-ImageNet", config=args, save_dir=save_dir)
    else:
        logger = None

    callbacks = [pl.callbacks.ModelCheckpoint(monitor='valid_acc', mode='max', save_last=True), TimerCallback()]
    if args.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))

    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, callbacks=callbacks, gradient_clip_val=0.5,
                         accelerator=accelerator, devices=devices, enable_progress_bar=args.enable_progress_bar,
                         strategy=DDPStrategy(find_unused_parameters=True))

    if args.test_ckpt is None:
        model = ImageNetModel(args) 
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader, ckpt_path='best')
    else:
        model = ImageNetModel.load_from_checkpoint(args.test_ckpt) 
        trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ImageNet Point Cloud Classification Training')  # Updated description
    
    # --- General Training Parameters (DEIT-III CHANGE: Updated defaults) ---
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--warmup', type=int, default=20, help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=8e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--train_percentage', type=float, default=100.0, help='Percentage of training data to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # DEIT-III CHANGE: Add configurable optimizer and loss function
    parser.add_argument('--optimizer', type=str, default='lamb', choices=['adamw', 'lamb'], help='Optimizer to use')
    parser.add_argument('--loss_fn', type=str, default='bce', choices=['cross_entropy', 'bce'], help='Loss function to use')

    # --- Model Architecture ---
    parser.add_argument('--model_type', type=str, default='platoformer', choices=['platoformer', 'vit'], help='Model to use')
    parser.add_argument('--hidden_dim', type=eval, default=768, help='Hidden dimension(s)')
    parser.add_argument('--layers', type=eval, default=5, help='Number of layers or layers per scale')
    parser.add_argument('--solid_name', type=str, default="trivial_2", help='Type of symmetry group ("trivial_2", "cyclic_#", "dihedral_#", "flop") (Tn, SEn)')
    parser.add_argument('--num_ori', type=int, default=8, help='Number of orientations')

    # --- Platonic Transformer Specific Parameters ---
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--head_dim', type=int, default=None, help='Implicitly defines number of heads')
    parser.add_argument('--rope_sigma', type=eval, default=1.0, help='Sigma for RFF positional encoding')
    parser.add_argument('--ape_sigma', type=eval, default=10.0, help='Sigma for RFF positional encoding')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--norm_first', type=eval, default=True, help='Use LayerNorm before attention')
    parser.add_argument('--learned_freqs', type=eval, default=True, help='Learnable frequencies for RFF')
    parser.add_argument('--dense_mode', type=eval, default=True, help='Use dense attention')
    parser.add_argument('--mean_aggregation', type=eval, default=False, help='Use mean aggregation instead of sum')
    parser.add_argument('--attention', type=eval, default=True, help='Use attention in the model')
    parser.add_argument('--ffn_readout', type=eval, default=True, help='Use FFN readout after pooling')
    # --- NEW ARGUMENT FOR PLATOFORMER LAYER SCALE ---
    parser.add_argument('--layer_scale_init_value', type=float, default=None, help='Initial value for LayerScale in Platonic Transformer (default: disabled)')

    # --- NEW, RENAMED GENERIC ARGUMENT FOR DROP PATH / STOCHASTIC DEPTH ---
    parser.add_argument('--drop_path_rate', type=float, default=0.0, help='Stochastic depth rate (uniform) for both models')

    # --- Data and Augmentation ---
    parser.add_argument('--train_augm', type=eval, default=True, help='Use rotation augmentation during training')
    parser.add_argument('--patch_size', type=int, default=16, help='Side length of the square image patches')
    parser.add_argument('--data_dir', type=str, default="/projects/0/prjs1161/imagenet", help='Data directory')

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
