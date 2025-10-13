import argparse
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from models.platoformer.platoformer import PlatonicTransformer
from utils import CosineWarmupScheduler, TimerCallback


# Ensure CIFAR-10 can be downloaded without SSL issues
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_patch_grid(patch_size: int) -> torch.Tensor:
    if 32 % patch_size != 0:
        raise ValueError("Image dimension (32) must be divisible by patch_size.")
    num_patches_1d = 32 // patch_size
    grid = torch.linspace(0.0, 1.0, num_patches_1d)
    grid_x, grid_y = torch.meshgrid(grid, grid, indexing="xy")
    patch_pos = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    return patch_pos - 0.5


def timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps + s) / (1 + s)) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0.0001, 0.9999).float()


def patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    p = patch_size
    patches = image.unfold(1, p, p).unfold(2, p, p)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    return patches.view(-1, 3 * p * p)


def unpatchify(patches: torch.Tensor, patch_size: int) -> torch.Tensor:
    p = patch_size
    num_patches = patches.shape[0]
    num_patches_1d = int(math.sqrt(num_patches))
    patches = patches.view(num_patches_1d, num_patches_1d, 3, p, p)
    patches = patches.permute(2, 0, 3, 1, 4).contiguous()
    return patches.view(3, 32, 32)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[
                name
            ]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.data.clone()
            param.data = self.shadow[name].clone()

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data = self.backup[name]
        self.backup = {}


class CIFAR10DiffusionModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        h = self.hparams
        self.patch_dim = h.patch_size * h.patch_size * 3
        self.num_patches = (32 // h.patch_size) ** 2
        self.register_buffer(
            "patch_pos", build_patch_grid(h.patch_size), persistent=False
        )

        self.net = PlatonicTransformer(
            input_dim=self.patch_dim,
            input_dim_vec=0,
            hidden_dim=h.hidden_dim,
            output_dim=self.patch_dim,
            output_dim_vec=0,
            nhead=h.num_heads,
            num_layers=h.layers,
            solid_name=h.solid_name,
            spatial_dim=2,
            dense_mode=h.dense_mode,
            scalar_task_level="node",
            vector_task_level="node",
            ffn_readout=h.ffn_readout,
            mean_aggregation=h.mean_aggregation,
            dropout=h.dropout,
            norm_first=h.norm_first,
            drop_path_rate=h.drop_path_rate,
            layer_scale_init_value=h.layer_scale_init_value,
            attention=h.attention,
            ffn_dim_factor=h.ffn_dim_factor,
            rope_sigma=h.rope_sigma,
            ape_sigma=h.ape_sigma,
            learned_freqs=h.learned_freqs,
            freq_init=h.freq_init,
            use_key=h.use_key,
            conditioning_dim=h.conditioning_dim,
            conditioning_mlp_dim=h.conditioning_mlp_dim,
        )

        cond_dim = h.conditioning_dim
        self.time_embed = nn.Sequential(
            nn.Linear(cond_dim, h.time_embed_dim),
            nn.SiLU(),
            nn.Linear(h.time_embed_dim, cond_dim),
        )

        if h.class_cond:
            self.class_embed = nn.Embedding(10, cond_dim)
        else:
            self.class_embed = None

        betas = cosine_beta_schedule(h.num_steps)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=alphas.device), alphas_cumprod[:-1]]
        )
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0)
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_variance = posterior_variance.clamp(min=1e-20)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", torch.log(posterior_variance))
        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

        self.ema = EMA(self.net, decay=h.ema_decay) if h.use_ema else None

    def build_conditioning(
        self, timesteps: torch.Tensor, labels: Optional[torch.Tensor], drop_prob: float
    ) -> torch.Tensor:
        cond = timestep_embedding(timesteps, self.hparams.conditioning_dim)
        cond = self.time_embed(cond)

        if self.class_embed is not None and labels is not None:
            if labels.ndim > 1:
                labels = labels.view(-1)
            class_cond = self.class_embed(labels)
            if drop_prob > 0.0:
                drop_mask = (
                    (torch.rand_like(timesteps.float()) < drop_prob)
                    .float()
                    .unsqueeze(-1)
                )
                class_cond = class_cond * (1.0 - drop_mask)
            cond = cond + class_cond
        return cond

    def predict_noise(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch_idx: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        pred, _ = self.net(
            x, pos, batch_idx, conditioning=cond, avg_num_nodes=float(self.num_patches)
        )
        return pred

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        x0 = batch.x
        noise = torch.randn_like(x0)
        batch_size = batch.num_graphs
        timesteps = torch.randint(
            0, self.hparams.num_steps, (batch_size,), device=x0.device, dtype=torch.long
        )

        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps][batch.batch].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps][
            batch.batch
        ].unsqueeze(-1)
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise

        labels = batch.y if hasattr(batch, "y") else None
        cond = self.build_conditioning(timesteps, labels, self.hparams.cfg_drop_prob)
        pred_noise = self.predict_noise(x_t, batch.pos, batch.batch, cond)
        loss = F.mse_loss(pred_noise, noise)
        self.log("train_loss", loss, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        x0 = batch.x
        noise = torch.randn_like(x0)
        batch_size = batch.num_graphs
        timesteps = torch.randint(
            0, self.hparams.num_steps, (batch_size,), device=x0.device, dtype=torch.long
        )

        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps][batch.batch].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps][
            batch.batch
        ].unsqueeze(-1)
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise

        labels = batch.y if hasattr(batch, "y") else None
        cond = self.build_conditioning(timesteps, labels, 0.0)
        pred_noise = self.predict_noise(x_t, batch.pos, batch.batch, cond)
        loss = F.mse_loss(pred_noise, noise)
        self.log("val_loss", loss, prog_bar=True, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(self.hparams.beta1, 0.999),
        )
        scheduler = CosineWarmupScheduler(
            optimizer, self.hparams.warmup, self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_after_backward(self) -> None:
        if self.ema is not None:
            self.ema.update(self.net)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        guidance_scale: float = 0.0,
        class_labels: Optional[torch.Tensor] = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        device = self.device
        was_training = self.training
        self.eval()
        ema_applied = False
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.net)
            ema_applied = True

        if class_labels is None and self.class_embed is not None:
            class_labels = torch.randint(0, 10, (num_samples,), device=device)
        elif class_labels is not None:
            class_labels = class_labels.to(device)

        data_list = []
        for idx in range(num_samples):
            node_pos = self.patch_pos.detach().cpu().clone()
            if class_labels is not None:
                label_tensor = torch.tensor(
                    int(class_labels[idx].item()), dtype=torch.long
                )
                data_list.append(Data(pos=node_pos, y=label_tensor))
            else:
                data_list.append(Data(pos=node_pos))
        batch_template = Batch.from_data_list(data_list).to(device)
        x = torch.randn(num_samples * self.num_patches, self.patch_dim, device=device)

        for t in reversed(range(self.hparams.num_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            labels = (
                batch_template.y
                if (class_labels is not None and hasattr(batch_template, "y"))
                else None
            )
            cond = self.build_conditioning(t_tensor, labels, 0.0)
            cond = cond.to(device)
            noise_pred = self.predict_noise(
                x, batch_template.pos, batch_template.batch, cond
            )

            if guidance_scale > 0.0 and self.class_embed is not None:
                uncond = self.build_conditioning(t_tensor, labels, 1.0)
                uncond = uncond.to(device)
                noise_uncond = self.predict_noise(
                    x, batch_template.pos, batch_template.batch, uncond
                )
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            betas_t = self.betas[t_tensor][batch_template.batch].unsqueeze(-1)
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_tensor][
                batch_template.batch
            ].unsqueeze(-1)
            sqrt_recip_alpha = torch.sqrt(1.0 / (1.0 - betas_t))
            model_mean = sqrt_recip_alpha * (x - betas_t / sqrt_one_minus * noise_pred)

            if t > 0:
                noise = torch.randn_like(x)
                posterior_var = self.posterior_variance[t_tensor][
                    batch_template.batch
                ].unsqueeze(-1)
                x = model_mean + torch.sqrt(posterior_var) * noise
            else:
                x = model_mean

        if ema_applied:
            self.ema.restore(self.net)
        if was_training:
            self.train()

        images = []
        for i in range(num_samples):
            start = i * self.num_patches
            end = (i + 1) * self.num_patches
            patches = x[start:end]
            image = unpatchify(patches, self.hparams.patch_size)
            images.append(image)
        return torch.stack(images)


def load_data(args):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    full_train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, transform=transform, download=True
    )

    train_size = int((1.0 - args.val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    patch_pos = build_patch_grid(args.patch_size)

    def collate_fn(batch):
        data_list = []
        for image_tensor, label in batch:
            patches = patchify(image_tensor, args.patch_size)
            data = Data(
                x=patches,
                pos=patch_pos.clone(),
                y=torch.tensor(label, dtype=torch.long),
            )
            data_list.append(data)
        return Batch.from_data_list(data_list)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def main(args):
    set_seed(args.seed)
    train_loader, val_loader, _ = load_data(args)

    if args.gpus > 0:
        accelerator, devices = "gpu", args.gpus
    else:
        accelerator, devices = "cpu", "auto"

    model = CIFAR10DiffusionModel(**vars(args))

    logger = (
        pl.loggers.WandbLogger(
            project="Platonic-CIFAR10-Diffusion",
            save_dir=os.path.join(os.getcwd(), "logs"),
        )
        if args.log
        else None
    )
    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_last=True),
        TimerCallback(),
    ]
    if args.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=args.enable_progress_bar,
        gradient_clip_val=args.grad_clip,
    )

    if not args.sample_only:
        trainer.fit(model, train_loader, val_loader)
    else:
        if args.sample_ckpt is None:
            raise ValueError("sample_only=True requires --sample_ckpt")
        device = "cuda" if (args.gpus > 0 and torch.cuda.is_available()) else "cpu"
        model = CIFAR10DiffusionModel.load_from_checkpoint(
            args.sample_ckpt, map_location=device
        )
        model = model.to(device)
        samples = model.sample(
            args.sample_count, guidance_scale=args.guidance_scale, use_ema=args.use_ema
        )
        out_dir = os.path.join(os.getcwd(), "samples")
        os.makedirs(out_dir, exist_ok=True)
        for idx, img in enumerate(samples):
            img = (img.clamp(-1, 1) + 1.0) / 2.0
            torchvision.utils.save_image(
                img.cpu(), os.path.join(out_dir, f"sample_{idx:04d}.png")
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CIFAR10 Diffusion Training with Platonic Transformer"
    )
    parser.add_argument(
        "--epochs", type=int, default=400, help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup epochs"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="AdamW beta1 parameter"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_steps", type=int, default=1000, help="Diffusion steps")
    parser.add_argument(
        "--conditioning_dim",
        type=int,
        default=512,
        help="Conditioning embedding dimension",
    )
    parser.add_argument(
        "--conditioning_mlp_dim",
        type=int,
        default=2048,
        help="Hidden size of conditioning MLP",
    )
    parser.add_argument(
        "--time_embed_dim",
        type=int,
        default=2048,
        help="Hidden dimension for timestep MLP",
    )
    parser.add_argument(
        "--class_cond",
        type=eval,
        default=True,
        help="Enable class conditional guidance",
    )
    parser.add_argument(
        "--cfg_drop_prob",
        type=float,
        default=0.1,
        help="Classifier-free guidance drop probability during training",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale at sampling time",
    )
    parser.add_argument(
        "--use_ema", type=eval, default=True, help="Track EMA weights for sampling"
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.9999, help="EMA decay factor"
    )
    parser.add_argument(
        "--patch_size", type=int, default=4, help="Patch size for tokenization"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=768, help="Transformer hidden dimension"
    )
    parser.add_argument(
        "--layers", type=int, default=12, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument(
        "--ffn_dim_factor", type=int, default=4, help="FFN dimensional multiplier"
    )
    parser.add_argument(
        "--solid_name", type=str, default="cyclic_4", help="Symmetry group name"
    )
    parser.add_argument(
        "--dense_mode", type=eval, default=True, help="Operate in dense mode"
    )
    parser.add_argument(
        "--ffn_readout", type=eval, default=True, help="Use FFN readout heads"
    )
    parser.add_argument(
        "--mean_aggregation",
        type=eval,
        default=False,
        help="Use mean aggregation inside pooling",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--norm_first", type=eval, default=True, help="Pre-norm transformer blocks"
    )
    parser.add_argument(
        "--drop_path_rate", type=float, default=0.1, help="Drop path rate"
    )
    parser.add_argument(
        "--layer_scale_init_value", type=float, default=1e-5, help="Layer scale init"
    )
    parser.add_argument(
        "--attention",
        type=eval,
        default=True,
        help="Enable attention mode in PlatonicConv",
    )
    parser.add_argument("--rope_sigma", type=float, default=16.0, help="RoPE sigma")
    parser.add_argument("--ape_sigma", type=float, default=16.0, help="APE sigma")
    parser.add_argument(
        "--learned_freqs", type=eval, default=True, help="Learnable RoPE frequencies"
    )
    parser.add_argument(
        "--freq_init",
        type=str,
        default="spiral",
        choices=["random", "spiral"],
        help="RoPE init",
    )
    parser.add_argument(
        "--use_key", type=eval, default=False, help="Use key projection under RoPE"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./datasets/cifar10", help="Data directory"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Validation split from training set",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Data loader workers"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--log", type=eval, default=True, help="Enable WandB logging")
    parser.add_argument(
        "--enable_progress_bar", type=eval, default=True, help="Show progress bar"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--sample_only", type=eval, default=False, help="Skip training and only sample"
    )
    parser.add_argument(
        "--sample_ckpt", type=str, default=None, help="Checkpoint path for sampling"
    )
    parser.add_argument(
        "--sample_count", type=int, default=16, help="Number of samples to draw"
    )

    args = parser.parse_args()
    main(args)
