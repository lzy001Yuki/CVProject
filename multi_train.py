"""
Training script using Multi-Dataset Loader

Supports training with Touch-and-Go, Feeling, Octopi, and other datasets combined.
"""
import argparse
import os
import time
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from dataloader.multi_dataset import MultiTactileDataset, collate_multi_dataset
from cond_encoder import ConditionEncoders

# Import your encoder (choose one)
# from tactile_encoder import DiffusionTactileEncoder
# from dino_encoder import TactileVisionTower

# Setup logging
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs('logs', exist_ok=True)
log_name = f'logs/multi_dataset_{timestamp}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_name),
        logging.StreamHandler()
    ]
)

def setup_distributed(args: argparse.Namespace):
    if not args.distributed:
        args.rank = 0
        args.world_size = 1
        return torch.device(args.device)
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
    return torch.device("cuda", args.local_rank)

def is_main_process(args: argparse.Namespace) -> bool:
    return getattr(args, "rank", 0) == 0

def cleanup_distributed(args: argparse.Namespace):
    if args.distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = q @ k.t() / self.tau
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Multi-Dataset Tactile Training")
    
    # Multi-dataset configuration
    p.add_argument("--tag_root", type=str, default=None,
                   help="Root directory for Touch-and-Go dataset")
    p.add_argument("--feeling_root", type=str, default=None,
                   help="Root directory for Feeling dataset")
    p.add_argument("--octopi_root", type=str, default=None,
                   help="Root directory for Octopi dataset")
    p.add_argument("--max_samples_per_dataset", type=int, default=None,
                   help="Max samples to use from each dataset (None = all)")
    
    # Model selection
    p.add_argument("--encoder", type=str, default="dino", choices=["dit", "dino"],
                   help="Which encoder to use: dit or dino")
    
    # Common model params
    p.add_argument("--image_size", type=int, default=224,
                   help="Input image size")
    p.add_argument("--embed_dim", type=int, default=768,
                   help="Embedding dimension")
    
    # DiT-specific params
    p.add_argument("--patch_size_dit", type=int, default=8,
                   help="Patch size for DiT encoder")
    p.add_argument("--depth_dit", type=int, default=8,
                   help="Depth for DiT encoder")
    p.add_argument("--timesteps", type=int, default=1000,
                   help="Diffusion timesteps for DiT")
    
    # DINO-specific params
    p.add_argument("--patch_size_dino", type=int, default=16,
                   help="Patch size for DINO encoder")
    p.add_argument("--transformer_layers", type=int, default=6,
                   help="Transformer layers for DINO")
    p.add_argument("--local_blocks", type=int, default=2,
                   help="Local CNN blocks for DINO")
    
    # Conditioning
    p.add_argument("--siglip_model", type=str, default="google/siglip-base-patch16-224",
                   help="SigLIP vision backbone")
    p.add_argument("--t5_model", type=str, default="t5-base",
                   help="T5 text encoder")
    p.add_argument("--use_vision_cond", action="store_true",
                   help="Use vision as conditioning (only for datasets with vision)")
    
    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    # Distributed
    p.add_argument("--distributed", action="store_true", help="Enable DistributedDataParallel training")
    p.add_argument("--dist_backend", type=str, default="nccl")
    p.add_argument("--dist_url", type=str, default="env://")
    p.add_argument("--local_rank", type=int, default=-1)
    
    # System
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="checkpoints_multi")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--amp", action="store_true", help="Mixed precision training")
    
    return p


def main(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    device = setup_distributed(args)
    
    if is_main_process(args):
        logging.info("="*60)
        logging.info("Multi-Dataset Tactile Training")
        logging.info("="*60)
        logging.info(f"Device: {device}")
        logging.info(f"Encoder: {args.encoder}")
    
    # Build dataroots dict
    dataroots = {}
    if args.tag_root:
        dataroots['tag'] = args.tag_root
    if args.feeling_root:
        dataroots['feeling'] = args.feeling_root
    if args.octopi_root:
        dataroots['octopi'] = args.octopi_root
    
    if not dataroots:
        raise ValueError("At least one dataset root must be provided!")
    
    if is_main_process(args):
        logging.info(f"Datasets: {list(dataroots.keys())}")
    
    # Create dataset
    dataset = MultiTactileDataset(
        dataroots=dataroots,
        mode='train',
        image_size=args.image_size,
        use_augmentation=True,
        max_samples_per_dataset=args.max_samples_per_dataset,
    )
    
    # DataLoader
    # Sampler for DDP
    if args.distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_multi_dataset,
    )
    
    # Conditioning encoders (for vision-conditioned training)
    cond_enc = None
    if args.use_vision_cond:
        cond_enc = ConditionEncoders(
            device=device,
            siglip_model=args.siglip_model,
            t5_model=args.t5_model,
            use_text=False,  # Multi-dataset doesn't have text
        )
        logging.info("Vision conditioning enabled")
    
    # Create model based on encoder type
    if args.encoder == "dit":
        from tactile_encoder import DiffusionTactileEncoder
        
        model = DiffusionTactileEncoder(
            image_size=args.image_size,
            patch_size=args.patch_size_dit,
            embed_dim=args.embed_dim,
            depth=args.depth_dit,
            timesteps=args.timesteps,
            use_cross_attn=args.use_vision_cond,
            cond_dim=768 if args.use_vision_cond else 0,
            use_noise_pred_head=False,  # Contrastive mode
            pool="mean",
        ).to(device)
        
    elif args.encoder == "dino":
        from dino_encoder import TactileVisionTower
        
        model = TactileVisionTower(
            image_size=args.image_size,
            patch_size=args.patch_size_dino,
            embed_dim=args.embed_dim,
            local_blocks=args.local_blocks,
            transformer_layers=args.transformer_layers,
            use_cls_token=True,
            enable_dino=False,
        ).to(device)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main_process(args):
        logging.info(f"Model parameters: {num_params:.2f}M")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss (simple self-supervised contrastive loss)
    # For multi-dataset training without labels, we use temporal consistency
    # or cross-sensor consistency (if available)
    criterion = InfoNCELoss(temperature=args.temperature)
    
    os.makedirs(args.out_dir, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        if args.distributed and hasattr(loader, "sampler") and loader.sampler is not None:
            loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        start = time.time()
        
        batch_iter = loader
        if is_main_process(args):
            batch_iter = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        
        for batch in batch_iter:
            tactile = batch['tactile'].to(device, non_blocking=True)
            vision_pils = batch['vision']
            
            # Encode vision if available and conditioning is enabled
            if args.use_vision_cond and cond_enc is not None:
                # Filter out None values
                valid_vision = [v for v in vision_pils if v is not None]
                if valid_vision:
                    with torch.no_grad():
                        vis_emb = cond_enc.encode_images(valid_vision)
                    # Pad to match batch size
                    if len(valid_vision) < len(tactile):
                        # Use mean embedding for missing vision
                        mean_emb = vis_emb.mean(dim=0, keepdim=True)
                        padding = mean_emb.repeat(len(tactile) - len(valid_vision), 1)
                        vis_emb = torch.cat([vis_emb, padding], dim=0)
                    cond_tokens = vis_emb.unsqueeze(1)  # (B, 1, D)
                else:
                    cond_tokens = None
            else:
                cond_tokens = None
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=args.amp):
                if args.encoder == "dit":
                    # Sample random timesteps for DiT
                    t = torch.randint(0, args.timesteps, (tactile.size(0),), device=device)
                    features = model(tactile, t, condition_tokens=cond_tokens)
                else:
                    # DINO encoder
                    patch_features = model(tactile)  # (B, N, D)
                    features = patch_features.mean(dim=1)  # (B, D)
                
                # Simple contrastive loss: features should be similar within batch
                # (assuming batch contains related samples)
                # For proper training, you should implement dataset-specific losses
                features = F.normalize(features, dim=-1)
                
                # Self-supervised loss: maximize agreement with shifted/augmented versions
                # Here we use a simple consistency loss
                # TODO: Implement proper contrastive objective based on your needs
                loss = torch.tensor(0.0, device=device)  # Placeholder
                
                # Example: If you have pairs in batch, use contrastive loss
                if len(features) >= 2:
                    # Simple pairwise contrastive
                    loss = criterion(features[:len(features)//2], features[len(features)//2:])
            
            if loss.item() > 0:  # Only backprop if there's actual loss
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                bs = tactile.size(0)
                epoch_loss += loss.item() * bs
                n_samples += bs
            
            if is_main_process(args) and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        
        dt = time.time() - start
        avg_loss = epoch_loss / max(n_samples, 1)
        
        if is_main_process(args):
            logging.info(
                f"Epoch {epoch:03d}/{args.epochs}  Loss={avg_loss:.4f}  "
                f"LR={scheduler.get_last_lr()[0]:.2e}  Time={dt:.1f}s"
            )
        
        if is_main_process(args) and (epoch % args.save_every == 0 or epoch == args.epochs):
            model_to_save = model.module if hasattr(model, "module") else model
            ckpt = {
                "model": model_to_save.state_dict(),
                "config": vars(args),
                "epoch": epoch,
            }
            path = os.path.join(args.out_dir, f"multi_dataset_epoch{epoch:03d}.pt")
            torch.save(ckpt, path)
            logging.info(f"Saved checkpoint: {path}")
    
    if is_main_process(args):
        logging.info("Training completed!")

    cleanup_distributed(args)


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    
    main(args)
