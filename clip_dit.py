"""
Stage 2 Training: Vision-Language-Tactile Contrastive Learning

This script fine-tunes the tactile encoder (pre-trained in Stage 1) using contrastive learning.
The model learns to align tactile representations with vision and text modalities.
Uses InfoNCE loss for multi-modal alignment.
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
from torchvision import transforms
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from tactile_encoder import DiffusionTactileEncoder
from dataloader.tag_dataset import TouchAndGoPairDataset, load_captions_csv
from cond_encoder import ConditionEncoders

# Setup logging
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs('logs', exist_ok=True)
log_name = f'logs/dit_clip_{timestamp}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_name),
        logging.StreamHandler()
    ]
)


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: query embeddings (B, D)
            k: key embeddings (B, D)
        Returns:
            InfoNCE loss
        """
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = q @ k.t() / self.tau  # (B, B)
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Stage 2: Vision-Language-Tactile Contrastive Learning")
    # Data
    p.add_argument("--dataroot", type=str, required=True, help="Root folder of Touch-and-Go dataset")
    p.add_argument("--list_dir", type=str, default="Visuo-tactile contrastive learning/dataset",
                   help="Directory containing train/test/pretrain txt files")
    p.add_argument("--split", type=str, default="train", choices=["train","pretrain"],
                   help="Which list to use for training")
    p.add_argument("--label", type=str, default="full", choices=["full","rough","hard"],
                   help="Label scheme")
    p.add_argument("--image_size", type=int, default=64, help="Input size for tactile encoder")
    
    # Conditioning
    p.add_argument("--use_text", action="store_true",
                   help="Use text captions as additional conditioning")
    p.add_argument("--captions", type=str, default=None,
                   help="Optional CSV/TSV mapping raw->caption")
    p.add_argument("--text_from_label", action="store_true",
                   help="Derive text prompts from numeric labels")
    p.add_argument("--siglip_model", type=str, default="google/siglip-base-patch16-224",
                   help="SigLIP vision backbone")
    p.add_argument("--t5_model", type=str, default="t5-base",
                   help="T5 text encoder")
    
    # Model
    p.add_argument("--embed_dim", type=int, default=-1,
                   help="Latent dim; if <0, match condition dim")
    p.add_argument("--depth", type=int, default=8, help="Number of transformer blocks")
    p.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    p.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP expansion ratio")
    p.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    p.add_argument("--pool", type=str, default="mean", choices=["mean","cls","none"],
                   help="Pooling strategy for tactile features")
    p.add_argument("--use_cond_type_embed", action="store_true",
                   help="Add learned modality embeddings to condition tokens")
    p.add_argument("--stage1_ckpt", type=str, required=True,
                   help="Path to Stage 1 checkpoint (required)")
    
    # Contrastive learning
    p.add_argument("--temperature", type=float, default=0.07,
                   help="Temperature for InfoNCE loss")
    p.add_argument("--contrast_mode", type=str, default="fused",
                   choices=["vision", "text", "fused", "all"],
                   help="Contrastive mode: vision, text, fused (vision+text), or all")
    
    # Training
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate (lower than Stage 1)")
    p.add_argument("--wd", type=float, default=0.05, help="Weight decay")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--num_workers", type=int, default=4, help="Number of data workers")
    
    # System
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="checkpoints_stage2",
                   help="Output directory for checkpoints")
    p.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    
    # Distributed
    p.add_argument("--distributed", action="store_true",
                   help="Enable DistributedDataParallel training")
    p.add_argument("--dist_backend", type=str, default="nccl")
    p.add_argument("--dist_url", type=str, default="env://")
    p.add_argument("--local_rank", type=int, default=-1)
    
    # Mixed precision
    p.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    
    return p


def collate_keep_pil(batch):
    """Custom collate: keep 'vision' as list of PILs; stack tensors."""
    vision = [b["vision"] for b in batch]
    tactile = torch.stack([b["tactile"] for b in batch], dim=0)
    label = torch.as_tensor([b["label"] for b in batch], dtype=torch.long)
    raw = [b["raw"] for b in batch]
    caption = [b.get("caption") for b in batch]
    return {"vision": vision, "tactile": tactile, "label": label, "raw": raw, "caption": caption}


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


def load_stage1_weights(model, stage1_ckpt_path, device, strict=False):
    """Load Stage 1 weights into Stage 2 model (without noise prediction head)."""
    ckpt = torch.load(stage1_ckpt_path, map_location=device)
    stage1_state_dict = ckpt["model"]
    
    # Filter out noise prediction head weights
    filtered_state_dict = {k: v for k, v in stage1_state_dict.items() 
                          if not k.startswith("noise_pred_head")}
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=strict)
    
    return missing_keys, unexpected_keys


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = setup_distributed(args)
    
    if is_main_process(args):
        logging.info("="*60)
        logging.info("Stage 2: Contrastive Learning Training")
        logging.info("="*60)
        logging.info(f"Device: {device}")
        logging.info(f"Batch size: {args.batch_size}")
        logging.info(f"Epochs: {args.epochs}")
        logging.info(f"Learning rate: {args.lr}")
        logging.info(f"Contrast mode: {args.contrast_mode}")
        logging.info(f"Stage 1 checkpoint: {args.stage1_ckpt}")
    
    # Conditioning encoders (frozen)
    cond_enc = ConditionEncoders(
        device=device,
        siglip_model=args.siglip_model,
        t5_model=args.t5_model,
        use_text=args.use_text,
    )
    
    # Transforms
    gel_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Captions
    captions = load_captions_csv(args.captions) if (args.use_text and args.captions) else None
    
    # Dataset
    dataset = TouchAndGoPairDataset(
        list_dir=args.list_dir,
        dataroot=args.dataroot,
        split=args.split,
        label=args.label,
        img_transform=None,
        gel_transform=gel_transform,
        captions=captions,
    )
    
    # DataLoader
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
        collate_fn=collate_keep_pil,
    )
    
    # Model config
    cond_dim = cond_enc.image_dim
    if args.use_text and cond_enc.text_dim is not None and cond_enc.text_dim != cond_dim:
        raise ValueError(
            f"Dimension mismatch: SigLIP={cond_dim}, T5={cond_enc.text_dim}. "
            "Use matching model sizes."
        )
    
    # For contrastive learning, embed_dim must match cond_dim
    embed_dim = cond_dim if args.embed_dim < 0 else args.embed_dim
    if embed_dim != cond_dim:
        if is_main_process(args):
            logging.warning(f"Setting embed_dim={cond_dim} to match condition dim for contrastive learning")
        embed_dim = cond_dim
    
    if is_main_process(args):
        logging.info(f"Embed dim: {embed_dim}, Cond dim: {cond_dim}")
    
    # Create model WITHOUT noise prediction head (Stage 2)
    model = DiffusionTactileEncoder(
        image_size=args.image_size,
        embed_dim=embed_dim,
        depth=args.depth,
        num_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        cond_dim=cond_dim,
        timesteps=args.timesteps,
        use_cross_attn=True,
        pool=args.pool,
        use_cond_type_embed=args.use_cond_type_embed,
        use_noise_pred_head=False,  # No noise prediction head for Stage 2
    ).to(device)
    
    # Load Stage 1 weights
    if is_main_process(args):
        logging.info(f"Loading Stage 1 weights from: {args.stage1_ckpt}")
    
    model_to_load = model.module if hasattr(model, "module") else model
    missing_keys, unexpected_keys = load_stage1_weights(
        model_to_load, args.stage1_ckpt, device, strict=False
    )
    
    if is_main_process(args):
        if missing_keys:
            logging.info(f"Missing keys (expected for Stage 2): {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys: {unexpected_keys}")
        logging.info("Stage 1 weights loaded successfully!")
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # InfoNCE Loss for contrastive learning
    criterion = InfoNCELoss(temperature=args.temperature)
    
    os.makedirs(args.out_dir, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    # Training loop
    epoch_iter = range(1, args.epochs + 1)
    if is_main_process(args):
        epoch_iter = tqdm(epoch_iter, desc="Epochs", total=args.epochs, position=0)
    
    for epoch in epoch_iter:
        if args.distributed and hasattr(loader, "sampler") and loader.sampler is not None:
            loader.sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        epoch_loss_v = 0.0  # vision loss
        epoch_loss_t = 0.0  # text loss
        epoch_loss_f = 0.0  # fused loss
        n_samples = 0
        start = time.time()
        
        batch_iter = loader
        if is_main_process(args):
            try:
                total_steps = len(loader)
            except Exception:
                total_steps = None
            batch_iter = tqdm(loader, desc=f"Epoch {epoch}", total=total_steps, position=1, leave=False)
        
        for batch in batch_iter:
            vision_pils = batch["vision"]
            tactile = batch["tactile"].to(device, non_blocking=True)
            captions_batch = batch["caption"] if args.use_text else None
            labels_batch = batch.get("label") if args.use_text else None
            
            # Encode conditions (frozen)
            with torch.no_grad():
                vis_emb = cond_enc.encode_images(vision_pils)  # (B, D)
                txt_emb = None
                if args.use_text:
                    # if captions_batch is not None:
                    #     texts = [c if (c is not None and len(c) > 0) else "" for c in captions_batch]
                    #     txt_emb = cond_enc.encode_texts(texts)  # (B, D)
                    #     print(f"text is {texts}")
                    # elif args.text_from_label and labels_batch is not None:
                    # texts = [f"object id {int(l)}" for l in labels_batch.tolist()]
                    texts = [f"{int(l)}" for l in labels_batch.tolist()]
                    txt_emb = cond_enc.encode_texts(texts)
                    # print(texts)
            
            # Build conditioning tokens
            if txt_emb is not None:
                cond_tokens = torch.stack([vis_emb, txt_emb], dim=1)  # (B, 2, D)
                cond_mod_ids = torch.tensor([0, 1], device=device).unsqueeze(0).repeat(vis_emb.size(0), 1)
                fused_cond = F.normalize((vis_emb + txt_emb) / 2.0, dim=-1)
            else:
                cond_tokens = vis_emb.unsqueeze(1)  # (B, 1, D)
                cond_mod_ids = torch.zeros((vis_emb.size(0), 1), device=device, dtype=torch.long)
                fused_cond = vis_emb
            
            # Sample random timesteps
            t = torch.randint(0, args.timesteps, (tactile.size(0),), device=device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=args.amp):
                z_tactile = model(
                    tactile,
                    t,
                    condition_tokens=cond_tokens,
                    condition_modality_ids=cond_mod_ids if args.use_cond_type_embed else None,
                )  # (B, D)
                
                # Contrastive loss based on mode
                loss = 0.0
                loss_v = 0.0
                loss_t = 0.0
                loss_f = 0.0
                
                if args.contrast_mode in ["vision", "all"]:
                    loss_v = criterion(z_tactile, vis_emb)
                    loss += loss_v
                
                if args.contrast_mode in ["text", "all"] and txt_emb is not None:
                    loss_t = criterion(z_tactile, txt_emb)
                    loss += loss_t
                
                if args.contrast_mode in ["fused", "all"]:
                    loss_f = criterion(z_tactile, fused_cond)
                    loss += loss_f
                
                # Average if using "all" mode
                if args.contrast_mode == "all":
                    n_losses = 2 if txt_emb is not None else 1
                    loss = loss / n_losses
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            bs = tactile.size(0)
            epoch_loss += loss.item() * bs
            epoch_loss_v += loss_v.item() * bs if isinstance(loss_v, torch.Tensor) else 0
            epoch_loss_t += loss_t.item() * bs if isinstance(loss_t, torch.Tensor) else 0
            epoch_loss_f += loss_f.item() * bs if isinstance(loss_f, torch.Tensor) else 0
            n_samples += bs
            
            if is_main_process(args):
                try:
                    lr_now = scheduler.get_last_lr()[0]
                except Exception:
                    lr_now = optimizer.param_groups[0]["lr"]
                if hasattr(batch_iter, "set_postfix"):
                    batch_iter.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")
        
        scheduler.step()
        
        dt = time.time() - start
        avg_loss = epoch_loss / max(n_samples, 1)
        avg_loss_v = epoch_loss_v / max(n_samples, 1)
        avg_loss_t = epoch_loss_t / max(n_samples, 1)
        avg_loss_f = epoch_loss_f / max(n_samples, 1)
        
        if is_main_process(args):
            log_msg = (
                f"Epoch {epoch:03d}/{args.epochs}  Loss={avg_loss:.4f}  "
                f"LR={scheduler.get_last_lr()[0]:.2e}  Time={dt:.1f}s"
            )
            if args.contrast_mode in ["vision", "all"]:
                log_msg += f"  Loss_V={avg_loss_v:.4f}"
            if args.contrast_mode in ["text", "all"]:
                log_msg += f"  Loss_T={avg_loss_t:.4f}"
            if args.contrast_mode in ["fused", "all"]:
                log_msg += f"  Loss_F={avg_loss_f:.4f}"
            
            logging.info(log_msg)
            
            if epoch % args.save_every == 0 or epoch == args.epochs:
                model_to_save = model.module if hasattr(model, "module") else model
                ckpt = {
                    "model": model_to_save.state_dict(),
                    "config": vars(args),
                    "epoch": epoch,
                    "stage": 2,
                }
                path = os.path.join(args.out_dir, f"stage2_epoch{epoch:03d}.pt")
                torch.save(ckpt, path)
                logging.info(f"Saved checkpoint: {path}")
    
    if is_main_process(args):
        logging.info("Stage 2 training completed!")


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.stage1_ckpt):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {args.stage1_ckpt}")
    
    if args.contrast_mode in ["text", "all"] and not args.use_text:
        raise ValueError("--use_text must be enabled when using text or all contrast mode")
    
    try:
        main(args)
    finally:
        cleanup_distributed(args)
