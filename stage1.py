import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
import logging
from datetime import datetime
from tqdm import tqdm

from models.autoencoder import AutoEncoder
from models.contrasive import MultiModalContrastiveModel
from dataloader.tag_dataset import TouchAndGoPairDataset, load_captions_csv


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_argparser():
    p = argparse.ArgumentParser("Stage 1: Contrastive Learning for Vision-Tactile Encoder")
    
    p.add_argument("--dataroot", type=str, required=True)
    p.add_argument("--list_dir", type=str, required=True)
    p.add_argument("--captions_file", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--label", type=str, default="full", choices=["full", "rough", "hard"])
    
    p.add_argument("--siglip_model_path", type=str, required=True, help="Path to SigLIP model")
    p.add_argument("--t5_model_path", type=str, required=True, help="Path to T5 model")
    
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--projection_dim", type=int, default=512)
    p.add_argument("--base_channels", type=int, default=64)
    
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup_epochs", type=int, default=5)
    
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_dir", type=str, default="checkpoints/stage1_con")
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=10)
    
    
    p.add_argument("--resume", type=str, default=None,
               help="Path to checkpoint to resume training from")
    
    return p


class SigLIPWrapper(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        
    def forward(self, pixel_values):
        outputs = self.model.vision_model(pixel_values=pixel_values)
        return outputs.pooler_output


class T5Wrapper(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state.mean(dim=1)
    
    def tokenize(self, texts, device):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        return {k: v.to(device) for k, v in encoded.items()}


def collate_fn(batch):
    vision = torch.stack([b["vision"] for b in batch])
    tactile = torch.stack([b["tactile"] for b in batch])
    label = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    caption = [b.get("caption", "") or "" for b in batch]
    return {"vision": vision, "tactile": tactile, "label": label, "caption": caption}


def train_epoch(model, dataloader, optimizer, t5_wrapper, epoch, rank, args):
    model.train()
    total_loss = 0
    total_loss_tv = 0
    total_loss_tt = 0
    total_loss_vt = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for step, batch in enumerate(pbar):
        vision = batch["vision"].cuda(non_blocking=True)
        tactile = batch["tactile"].cuda(non_blocking=True)
        label_ =  batch["label"].tolist()
        label_ = [str(x) for x in label_]
        # captions = batch["caption"]
        
        text_inputs = t5_wrapper.tokenize(label_, vision.device)
        
        loss, loss_dict = model(tactile, vision, text_inputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_tv += loss_dict['loss_tactile_vision']
        total_loss_tt += loss_dict['loss_tactile_text']
        total_loss_vt += loss_dict['loss_vision_text']
        
        if rank == 0 and step % args.log_interval == 0:
            avg_loss = total_loss / (step + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'tv': f'{total_loss_tv/(step+1):.4f}',
                'tt': f'{total_loss_tt/(step+1):.4f}',
                'vt': f'{total_loss_vt/(step+1):.4f}',
            })
    
    return total_loss / len(dataloader)


def main():
    args = build_argparser().parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = os.path.join(args.save_dir, f'train_stage1_{timestamp}.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info(f"Arguments: {args}")
    
    gel_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    img_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    captions = None
    train_dataset = None
    if args.all:
        ######
        pass
    else:
        train_dataset = TouchAndGoPairDataset(
            list_dir=args.list_dir,
            dataroot=args.dataroot,
            split="pretrain",
            label=args.label,
            img_transform=img_transform,
            gel_transform=gel_transform,
            captions=captions,
        )
    
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    autoencoder = AutoEncoder(
        in_channels=3,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels
    ).cuda()
    
    siglip_model = SigLIPWrapper(args.siglip_model_path).cuda()
    t5_model = T5Wrapper(args.t5_model_path).cuda()
    t5_wrapper = t5_model
    
    model = MultiModalContrastiveModel(
        autoencoder=autoencoder,
        siglip_model=siglip_model,
        t5_model=t5_model,
        latent_dim=args.latent_dim,
        embed_dim=args.embed_dim,
        projection_dim=args.projection_dim,
    ).cuda()
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.wd
    )
    
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + torch.cos(torch.tensor((step - warmup_steps) / (total_steps - warmup_steps) * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if args.resume:
        if rank == 0:
            logging.info(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=f'cuda:{local_rank}')
        
        # Load model state dict
        model_to_load = model.module if world_size > 1 else model
        model_to_load.load_state_dict(ckpt['model'])

        # Optionally load optimizer and scheduler (recommended for full resume)
        optimizer.load_state_dict(ckpt['optimizer'])
    
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        avg_loss = train_epoch(model, train_loader, optimizer, t5_wrapper, epoch, rank, args)
        scheduler.step()
        
        if rank == 0:
            logging.info(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")
            
            if epoch % args.save_interval == 0:
                save_dict = {
                    'epoch': epoch,
                    'model': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args),
                }
                save_path = os.path.join(args.save_dir, f'stage1_epoch_{epoch}.pt')
                torch.save(save_dict, save_path)
                logging.info(f"Saved checkpoint to {save_path}")
    
    if rank == 0:
        final_path = os.path.join(args.save_dir, 'stage1_final.pt')
        save_dict = {
            'epoch': args.epochs,
            'model': model.module.state_dict() if world_size > 1 else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
        }
        torch.save(save_dict, final_path)
        logging.info(f"Training completed. Final model saved to {final_path}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
