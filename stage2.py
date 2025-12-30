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
from models.unet import UNet
from models.ldm import LatentDiffusionModel
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
    p = argparse.ArgumentParser("Stage 2: Latent Diffusion Model Training")
    
    p.add_argument("--dataroot", type=str, required=True)
    p.add_argument("--list_dir", type=str, required=True)
    p.add_argument("--captions_file", type=str, default=None)
    p.add_argument("--label", type=str, default="full", choices=["full", "rough", "hard"])
    
    p.add_argument("--autoencoder_checkpoint", type=str, required=True, help="Path to pretrained autoencoder from stage 1")
    p.add_argument("--siglip_model_path", type=str, required=True, help="Path to SigLIP model")
    p.add_argument("--t5_model_path", type=str, required=True, help="Path to T5 model")
    
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=128)
    p.add_argument("--cond_dim", type=int, default=768)
    p.add_argument("--timesteps", type=int, default=1000)
    
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup_epochs", type=int, default=10)
    
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_dir", type=str, default="checkpoints/stage2")
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=10)
    
    return p


class SigLIPWrapper(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        
    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values=pixel_values)
            return outputs.pooler_output


class T5Wrapper(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
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


def train_epoch(model, dataloader, optimizer, siglip_model, t5_wrapper, epoch, rank, args):
    model.train()
    total_loss = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for step, batch in enumerate(pbar):
        vision = batch["vision"].cuda(non_blocking=True)
        tactile = batch["tactile"].cuda(non_blocking=True)
        # captions = batch["caption"]
        label_ =  batch["label"].tolist()
        label_ = [str(x) for x in label_]
        
        vision_feat = siglip_model(vision)
        
        # text_inputs = t5_wrapper.tokenize(label_, vision.device)
        # text_feat = t5_wrapper(**text_inputs)
        
        # cond = torch.cat([vision_feat, text_feat], dim=-1)
        cond = vision_feat
        
        loss = model(tactile, cond)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if rank == 0 and step % args.log_interval == 0:
            avg_loss = total_loss / (step + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / len(dataloader)


def main():
    args = build_argparser().parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = os.path.join(args.save_dir, f'train_stage2_{timestamp}.log')
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
    
    captions = load_captions_csv(args.captions_file) if args.captions_file else None
    
    train_dataset = TouchAndGoPairDataset(
        list_dir=args.list_dir,
        dataroot=args.dataroot,
        split="train",
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
        base_channels=64
    ).cuda()
    
    if rank == 0:
        logging.info(f"Loading autoencoder from {args.autoencoder_checkpoint}")
    ckpt = torch.load(args.autoencoder_checkpoint, map_location='cuda')
    if 'model' in ckpt:
        state_dict = ckpt['model']
        autoencoder_state = {k.replace('autoencoder.', ''): v for k, v in state_dict.items() if k.startswith('autoencoder.')}
        autoencoder.load_state_dict(autoencoder_state, strict=False)
    else:
        autoencoder.load_state_dict(ckpt, strict=False)
    
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.eval()
    
    unet = UNet(
        in_channels=args.latent_dim,
        out_channels=args.latent_dim,
        base_channels=args.base_channels,
        # cond_dim=args.cond_dim * 2,
        cond_dim=args.cond_dim,
    ).cuda()
    
    ldm = LatentDiffusionModel(
        autoencoder=autoencoder,
        unet=unet,
        timesteps=args.timesteps,
        latent_dim=args.latent_dim,
    ).cuda()
    
    siglip_model = SigLIPWrapper(args.siglip_model_path).cuda()
    t5_wrapper = T5Wrapper(args.t5_model_path).cuda()
    
    if world_size > 1:
        ldm = DDP(ldm, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(
        [p for p in ldm.parameters() if p.requires_grad],
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
    
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        avg_loss = train_epoch(ldm, train_loader, optimizer, siglip_model, t5_wrapper, epoch, rank, args)
        
        for _ in range(len(train_loader)):
            scheduler.step()
            global_step += 1
        
        if rank == 0:
            logging.info(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")
            
            if epoch % args.save_interval == 0:
                save_dict = {
                    'epoch': epoch,
                    'model': ldm.module.state_dict() if world_size > 1 else ldm.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args),
                }
                save_path = os.path.join(args.save_dir, f'stage2_epoch_{epoch}.pt')
                torch.save(save_dict, save_path)
                logging.info(f"Saved checkpoint to {save_path}")
    
    if rank == 0:
        final_path = os.path.join(args.save_dir, 'stage2_final.pt')
        save_dict = {
            'epoch': args.epochs,
            'model': ldm.module.state_dict() if world_size > 1 else ldm.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
        }
        torch.save(save_dict, final_path)
        logging.info(f"Training completed. Final model saved to {final_path}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
