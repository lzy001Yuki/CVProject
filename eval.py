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
from models.ldm import LatentDiffusionModel, VisionTactileEncoder
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
    p = argparse.ArgumentParser("Linear Probe Evaluation for Vision-Tactile Encoder")
    
    p.add_argument("--dataroot", type=str, required=True)
    p.add_argument("--list_dir", type=str, required=True)
    p.add_argument("--captions_file", type=str, default=None)
    p.add_argument("--label", type=str, default="full", choices=["full", "rough", "hard"])
    
    p.add_argument("--ldm_checkpoint", type=str, required=True, help="Path to trained LDM checkpoint")
    p.add_argument("--siglip_model_path", type=str, required=True, help="Path to SigLIP model")
    p.add_argument("--t5_model_path", type=str, required=True, help="Path to T5 model")
    
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=128)
    p.add_argument("--cond_dim", type=int, default=768)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--feature_dim", type=int, default=768)
    p.add_argument("--pool", type=str, default="mean", choices=["mean", "max", "flatten"])
    
    p.add_argument("--sample_timestep", type=int, default=None, help="Fixed timestep for sampling (default: random)")
    p.add_argument("--use_conditioning", action="store_true", help="Use vision+text conditioning during inference")
    
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--probe_epochs", type=int, default=30)
    p.add_argument("--probe_lr", type=float, default=1e-2)
    p.add_argument("--probe_wd", type=float, default=0.0)
    
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_dir", type=str, default="checkpoints/eval")
    p.add_argument("--log_interval", type=int, default=50)
    
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


def get_num_classes(label_scheme):
    if label_scheme == "full":
        return 20
    elif label_scheme in ("rough", "hard"):
        return 2
    raise ValueError(f"Unknown label scheme: {label_scheme}")


@torch.no_grad()
def extract_features(encoder, batch, siglip_model, t5_wrapper, args):
    vision = batch["vision"].cuda(non_blocking=True)
    tactile = batch["tactile"].cuda(non_blocking=True)
    labels = batch["label"].cuda(non_blocking=True)
    # print(labels)
    label_ =  batch["label"].tolist()
    label_ = [str(x) for x in label_]
    # print(batch)
    # captions = batch["caption"]
    
    
    if args.sample_timestep is not None:
        t = torch.full((tactile.size(0),), args.sample_timestep, device=tactile.device, dtype=torch.long)
    else:
        t = torch.randint(0, args.timesteps, (tactile.size(0),), device=tactile.device, dtype=torch.long)
    
    cond = None
    if args.use_conditioning:
        vision_feat = siglip_model(vision)
        # text_inputs = t5_wrapper.tokenize(captions, vision.device)
        # text_inputs = t5_wrapper.tokenize(label_, vision.device)
        # text_feat = t5_wrapper(**text_inputs)
        # text_feat = torch.zeros(vision.size(0), 768, device=vision.device)  # 无文本
        # cond = torch.cat([vision_feat, text_feat], dim=-1)
        # cond = vision_feat
        cond = torch.cat([vision_feat, vision_feat], dim=-1)
        # cond = text_feat
    features = encoder(tactile, t, cond)
    # print(features.shape)
    # print(vision_feat.shape)
    # exit(0)
    # if args.use_conditioning:
    #     features = torch.cat([features, vision_feat], dim=-1)
    # print(features.shape)
    # exit(0)
    
    return features, labels


def train_probe_epoch(encoder, classifier, dataloader, optimizer, criterion, siglip_model, t5_wrapper, epoch, rank, args):
    classifier.train()
    encoder.eval()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    else:
        pbar = dataloader
    
    for step, batch in enumerate(pbar):
        features, labels = extract_features(encoder, batch, siglip_model, t5_wrapper, args)
        
        logits = classifier(features)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = logits.argmax(dim=1)
        correct = (pred == labels).sum().item()
        
        total_loss += loss.item() * labels.size(0)
        total_correct += correct
        total_samples += labels.size(0)
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{total_loss/total_samples:.4f}',
                'acc': f'{100*total_correct/total_samples:.2f}%'
            })
    
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_probe(encoder, classifier, dataloader, siglip_model, t5_wrapper, rank, args):
    classifier.eval()
    encoder.eval()
    
    total_correct = 0
    total_samples = 0
    
    n_classes = get_num_classes(args.label)
    per_class_correct = torch.zeros(n_classes, dtype=torch.long, device='cuda')
    per_class_total = torch.zeros(n_classes, dtype=torch.long, device='cuda')
    
    if rank == 0:
        pbar = tqdm(dataloader, desc="Evaluating")
    else:
        pbar = dataloader
    
    for batch in pbar:
        features, labels = extract_features(encoder, batch, siglip_model, t5_wrapper, args)
        
        logits = classifier(features)
        pred = logits.argmax(dim=1)
        
        correct = (pred == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)
        
        for c in range(n_classes):
            mask = (labels == c)
            per_class_total[c] += mask.sum()
            per_class_correct[c] += (pred[mask] == c).sum()

    if dist.is_initialized():
        total_correct_tensor = torch.tensor(total_correct, device='cuda')
        total_samples_tensor = torch.tensor(total_samples, device='cuda')
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(per_class_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(per_class_total, op=dist.ReduceOp.SUM)
        total_correct = total_correct_tensor.item()
        total_samples = total_samples_tensor.item()
    
    accuracy = total_correct / max(total_samples, 1)
    per_class_acc = [(per_class_correct[c].item() / max(per_class_total[c].item(), 1)) for c in range(n_classes)]
    
    return accuracy, per_class_acc


def main():
    args = build_argparser().parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = os.path.join(args.save_dir, f'eval_linear_probe_{timestamp}.log')
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
    
    test_dataset = TouchAndGoPairDataset(
        list_dir=args.list_dir,
        dataroot=args.dataroot,
        split="test",
        label=args.label,
        img_transform=img_transform,
        gel_transform=gel_transform,
        captions=captions,
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    autoencoder = AutoEncoder(
        in_channels=3,
        latent_dim=args.latent_dim,
        base_channels=64
    ).cuda()
    
    unet = UNet(
        in_channels=args.latent_dim,
        out_channels=args.latent_dim,
        base_channels=args.base_channels,
        cond_dim=args.cond_dim * 2 if args.use_conditioning else None,
        # cond_dim=args.cond_dim,
    ).cuda()
    
    ldm = LatentDiffusionModel(
        autoencoder=autoencoder,
        unet=unet,
        timesteps=args.timesteps,
        latent_dim=args.latent_dim,
    ).cuda()
    
    if rank == 0:
        logging.info(f"Loading LDM checkpoint from {args.ldm_checkpoint}")
    ckpt = torch.load(args.ldm_checkpoint, map_location='cuda')
    if 'model' in ckpt:
        ldm.load_state_dict(ckpt['model'], strict=False)
    else:
        ldm.load_state_dict(ckpt, strict=False)
    
    encoder = VisionTactileEncoder(
        ldm=ldm,
        feature_dim=args.feature_dim,
        pool=args.pool
    ).cuda()
    encoder.eval()
    
    siglip_model = None
    t5_wrapper = None
    if args.use_conditioning:
        siglip_model = SigLIPWrapper(args.siglip_model_path).cuda()
        t5_wrapper = T5Wrapper(args.t5_model_path).cuda()
    
    n_classes = get_num_classes(args.label)
    classifier = nn.Linear(args.feature_dim, n_classes).cuda()
    # classifier = nn.Linear(args.feature_dim * 2, n_classes).cuda()
    if world_size > 1:
        classifier = DDP(classifier, device_ids=[local_rank])
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.probe_lr,
        momentum=0.9,
        weight_decay=args.probe_wd
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.probe_epochs)
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.probe_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_probe_epoch(
            encoder, classifier, train_loader, optimizer, criterion,
            siglip_model, t5_wrapper, epoch, rank, args
        )
        
        test_acc, per_class_acc = evaluate_probe(
            encoder, classifier, test_loader,
            siglip_model, t5_wrapper, rank, args
        )
        
        scheduler.step()
        
        if rank == 0:
            logging.info(
                f"Epoch {epoch}/{args.probe_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Test Acc: {test_acc*100:.2f}%"
            )
            logging.info(f"Per-class accuracy: {[f'{acc*100:.2f}%' for acc in per_class_acc]}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                
                save_dict = {
                    'epoch': epoch,
                    'classifier': classifier.module.state_dict() if world_size > 1 else classifier.state_dict(),
                    'test_acc': test_acc,
                    'per_class_acc': per_class_acc,
                    'args': vars(args),
                }
                save_path = os.path.join(args.save_dir, f'linear_probe_best_{args.label}.pt')
                torch.save(save_dict, save_path)
                logging.info(f"Saved best probe (acc={test_acc*100:.2f}%) to {save_path}")
    
    if rank == 0:
        logging.info(f"Evaluation completed. Best accuracy: {best_acc*100:.2f}% at epoch {best_epoch}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
