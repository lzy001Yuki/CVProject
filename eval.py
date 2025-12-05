import argparse
import os
import random
import time
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from tactile_encoder import DiffusionTactileEncoder
from dataloader.tag_dataset import TouchAndGoPairDataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_name=f'logs/eval_dit_{timestamp}.log'
logging.basicConfig(filename=log_name, level=logging.DEBUG)

def collate_keep_pil(batch):
    vision = [b["vision"] for b in batch]  # not used, keep as list
    tactile = torch.stack([b["tactile"] for b in batch], dim=0)
    label = torch.as_tensor([b["label"] for b in batch], dtype=torch.long)
    raw = [b["raw"] for b in batch]
    caption = [b.get("caption") for b in batch]
    return {"vision": vision, "tactile": tactile, "label": label, "raw": raw, "caption": caption}


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    print(f"prediction {pred}")
    print(f"groundTruth {targets}")
    return (pred == targets).float().mean().item()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Linear probe on Diffusion Tactile Encoder latents")
    # Data
    p.add_argument("--dataroot", type=str, required=True, help="Root folder of Touch-and-Go dataset content")
    p.add_argument("--list_dir", type=str, required=True, help="Directory containing train.txt/test.txt")
    p.add_argument("--label", type=str, default="full", choices=["full","rough","hard"], help="Label scheme")
    p.add_argument("--image_size", type=int, default=64, help="Tactile resize (HxW)")
    # Model / checkpoint
    p.add_argument("--model", type=str, required=True, help="choose your encoder type and register into eval.py")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained tactile encoder checkpoint (*.pt)")
    p.add_argument("--pool", type=str, default="mean", choices=["mean","cls","none"], help="Encoder output pooling")
    p.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps (for encoder consistency)")
    # Probe
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--balance", type=str, default="none", choices=["none","loss","sampler","both"], help="Handle class imbalance via weighted loss and/or weighted sampler")
    p.add_argument("--class_weights", type=str, default=None, help="Optional comma-separated class weights; overrides auto-computed weights")
    # System
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="probes")
    return p


def get_num_classes(label_scheme: str) -> int:
    if label_scheme == "full":
        return 20
    if label_scheme in ("rough", "hard"):
        return 2
    raise ValueError(f"Unknown label scheme: {label_scheme}")


def _parse_class_weights(n_classes: int, weights_str: str):
    vals = [float(x) for x in weights_str.split(",")]
    if len(vals) != n_classes:
        raise ValueError(f"--class_weights length {len(vals)} != n_classes {n_classes}")
    return torch.tensor(vals, dtype=torch.float)


def _compute_class_weights(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    # Inverse frequency weighting: w_c = N / (C * n_c)
    counts = torch.bincount(labels, minlength=n_classes).float()
    N = counts.sum().clamp_min(1.0)
    C = float(n_classes)
    weights = N / (counts.clamp_min(1.0) * C)
    return weights

def build_dataloaders(args: argparse.Namespace):
    gel_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_ds = TouchAndGoPairDataset(
        list_dir=args.list_dir,
        dataroot=args.dataroot,
        split="train",
        label=args.label,
        img_transform=None,
        gel_transform=gel_transform,
        captions=None,
    )
    

    test_ds = TouchAndGoPairDataset(
        list_dir=args.list_dir,
        dataroot=args.dataroot,
        split="test",
        label=args.label,
        img_transform=None,
        gel_transform=gel_transform,
        captions=None,
    )
    
    sampler = None
    shuffle = True
    print(f"dataset building=================================")
    if args.balance in ("sampler", "both"):
        # train_labels = torch.as_tensor([train_ds[i]["label"] for i in range(len(train_ds))], dtype=torch.long)
        train_labels = train_ds.train_labels
        class_weights = _compute_class_weights(train_labels, get_num_classes(args.label))
        sample_weights = class_weights[train_labels]
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(train_labels), replacement=True)
        shuffle = False

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=args.batch_size,
    #     shuffle=shuffle,
    #     sampler=sampler,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     collate_fn=collate_keep_pil,
    # )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        # sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_keep_pil,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_keep_pil,
    )

    return train_loader, test_loader


def load_encoder(args: argparse.Namespace, device: torch.device) -> Tuple[DiffusionTactileEncoder, int]:
    ckpt = torch.load(args.checkpoint, map_location=device)
    if args.model=="dit":
        cfg = ckpt.get("config", {})

        # Resolve dims robustly
        cond_dim = int(cfg.get("cond_dim", 768))
        saved_embed = cfg.get("embed_dim", cond_dim)
        try:
            saved_embed = int(saved_embed)
        except Exception:
            saved_embed = cond_dim
        embed_dim = cond_dim if (saved_embed is None or saved_embed <= 0) else saved_embed

        model = DiffusionTactileEncoder(
            image_size=args.image_size,
            embed_dim=embed_dim,
            depth=cfg.get("depth", 8),
            num_heads=cfg.get("heads", 8),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            cond_dim=cond_dim,
            timesteps=args.timesteps,
            use_cross_attn=True,
            pool=args.pool,
            use_cond_type_embed=cfg.get("use_cond_type_embed", False),
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=False)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        return model, embed_dim
    elif args.model == "dino" :
        if "args" in ckpt:
            cfg = ckpt["args"]
        elif "config" in ckpt:
            cfg = ckpt["config"]
        else:
            cfg = {}

        image_size = cfg.get("image_size", args.image_size)
        embed_dim = cfg.get("embed_dim", 384)
        patch_size = cfg.get("patch_size", 8)
        local_blocks = cfg.get("local_blocks", 2)
        transformer_layers = cfg.get("transformer_layers", 4)
        nhead = cfg.get("nhead", 6)
        mlp_dim = cfg.get("mlp_dim", 1024)
        use_cls_token = cfg.get("use_cls_token", args.use_cls_token)
        dino_out_dim = cfg.get("dino_out_dim", 256)  # 不用于特征提取

        model = TactileVisionTower(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            local_blocks=local_blocks,
            transformer_layers=transformer_layers,
            nhead=nhead,
            mlp_dim=mlp_dim,
            use_cls_token=use_cls_token,
            enable_dino=False,  # 推理时不需要 projector
        ).to(device)

        # 加载权重：优先尝试 'student'，否则尝试原始 state_dict
        if "student" in ckpt:
            state_dict = ckpt["student"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # 处理可能的 DDP 保存（带 module. 前缀）
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        return model, embed_dim


def extract_features(encoder: DiffusionTactileEncoder, batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    tactile = batch["tactile"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    # Use t=0 to minimize noise variance in probing
    t = torch.zeros((tactile.size(0),), device=device, dtype=torch.long)
    with torch.no_grad():
        feats = encoder(tactile, t, condition_tokens=None, return_tokens=False)  # (B, E)
    return feats, labels


def main(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device(args.device)

    n_classes = get_num_classes(args.label)
    train_loader, test_loader = build_dataloaders(args)

    encoder, embed_dim = load_encoder(args, device)
    classifier = nn.Linear(embed_dim, n_classes).to(device)

    
    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Prepare class weights for loss if requested
    loss_weight = None
    if args.balance in ("loss", "both"):
        train_labels_for_weights = train_loader.dataset.train_labels
        if args.class_weights is not None:
            loss_weight = _parse_class_weights(n_classes, args.class_weights)
        else:
            loss_weight = _compute_class_weights(train_labels_for_weights, n_classes)
        loss_weight = loss_weight.to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weight)

    os.makedirs(args.out_dir, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # Train
        classifier.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_train = 0

        pbar = tqdm(train_loader, desc=f"Train ep{epoch}")
        for batch in pbar:
            feats, labels = extract_features(encoder, batch, device)
            logits = classifier(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            n_train += bs
            epoch_loss += loss.item() * bs
            epoch_acc += top1_accuracy(logits.detach(), labels) * bs

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(epoch_acc/n_train)*100:.2f}%")
            # logging.info(f"train process {n_train} has loss {loss.item():.4f}, accuracy {(epoch_acc/n_train)*100:.2f}")

        scheduler.step()
        train_loss = epoch_loss / max(n_train, 1)
        train_acc = epoch_acc / max(n_train, 1)
        # Eval
        classifier.eval()
        correct = 0
        total = 0
        # Per-class stats
        per_class_correct = torch.zeros(n_classes, dtype=torch.long)
        per_class_total = torch.zeros(n_classes, dtype=torch.long)
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Eval"):
                feats, labels = extract_features(encoder, batch, device)
                logits = classifier(feats)
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                for c in range(n_classes):
                    mask = (labels == c)
                    per_class_total[c] += mask.sum().item()
                    per_class_correct[c] += ((pred[mask] == c).sum().item())
        test_acc = correct / max(total, 1)

        per_class_acc = [(per_class_correct[c].item() / max(per_class_total[c].item(), 1)) for c in range(n_classes)]
        logging.info(f"Epoch {epoch:03d}/{args.epochs} | train_acc={train_acc*100:.2f}% | train_loss={train_loss:.4f} | test_acc={test_acc*100:.2f}% | per_class_acc={per_class_acc}")


        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                "epoch": epoch,
                "classifier": classifier.state_dict(),
                "embed_dim": embed_dim,
                "n_classes": n_classes,
                "args": vars(args),
            }
            save_path = os.path.join(args.out_dir, f"linear_probe_best_{args.label}.pt")
            torch.save(state, save_path)
            logging.info(f"Saved best probe to: {save_path}")


if __name__ == "__main__":
    parser = build_argparser()
    main(parser.parse_args())
