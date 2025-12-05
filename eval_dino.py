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


from dinoTac import TactileVisionTower  # 请根据实际路径调整
from dataloader.tag_dataset import TouchAndGoPairDataset

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_name = f'logs/eval_dino_{timestamp}.log'
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename=log_name, level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s')

def collate_keep_tactile(batch):
    """只保留 tactile 和 label，用于评估"""
    tactile = torch.stack([b["tactile"] for b in batch], dim=0)
    label = torch.as_tensor([b["label"] for b in batch], dtype=torch.long)
    return {"tactile": tactile, "label": label}


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    print(targets)
    print(pred)
    return (pred == targets).float().mean().item()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Linear probe on DINO Tactile Encoder features")
    # Data
    p.add_argument("--dataroot", type=str, required=True, help="Root folder of Touch-and-Go dataset")
    p.add_argument("--list_dir", type=str, required=True, help="Directory containing train.txt/test.txt")
    p.add_argument("--label", type=str, default="full", choices=["full", "rough", "hard"], help="Label scheme")
    p.add_argument("--image_size", type=int, default=64, help="Tactile image size (HxW)")
    # Model
    p.add_argument("--checkpoint", type=str, required=True, help="Path to DINO encoder checkpoint (*.pt)")
    p.add_argument("--use_cls_token", action="store_true", help="Use cls token (if saved model used it)")
    # Probe
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=8)
    # System
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="probes_dino")
    return p


def get_num_classes(label_scheme: str) -> int:
    if label_scheme == "full":
        return 20
    if label_scheme in ("rough", "hard"):
        return 2
    raise ValueError(f"Unknown label scheme: {label_scheme}")


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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_keep_tactile,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_keep_tactile,
    )

    return train_loader, test_loader


def load_dino_encoder(args: argparse.Namespace, device: torch.device) -> Tuple[TactileVisionTower, int]:
    ckpt = torch.load(args.checkpoint, map_location=device)

    # 尝试从 checkpoint 读取配置，否则用默认值
    if "args" in ckpt:
        cfg = ckpt["args"]
    elif "config" in ckpt:
        cfg = ckpt["config"]
    else:
        cfg = {}
    print(cfg)

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


def extract_features(encoder: TactileVisionTower, batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    tactile = batch["tactile"].to(device, non_blocking=True)
    # print(tactile)
    labels = batch["label"].to(device, non_blocking=True)
    # print(labels)

    with torch.no_grad():
        # 提取 per-patch features: (B, N, C)
        patch_features = encoder(tactile)  # calls forward() → forward_feature()

        # 全局 pooling
        if encoder.use_cls_token:
            # 如果你保存时用了 cls token，但 forward_feature 返回的是 patch tokens（不含 cls），
            # 则需要特殊处理。为简化，我们统一用 mean pooling。
            # 你可以根据实际模型行为调整。
            global_feat = patch_features.mean(dim=1)  # (B, C)
        else:
            global_feat = patch_features.mean(dim=1)  # (B, C)
        

    return global_feat, labels


def main(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device(args.device)

    n_classes = get_num_classes(args.label)
    train_loader, test_loader = build_dataloaders(args)

    encoder, embed_dim = load_dino_encoder(args, device)
    classifier = nn.Linear(embed_dim, n_classes).to(device)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

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

        scheduler.step()
        train_loss = epoch_loss / max(n_train, 1)
        train_acc = epoch_acc / max(n_train, 1)

        # Eval
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Eval"):
                feats, labels = extract_features(encoder, batch, device)
                logits = classifier(feats)
                pred = logits.argmax(dim=1)
                print(labels)
                print(pred)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / max(total, 1)

        logging.info(f"Epoch {epoch:03d}/{args.epochs} | "
                     f"train_acc={train_acc*100:.2f}% | train_loss={train_loss:.4f} | "
                     f"test_acc={test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                "epoch": epoch,
                "classifier": classifier.state_dict(),
                "embed_dim": embed_dim,
                "n_classes": n_classes,
                "args": vars(args),
            }
            save_path = os.path.join(args.out_dir, f"dino_linear_probe_best_{args.label}.pt")
            torch.save(state, save_path)
            logging.info(f"Saved best probe to: {save_path}")

    logging.info(f"Final best test accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)