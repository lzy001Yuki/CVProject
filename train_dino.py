import argparse
import os
import time
import math
from typing import Optional
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from dinoTac import TactileVisionTower
from dataloader.tag_dataset import TouchAndGoPairDataset  # 你需要根据实际路径调整

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_name = f'logs/dino_tactile_{timestamp}.log'
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename=log_name,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# ----------------------------
# 数据增强（Tactile-specific）
# ----------------------------
def get_tactile_augmentations(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ], p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

# ----------------------------
# DINO 损失函数
# ----------------------------
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = torch.linspace(
            warmup_teacher_temp, teacher_temp, nepochs - warmup_teacher_temp_epochs
        )

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = [s / self.student_temp for s in student_output]
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch] if epoch < len(self.teacher_temp_schedule) else self.teacher_temp_schedule[-1]
        teacher_out = [F.softmax((t - self.center) / temp, dim=-1).detach() for t in teacher_output]
        # teacher_out = teacher_out.detach()
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                # if v == iq:
                #     continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        # print(n_loss_terms)
        total_loss /= n_loss_terms

        # update center
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# ----------------------------
# 参数解析
# ----------------------------
def build_argparser():
    p = argparse.ArgumentParser("Train Tactile Encoder with DINO-style SSL")
    # Data
    p.add_argument("--dataroot", type=str, required=True)
    p.add_argument("--list_dir", type=str, default="Visuo-tactile contrastive learning/dataset")
    p.add_argument("--split", type=str, default="pretrain", choices=["train", "pretrain"])
    p.add_argument("--image_size", type=int, default=64)
    # Model
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--local_blocks", type=int, default=2)
    p.add_argument("--transformer_layers", type=int, default=4)
    p.add_argument("--nhead", type=int, default=6)
    p.add_argument("--mlp_dim", type=int, default=1024)
    p.add_argument("--dino_out_dim", type=int, default=256)
    p.add_argument("--use_cls_token", action="store_true")
    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--weight_decay_end", type=float, default=0.4)
    p.add_argument("--clip_grad", type=float, default=3.0)
    # DINO specific
    p.add_argument("--teacher_temp", type=float, default=0.04)
    p.add_argument("--warmup_teacher_temp", type=float, default=0.04)
    p.add_argument("--warmup_teacher_temp_epochs", type=int, default=30)
    p.add_argument("--student_temp", type=float, default=0.1)
    p.add_argument("--center_momentum", type=float, default=0.9)
    p.add_argument("--local_crops_number", type=int, default=4)  # 可选：局部裁剪（这里简化为只有全局增强）
    # System
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="checkpoints_dino")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # Distributed & AMP
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--amp", action="store_true")
    return p


def setup_distributed(args):
    if not args.distributed:
        args.rank = 0
        args.world_size = 1
        return torch.device(args.device)
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return torch.device("cuda", args.local_rank)


def is_main_process(args):
    return getattr(args, "rank", 0) == 0


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = []
    if warmup_epochs > 0:
        warmup_schedule = list(torch.linspace(start_warmup_value, base_value, warmup_epochs * niter_per_ep))
    iters = epochs * niter_per_ep
    schedule = [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / iters))
                for i in range(warmup_epochs * niter_per_ep, iters)]
    return warmup_schedule + schedule


def cancel_gradients_last_layer(epoch, model, freeze_last_layer_epochs):
    if epoch >= freeze_last_layer_epochs:
        return
    for n, p in model.named_parameters():
        if "dino_global_proj" in n or "dino_patch_proj" in n:
            p.grad = None


# ----------------------------
# 主训练函数
# ----------------------------
def main(args):
    torch.manual_seed(args.seed)
    device = setup_distributed(args)
    logging.info(f"Using device {device}")

    # ========================
    # 数据加载
    # ========================
    aug_global1 = get_tactile_augmentations(args.image_size)
    aug_global2 = get_tactile_augmentations(args.image_size)

    dataset = TouchAndGoPairDataset(
        list_dir=args.list_dir,
        dataroot=args.dataroot,
        split=args.split,
        img_transform=None,  # 不使用 vision
        gel_transform=None,  # 我们自己处理 tactile 增强
        label="full",
    )

    # 自定义 collate：对 tactile 做两次增强
    def collate_dino(batch):
        # print("DEBUG: Keys in a data sample:", list(batch[0].keys()))
        tactile_raw = [b["tactile"] for b in batch]  # 假设原始 PIL 或 tensor 在 "tactile_raw"
        # 如果你的 dataset 返回的是 tensor，可跳过 ToPIL；否则需确保支持 transform
        views = []
        for t in tactile_raw:
            views.append(aug_global1(t))
            views.append(aug_global2(t))
        # shape: [2B, C, H, W]
        return torch.stack(views, dim=0)

    if args.distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=True)
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
        collate_fn=collate_dino,
    )

    # ========================
    # 模型构建
    # ========================
    student = TactileVisionTower(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        local_blocks=args.local_blocks,
        transformer_layers=args.transformer_layers,
        nhead=args.nhead,
        mlp_dim=args.mlp_dim,
        use_cls_token=args.use_cls_token,
        enable_dino=True,
        dino_out_dim=args.dino_out_dim,
    ).to(device)

    teacher = TactileVisionTower(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        local_blocks=args.local_blocks,
        transformer_layers=args.transformer_layers,
        nhead=args.nhead,
        mlp_dim=args.mlp_dim,
        use_cls_token=args.use_cls_token,
        enable_dino=True,
        dino_out_dim=args.dino_out_dim,
    ).to(device)

    # 没有梯度给 teacher
    for p in teacher.parameters():
        p.requires_grad = False

    if args.distributed:
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[args.local_rank])
        # teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.local_rank])

    # DINO 损失
    dino_loss = DINOLoss(
        out_dim=args.dino_out_dim,
        ncrops=2,  # 两个全局视图
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
    ).to(device)

    # ========================
    # 优化器 & 调度器
    # ========================
    params_groups = [
        {'params': [p for n, p in student.named_parameters() if 'dino' not in n and p.requires_grad], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in student.named_parameters() if 'dino' in n and p.requires_grad], 'weight_decay': 0.0},
    ]

    optimizer = torch.optim.AdamW(params_groups, lr=args.lr)
    niter_per_ep = len(loader)
    lr_schedule = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, niter_per_ep, warmup_epochs=args.warmup_epochs
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, niter_per_ep
    )

    # 动量调度 (teacher 更新)
    momentum_schedule = cosine_scheduler(0.996, 0.9995, args.epochs, niter_per_ep)

    scaler = GradScaler(enabled=args.amp)
    os.makedirs(args.out_dir, exist_ok=True)

    # ========================
    # 训练循环
    # ========================
    for epoch in range(args.epochs):
        if args.distributed and sampler is not None:
            sampler.set_epoch(epoch)

        student.train()
        total_loss = 0.0
        n_batches = 0
        start_time = time.time()

        for it, views in enumerate(tqdm(loader, disable=not is_main_process(args))):
            it_total = len(loader) * epoch + it
            views = views.to(device, non_blocking=True)  # [2B, C, H, W]

            with autocast(enabled=args.amp):
                # 前向传播：学生
                student_global, student_patches = student.module.forward_for_dino(views) if args.distributed else student.forward_for_dino(views)
                # 教师（无梯度）
                with torch.no_grad():
                    teacher_global, teacher_patches = teacher.forward_for_dino(views) if args.distributed else teacher.forward_for_dino(views)

                # DINO loss：仅用全局 token（简化版）
                student_output = [student_global]
                teacher_output = [teacher_global]

                loss = dino_loss(student_output, teacher_output, epoch)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # 可选：冻结最后层（初期稳定训练）
            if args.distributed:
                cancel_gradients_last_layer(epoch, student.module, 0)
            else:
                cancel_gradients_last_layer(epoch, student, 0)

            # Clip gradient
            if args.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)

            scaler.step(optimizer)
            scaler.update()

            # 更新 teacher（动量）
            with torch.no_grad():
                m = momentum_schedule[it_total]
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # 更新 lr & wd
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it_total]
                if i == 0:  # main params
                    param_group["weight_decay"] = wd_schedule[it_total]

            total_loss += loss.item()
            n_batches += 1
            logging.info(f"Batch {n_batches} | Loss: {total_loss / n_batches:.6f}")

        avg_loss = total_loss / n_batches
        dt = time.time() - start_time
        if is_main_process(args):
            logging.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} | Time: {dt:.1f}s")

            if (epoch + 1) % args.save_every == 0:
                model_to_save = student.module if hasattr(student, "module") else student
                ckpt = {
                    "student": model_to_save.state_dict(),
                    "teacher": teacher.module.state_dict() if hasattr(teacher, "module") else teacher.state_dict(),
                    "epoch": epoch,
                    "args": vars(args)
                }
                path = os.path.join(args.out_dir, f"dino_tactile_e{epoch+1:03d}.pt")
                torch.save(ckpt, path)
                logging.info(f"Saved checkpoint: {path}")


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)