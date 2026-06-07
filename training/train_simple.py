"""
Simple UNet baseline — segmentation_models_pytorch with ResNet-18 encoder.

Pretrained ImageNet weights on the encoder give a big head start when
training on synthetic-only data. Good comparison against GateNet to see
if pretrained features help on this specific task.

Usage:
    python training/train_simple.py
    python training/train_simple.py --epochs 5   # quick smoke test
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import cv2

from training.dataset import build_datasets
from training.losses import simple_loss, compute_iou, compute_dice

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_DIR = Path("checkpoints")
LOG_DIR  = Path("logs")
VIS_DIR  = LOG_DIR / "vis" / "simple"
CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


# ── Visualisation grid ────────────────────────────────────────────────────────
def save_vis_grid(model, val_loader, epoch: int, n: int = 4) -> None:
    model.eval()
    imgs, preds, gts = [], [], []
    with torch.no_grad():
        for img_batch, mask_batch in val_loader:
            img_batch = img_batch.to(DEVICE)
            pred_logits = model(img_batch).cpu()
            for i in range(min(n - len(imgs), img_batch.size(0))):
                imgs.append(img_batch[i].cpu())
                preds.append(pred_logits[i])
                gts.append(mask_batch[i])
            if len(imgs) >= n:
                break

    rows = []
    for img_t, pred_t, gt_t in zip(imgs, preds, gts):
        img_np  = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        pred_np = (torch.sigmoid(pred_t).squeeze().numpy() * 255).astype(np.uint8)
        pred_3  = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)
        gt_np   = (gt_t.squeeze().numpy() * 255).astype(np.uint8)
        gt_3    = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2BGR)
        rows.append(np.hstack([img_bgr, pred_3, gt_3]))

    grid = np.vstack(rows)
    cv2.imwrite(str(VIS_DIR / f"epoch_{epoch:03d}.png"), grid)
    model.train()


# ── One epoch ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = simple_loss(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(model, loader):
    model.eval()
    total_loss, total_iou, total_dice = 0.0, 0.0, 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            total_loss += simple_loss(logits, masks).item()
            total_iou  += compute_iou(logits, masks)
            total_dice += compute_dice(logits, masks)
    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


# ── Main ──────────────────────────────────────────────────────────────────────
def main(
    epochs: int = 100,
    batch_size: int = 16,
    dataset: str = "sim",
    from_scratch: bool = False,
) -> None:
    print(f"Device: {DEVICE}")

    encoder_weights = None if from_scratch else "imagenet"
    normalize = not from_scratch
    ckpt_name = "simple_sim_scratch.pt" if from_scratch and dataset == "sim" else "simple_best.pt"

    train_ds, val_ds, _, split = build_datasets(normalize_imagenet=normalize, dataset=dataset)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Dataset: {dataset} | Train: {len(train_ds)} | Val: {len(val_ds)} | "
          f"Encoder: {'scratch' if from_scratch else 'imagenet'} | Split: {split}")

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,   # raw logits; sigmoid in loss
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Simple UNet (resnet18) parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10, factor=0.3)

    log_path = LOG_DIR / "simple_train.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "lr"])

    best_iou = 0.0
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_iou, val_dice = val_epoch(model, val_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_iou)

        print(f"[{epoch+1:03d}/{epochs}] train={train_loss:.4f} val={val_loss:.4f} "
              f"IoU={val_iou:.4f} Dice={val_dice:.4f} lr={current_lr:.2e}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{train_loss:.5f}", f"{val_loss:.5f}",
                                    f"{val_iou:.5f}", f"{val_dice:.5f}", f"{current_lr:.2e}"])

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"epoch": epoch + 1, "state_dict": model.state_dict(),
                        "val_iou": val_iou, "dataset": dataset,
                        "from_scratch": from_scratch}, CKPT_DIR / ckpt_name)

        if epoch % 10 == 0:
            save_vis_grid(model, val_loader, epoch)

    print(f"\nDone. Best val IoU: {best_iou:.4f}")
    print(f"Checkpoint: {CKPT_DIR}/{ckpt_name}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset", choices=["sim", "synthetic"], default="sim")
    parser.add_argument("--from-scratch", action="store_true", help="Random init, no ImageNet weights")
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, dataset=args.dataset,
         from_scratch=args.from_scratch)
