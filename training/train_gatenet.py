"""
GateNet training script — MonoRace architecture.

Hyperparameters match the paper exactly:
    AdamW, lr=1e-3, ×√0.1 at epochs 10/33/66/90, batch=16, 100 epochs.

Usage:
    python training/train_gatenet.py
    python training/train_gatenet.py --epochs 5   # quick smoke test
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2

from training.dataset import build_datasets
from training.gatenet import GateNet
from training.losses import gatenet_total_loss, compute_iou, compute_dice

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_DIR = Path("checkpoints")
LOG_DIR  = Path("logs")
VIS_DIR  = LOG_DIR / "vis" / "gatenet"
CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


# ── LR schedule (paper: ×√0.1 at epochs 10, 33, 66, 90) ─────────────────────
def make_scheduler(optimizer: torch.optim.Optimizer, milestones: list[int]) -> torch.optim.lr_scheduler.MultiStepLR:
    gamma = math.sqrt(0.1)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


# ── Visualisation grid ────────────────────────────────────────────────────────
def save_vis_grid(model: GateNet, val_loader: DataLoader, epoch: int, n: int = 4) -> None:
    model.eval()
    imgs, preds, gts = [], [], []
    with torch.no_grad():
        for img_batch, mask_batch in val_loader:
            img_batch = img_batch.to(DEVICE)
            outputs   = model(img_batch)
            pred_mask = torch.sigmoid(outputs[-1]).cpu()
            for i in range(min(n - len(imgs), img_batch.size(0))):
                imgs.append(img_batch[i].cpu())
                preds.append(pred_mask[i])
                gts.append(mask_batch[i])
            if len(imgs) >= n:
                break

    rows = []
    for img_t, pred_t, gt_t in zip(imgs, preds, gts):
        img_np  = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        pred_np = (pred_t.squeeze().numpy() * 255).astype(np.uint8)
        pred_3  = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)
        gt_np   = (gt_t.squeeze().numpy() * 255).astype(np.uint8)
        gt_3    = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2BGR)
        rows.append(np.hstack([img_bgr, pred_3, gt_3]))

    grid = np.vstack(rows)
    out_path = VIS_DIR / f"epoch_{epoch:03d}.png"
    cv2.imwrite(str(out_path), grid)
    model.train()


# ── One epoch ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = gatenet_total_loss(outputs, masks)
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
            outputs = model(imgs)
            loss    = gatenet_total_loss(outputs, masks)
            total_loss += loss.item()
            total_iou  += compute_iou(outputs[-1], masks)
            total_dice += compute_dice(outputs[-1], masks)
    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


# ── Main ──────────────────────────────────────────────────────────────────────
def main(epochs: int = 100, batch_size: int = 16) -> None:
    print(f"Device: {DEVICE}")

    # Data
    train_ds, val_ds, _ = build_datasets(normalize_imagenet=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model     = GateNet(f=4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = make_scheduler(optimizer, milestones=[10, 33, 66, 90])

    # CSV log
    log_path = LOG_DIR / "gatenet_train.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "lr"])

    best_iou = 0.0
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_iou, val_dice = val_epoch(model, val_loader)
        current_lr = scheduler.get_last_lr()[0] if epoch > 0 else 1e-3
        scheduler.step()

        print(f"[{epoch+1:03d}/{epochs}] train={train_loss:.4f} val={val_loss:.4f} "
              f"IoU={val_iou:.4f} Dice={val_dice:.4f} lr={current_lr:.2e}")

        # Log
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{train_loss:.5f}", f"{val_loss:.5f}",
                                    f"{val_iou:.5f}", f"{val_dice:.5f}", f"{current_lr:.2e}"])

        # Save best
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"epoch": epoch + 1, "state_dict": model.state_dict(),
                        "val_iou": val_iou}, CKPT_DIR / "gatenet_best.pt")

        # Visualise every 10 epochs (and epoch 0)
        if epoch % 10 == 0:
            save_vis_grid(model, val_loader, epoch)

    print(f"\nDone. Best val IoU: {best_iou:.4f}")
    print(f"Checkpoint: {CKPT_DIR}/gatenet_best.pt")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
