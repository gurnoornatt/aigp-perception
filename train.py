"""
train.py — U-Net training loop for binary gate segmentation.

Usage:
    python train.py --data-dir data/gates --epochs 100 --amp

Expects:
    <data-dir>/images/   — RGB images
    <data-dir>/masks/    — binary masks (same stem as image)

Demonstrates:
    - Structured train/val split with separate augmentation pipelines
    - Combined Focal + Dice loss for sparse binary masks
    - Mixed-precision training (--amp) via torch.amp
    - AdamW optimizer with ReduceLROnPlateau scheduler
    - Best-model checkpointing on validation Dice
    - CSV training log for plotting
"""

import argparse
import csv
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler

from model import UNet, CombinedLoss
from dataset import GateDataset, get_train_transforms, get_val_transforms
from evaluate import evaluate

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train U-Net for gate segmentation")
    p.add_argument("--data-dir",       type=Path,  required=True,
                   help="Root dir containing images/ and masks/ subdirectories")
    p.add_argument("--checkpoint-dir", type=Path,  default=Path("checkpoints"),
                   help="Where to save best.pt and train_log.csv")
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=1e-4,
                   help="Initial learning rate for AdamW")
    p.add_argument("--img-size",       type=int,   default=512,
                   help="Square crop/resize resolution fed to the model")
    p.add_argument("--val-split",      type=float, default=0.2,
                   help="Fraction of data reserved for validation [0, 1)")
    p.add_argument("--base-filters",   type=int,   default=64,
                   help="Channel width at first encoder block. Use 32 for lightweight.")
    p.add_argument("--bilinear",       action="store_true",
                   help="Bilinear upsampling (fewer params, no checkerboard artifacts)")
    p.add_argument("--amp",            action="store_true",
                   help="Enable automatic mixed precision (fp16 on CUDA)")
    p.add_argument("--load",           type=Path,  default=None,
                   help="Path to checkpoint .pt file to resume from")
    p.add_argument("--num-workers",    type=int,   default=4)
    return p.parse_args()


# ── Data helpers ──────────────────────────────────────────────────────────────

def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """
    Create train and val DataLoaders with distinct augmentation pipelines.

    Why two separate GateDataset instances?
    random_split shares the underlying dataset object, so setting transform on
    one subset would overwrite it for the other. Two instances avoid this bug.
    """
    img_dir  = args.data_dir / "images"
    mask_dir = args.data_dir / "masks"

    train_ds = GateDataset(img_dir, mask_dir, transform=get_train_transforms(args.img_size))
    val_ds   = GateDataset(img_dir, mask_dir, transform=get_val_transforms(args.img_size))

    n = len(train_ds)
    n_val   = max(1, int(n * args.val_split))
    n_train = n - n_val

    # Fixed seed split — same indices every run for reproducible val set.
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_loader = DataLoader(
        Subset(train_ds, indices[n_val:]),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, indices[:n_val]),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    log.info(f"Dataset split — train: {n_train}  val: {n_val}")
    return train_loader, val_loader


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     UNet,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    scaler:    GradScaler,
    device:    torch.device,
) -> float:
    """Run one epoch; return mean loss over batches."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        optimizer.zero_grad()

        # autocast enables fp16 compute on CUDA when --amp is set;
        # GradScaler handles the loss scaling needed to avoid fp16 underflow.
        with torch.autocast(device.type, enabled=scaler.is_enabled()):
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | AMP: {args.amp}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(args)

    model = UNet(
        n_channels=3, n_classes=1,
        bilinear=args.bilinear,
        base_filters=args.base_filters,
    ).to(device)

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        log.info(f"Resumed from {args.load}")

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,}")

    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # ReduceLROnPlateau monitors val Dice; halves LR if no improvement for 10 epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5, min_lr=1e-6,
    )
    # GradScaler is a no-op when AMP is disabled (enabled=False).
    scaler = GradScaler(device.type, enabled=args.amp)

    log_path = args.checkpoint_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_dice", "val_iou", "lr"])

    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        metrics    = evaluate(model, val_loader, device)
        scheduler.step(metrics["dice"])

        lr = optimizer.param_groups[0]["lr"]
        log.info(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"loss={train_loss:.4f}  "
            f"dice={metrics['dice']:.4f}  "
            f"iou={metrics['iou']:.4f}  "
            f"lr={lr:.2e}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{metrics['dice']:.6f}", f"{metrics['iou']:.6f}", f"{lr:.2e}"])

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            ckpt_path = args.checkpoint_dir / "best.pt"
            torch.save(model.state_dict(), ckpt_path)
            log.info(f"  Saved best model  dice={best_dice:.4f}  →  {ckpt_path}")

    log.info(f"Done. Best validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
