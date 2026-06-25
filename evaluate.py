"""
Segmentation metrics: Dice coefficient and IoU.

    Dice = 2·TP / (2·TP + FP + FN)   ← primary metric; equivalent to F1 score
    IoU  =   TP / (    TP + FP + FN)  ← standard benchmark; stricter than Dice

Relationship: IoU = Dice / (2 − Dice)  — they carry the same information but
IoU penalizes errors more heavily (denominator is larger).

For gate detection, Dice is the primary metric because:
  - It emphasizes true positives (gate pixels correctly detected), which is
    what matters for downstream navigation.
  - It is more sensitive to small, sparse gate regions than pixel accuracy.

Both are computed per-sample then averaged over the batch (micro-averaged).
The ε (1e-8) handles the degenerate case where both prediction and target
are all-zero (empty mask) — in that case, Dice and IoU are defined as 1.0.
"""

import torch
from torch.utils.data import DataLoader

from model import UNet


@torch.inference_mode()
def evaluate(model: UNet, dataloader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict[str, float]:
    """
    Compute mean Dice and IoU over a dataloader.

    Args:
        model:     Trained UNet (output is raw logits).
        dataloader: Yields batches with keys "image" and "mask".
        device:    torch.device to run on.
        threshold: Sigmoid threshold for binarizing predictions (default 0.5).

    Returns:
        {"dice": float, "iou": float}  — values in [0, 1].
    """
    model.eval()
    total_dice = total_iou = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        logits = model(images)
        preds  = (torch.sigmoid(logits) > threshold).float()

        # Compute per-sample TP, FP, FN by summing over (C, H, W).
        tp = (preds * masks).sum(dim=(1, 2, 3))
        fp = (preds * (1 - masks)).sum(dim=(1, 2, 3))
        fn = ((1 - preds) * masks).sum(dim=(1, 2, 3))

        dice = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()
        iou  = (tp / (tp + fp + fn + 1e-8)).mean().item()

        total_dice += dice
        total_iou  += iou

    n = len(dataloader)
    return {"dice": total_dice / n, "iou": total_iou / n}
