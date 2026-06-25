"""
Loss functions for gate segmentation.

MonoRace GateNet loss (from paper):
    Per head:  ℒᵢ = Dice(yᵢ, ŷᵢ) + 2·BCE(yᵢ, ŷᵢ)
    Total:     ℒtot = 4·ℒ₀ + 2·ℒ₁ + ℒ₂ + ℒ₃ + ℒ₄

Simple UNet loss:
    ℒ = BCE + Dice  (equal weight, one output)
"""

import torch
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss. pred must be sigmoid-activated probabilities in [0,1]."""
    intersection = (pred * target).sum()
    return 1.0 - (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)


def bce_dice_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ℒᵢ = Dice + 2·BCE  (MonoRace per-head formula)."""
    bce  = F.binary_cross_entropy_with_logits(pred_logits, target)
    pred = torch.sigmoid(pred_logits)
    return dice_loss(pred, target) + 2.0 * bce


def gatenet_total_loss(
    outputs: list[torch.Tensor],
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-scale GateNet loss.
    outputs = [y0, y1, y2, y3, y4] logits, each upsampled to target spatial size.
    target = (B, 1, H, W) binary float mask.
    """
    weights = [4.0, 2.0, 1.0, 1.0, 1.0]
    total = torch.zeros(1, device=target.device)
    for w, logits in zip(weights, outputs):
        # Upsample to target resolution if needed
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        total = total + w * bce_dice_loss(logits, target)
    return total


def simple_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + Dice, equal weight. For the single-output smp UNet."""
    bce  = F.binary_cross_entropy_with_logits(pred_logits, target)
    pred = torch.sigmoid(pred_logits)
    return bce + dice_loss(pred, target)


def compute_iou(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred_bin * target).sum().item()
    union = (pred_bin + target).clamp(0, 1).sum().item()
    return intersection / (union + 1e-6)


def compute_dice(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred_bin * target).sum().item()
    return (2.0 * intersection + 1e-6) / (pred_bin.sum().item() + target.sum().item() + 1e-6)
