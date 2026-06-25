"""
Loss functions for binary segmentation.

For gate detection, the foreground (gate pixels) is sparse — typically <5% of
the image. Standard BCE treats every pixel equally, so the model can cheat by
predicting all-background and still achieve low loss. The losses here address
that directly.

    DiceLoss    — optimizes the overlap ratio directly; robust to imbalance.
    FocalLoss   — down-weights easy negatives so gradients focus on hard pixels.
    CombinedLoss — Focal + Dice; recommended default for sparse binary masks.

All losses operate on raw logits (no sigmoid applied before calling).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Dice = 2·|A ∩ B| / (|A| + |B|)   →   Loss = 1 − Dice

    The ε (1e-8) in the denominator prevents division by zero on empty masks.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Sum over spatial dims (H, W) and channel; average over batch.
        tp    = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        return (1 - (2 * tp / (denom + 1e-8))).mean()


class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., "Focal Loss for Dense Object Detection", 2017).

    L_focal = −α · (1 − p_t)^γ · log(p_t)

    gamma=2 is the standard value from the paper; alpha=0.8 upweights the
    positive class (gate pixels), which are rare in drone footage.

    Args:
        alpha: Weight for the positive class. Higher → more weight on gates.
        gamma: Focusing exponent. Higher → steeper down-weighting of easy examples.
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)                          # predicted probability of ground-truth class
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    """
    Focal + Dice loss — recommended for sparse binary masks.

    Focal handles hard examples locally (pixel level).
    Dice optimizes the global overlap metric directly (object level).
    Together they are complementary: Focal prevents the model from ignoring
    rare gate pixels; Dice ensures the predicted region actually aligns.

    Args:
        focal_weight: Scalar multiplier for the Focal term (default 1.0).
        dice_weight:  Scalar multiplier for the Dice term (default 1.0).
        alpha, gamma: Passed through to FocalLoss.
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight:  float = 1.0,
        alpha:        float = 0.8,
        gamma:        float = 2.0,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice  = DiceLoss()
        self.fw    = focal_weight
        self.dw    = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.fw * self.focal(logits, targets) + self.dw * self.dice(logits, targets)
