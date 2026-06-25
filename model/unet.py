"""
U-Net for binary semantic segmentation.

Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation" (arXiv:1505.04597).
Adapted from github.com/milesial/Pytorch-UNet.

Architecture (default bilinear=True, base_filters=64):

    Input  (B, 3, H, W)
      │
      ├─ inc    DoubleConv(3   → 64)    → x1  (B, 64,  H,    W   )
      │
      ├─ down1  MaxPool + DoubleConv    → x2  (B, 128,  H/2,  W/2 )
      ├─ down2  MaxPool + DoubleConv    → x3  (B, 256,  H/4,  W/4 )
      ├─ down3  MaxPool + DoubleConv    → x4  (B, 512,  H/8,  W/8 )
      ├─ down4  MaxPool + DoubleConv    → x5  (B, 512,  H/16, W/16)  ← bottleneck
      │
      ├─ up1   Upsample + cat(x4) + DoubleConv  → (B, 256, H/8,  W/8 )
      ├─ up2   Upsample + cat(x3) + DoubleConv  → (B, 128, H/4,  W/4 )
      ├─ up3   Upsample + cat(x2) + DoubleConv  → (B, 64,  H/2,  W/2 )
      ├─ up4   Upsample + cat(x1) + DoubleConv  → (B, 64,  H,    W   )
      │
      └─ outc  Conv1×1(64 → n_classes)           → (B, 1,  H,    W   )  ← raw logits

The bilinear path uses factor=2 to halve channels at each Up block so that
after concatenation the total channels match the non-bilinear path.
"""

import torch
import torch.nn as nn

from .parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    """
    U-Net segmentation model.

    Args:
        n_channels:   Input image channels (3 for RGB).
        n_classes:    Output channels — 1 for binary segmentation.
        bilinear:     True → bilinear upsampling (no learnable params, no artifacts).
                      False → ConvTranspose2d (learnable, ~2× more parameters).
        base_filters: Channel width at the first encoder block (default 64, as in
                      the original paper). Halve to ~32 for a lightweight variant
                      suitable for embedded deployment.
    """

    def __init__(
        self,
        n_channels:   int  = 3,
        n_classes:    int  = 1,
        bilinear:     bool = True,
        base_filters: int  = 64,
    ):
        super().__init__()
        self.n_channels   = n_channels
        self.n_classes    = n_classes
        self.bilinear     = bilinear
        self.base_filters = base_filters

        f = base_filters
        # When bilinear, the Up block handles channel halving internally via
        # mid_channels, so the bottleneck stays at 8f instead of 16f.
        factor = 2 if bilinear else 1

        # Encoder
        self.inc   = DoubleConv(n_channels, f)
        self.down1 = Down(f,      2 * f)
        self.down2 = Down(2 * f,  4 * f)
        self.down3 = Down(4 * f,  8 * f)
        self.down4 = Down(8 * f,  16 * f // factor)

        # Decoder (each Up block receives upsampled + skip-connection tensors)
        self.up1  = Up(16 * f,  8 * f // factor, bilinear)
        self.up2  = Up(8 * f,   4 * f // factor, bilinear)
        self.up3  = Up(4 * f,   2 * f // factor, bilinear)
        self.up4  = Up(2 * f,   f,               bilinear)

        # Output head: 1×1 conv → raw logits (sigmoid applied in loss / predict)
        self.outc = OutConv(f, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder — each skip saved for the decoder
        x1 = self.inc(x)     # full resolution
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # bottleneck

        # Decoder — skip connections restore fine-grained spatial detail
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        return self.outc(x)  # (B, n_classes, H, W) — raw logits

    def use_checkpointing(self) -> None:
        """
        Wrap encoder/decoder blocks with gradient checkpointing.

        Trades compute for memory: activations are not stored during the forward
        pass and are recomputed during backprop. Useful when training at high
        resolution or with large base_filters on a memory-constrained GPU.
        """
        from torch.utils.checkpoint import checkpoint_sequential
        self.inc   = torch.utils.checkpoint.checkpoint_wrapper(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint_wrapper(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint_wrapper(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint_wrapper(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint_wrapper(self.down4)


if __name__ == "__main__":
    for bf in [32, 64]:
        model = UNet(n_channels=3, n_classes=1, bilinear=True, base_filters=bf)
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.randn(1, 3, 512, 512)
        out = model(x)
        print(f"base_filters={bf:3d} | params={n_params:,} | output={tuple(out.shape)}")
