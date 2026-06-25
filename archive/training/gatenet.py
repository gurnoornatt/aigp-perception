"""
MonoRace GateNet — faithful reimplementation from arXiv:2601.15222.

Architecture (f=4):
    Encoder: inc(16) → down1(32) → down2(64) → down3(128) → down4(128)
    Decoder: up1(64) → up2(32) → up3(16) → up4(16)
    5 output heads, one per stage including bottleneck.

Key details from paper:
    - Double 3×3 conv + BN + ReLU per block
    - Transposed conv for upsampling (not bilinear)
    - Xavier uniform weight init
    - Only y4 (final full-res output) used at inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _xavier_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """TransposedConv upsample → concat skip → DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.bn   = nn.BatchNorm2d(in_ch)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.up(x)), inplace=True)
        # Pad if spatial sizes differ (shouldn't happen at 384, but safe)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]])
        return self.conv(torch.cat([x, skip], dim=1))


class OutHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # logits; sigmoid applied in loss / inference


class GateNet(nn.Module):
    """
    MonoRace GateNet with f=4.

    Forward returns a list of 5 logit tensors [y0, y1, y2, y3, y4].
    y0 is at bottleneck resolution, y4 is at full input resolution.
    During inference use y4 only.
    """

    def __init__(self, f: int = 4):
        super().__init__()
        c = [64 // f, 128 // f, 256 // f, 512 // f, 512 // f]  # [16,32,64,128,128]

        # Encoder
        self.inc   = DoubleConv(3, c[0])
        self.down1 = Down(c[0], c[1])
        self.down2 = Down(c[1], c[2])
        self.down3 = Down(c[2], c[3])
        self.down4 = Down(c[3], c[4])

        # Decoder (up1 takes bottleneck + skip from down3)
        self.up1 = Up(c[4], c[3], 256 // f)   # 128+128 → 64
        self.up2 = Up(256 // f, c[2], 128 // f)  # 64+64 → 32
        self.up3 = Up(128 // f, c[1], 64 // f)   # 32+32 → 16
        self.up4 = Up(64 // f, c[0], 64 // f)    # 16+16 → 16

        # Output heads (one per stage + bottleneck)
        self.outc0 = OutHead(c[4])        # bottleneck → y0
        self.outc1 = OutHead(256 // f)    # up1        → y1
        self.outc2 = OutHead(128 // f)    # up2        → y2
        self.outc3 = OutHead(64 // f)     # up3        → y3
        self.outc4 = OutHead(64 // f)     # up4        → y4 (final)

        _xavier_init(self)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        H, W = x.shape[2], x.shape[3]

        # Encoder
        s0 = self.inc(x)     # (B,16,H,W)
        s1 = self.down1(s0)  # (B,32,H/2,W/2)
        s2 = self.down2(s1)  # (B,64,H/4,W/4)
        s3 = self.down3(s2)  # (B,128,H/8,W/8)
        s4 = self.down4(s3)  # (B,128,H/16,W/16) — bottleneck

        # Decoder
        d1 = self.up1(s4, s3)  # (B,64,H/8,W/8)
        d2 = self.up2(d1, s2)  # (B,32,H/4,W/4)
        d3 = self.up3(d2, s1)  # (B,16,H/2,W/2)
        d4 = self.up4(d3, s0)  # (B,16,H,W)

        # Output heads (all upsampled to input resolution for loss)
        def _up(t):
            return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)

        y0 = _up(self.outc0(s4))
        y1 = _up(self.outc1(d1))
        y2 = _up(self.outc2(d2))
        y3 = _up(self.outc3(d3))
        y4 = self.outc4(d4)       # already full resolution

        return [y0, y1, y2, y3, y4]

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Inference-only: returns binary mask from y4."""
        with torch.no_grad():
            outputs = self.forward(x)
            return (torch.sigmoid(outputs[-1]) > threshold).float()


if __name__ == "__main__":
    model = GateNet(f=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GateNet parameters: {n_params:,}")

    x = torch.randn(2, 3, 384, 384)
    outputs = model(x)
    print(f"Output shapes: {[o.shape for o in outputs]}")
    # Expected: all (2, 1, 384, 384)
