"""
PyTorch Dataset for binary gate segmentation.

Expected directory layout:
    <root>/
        images/   — RGB images  (.png / .jpg / .jpeg)
        masks/    — Binary masks (same filename stem, any extension)
                    Pixel value 0 = background, any nonzero = gate.

The mask is loaded as a single-channel (L-mode) image and binarized to {0, 1}.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GateDataset(Dataset):
    """
    Args:
        img_dir:   Path to the images directory.
        mask_dir:  Path to the masks directory.
        transform: Optional albumentations Compose transform applied jointly to
                   the image (keyword "image") and mask (keyword "mask").
                   Albumentations is used because it guarantees identical
                   geometric transforms are applied to both image and mask.
    """

    def __init__(self, img_dir: str | Path, mask_dir: str | Path, transform=None):
        self.img_dir   = Path(img_dir)
        self.mask_dir  = Path(mask_dir)
        self.transform = transform

        self.ids = sorted(
            p.stem for p in self.img_dir.glob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not self.ids:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        stem = self.ids[idx]

        img_path  = next(self.img_dir.glob(f"{stem}.*"))
        mask_path = next(self.mask_dir.glob(f"{stem}*"))

        # Load as numpy arrays — albumentations requires HWC uint8.
        image = np.array(Image.open(img_path).convert("RGB"),  dtype=np.uint8)
        mask  = np.array(Image.open(mask_path).convert("L"),   dtype=np.uint8)

        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out["image"]
            mask  = out["mask"]

        # Convert to tensors.
        # Image: (H, W, 3) uint8 → (3, H, W) float32 in [0, 1]
        # Mask:  (H, W)    uint8 → (1, H, W) float32 in {0, 1}
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask  = torch.from_numpy(mask).unsqueeze(0).float().div(255).gt(0).float()

        return {"image": image, "mask": mask, "id": stem}
