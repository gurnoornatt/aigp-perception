"""
Batch augmentation pipeline.

Takes the 150 base (image, mask) pairs from data/raw/ and generates
20 augmented variants of each → ~3000 training pairs in data/augmented/.

Albumentations applies spatial transforms identically to image + mask,
and pixel transforms to image only — masks stay binary and aligned.

Run:
    python scripts/augment_dataset.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A

# ── Directories ───────────────────────────────────────────────────────────────
RAW_IMG_DIR  = Path("data/raw/images")
RAW_MASK_DIR = Path("data/raw/masks")
AUG_IMG_DIR  = Path("data/augmented/images")
AUG_MASK_DIR = Path("data/augmented/masks")
AUG_IMG_DIR.mkdir(parents=True, exist_ok=True)
AUG_MASK_DIR.mkdir(parents=True, exist_ok=True)

AUGMENTATIONS_PER_BASE = 20
IMG_H, IMG_W = 360, 640
BG_COLOR = (15, 20, 30)   # used as border fill for spatial transforms

# ── Augmentation pipeline ─────────────────────────────────────────────────────
TRANSFORM = A.Compose([
    # Spatial — applied to BOTH image and mask identically
    A.RandomResizedCrop(
        size=(IMG_H, IMG_W),
        scale=(0.75, 1.0),
        ratio=(0.9, 1.1),
        p=0.5,
    ),
    A.Affine(
        translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
        scale=(0.80, 1.20),
        rotate=(-25, 25),
        border_mode=cv2.BORDER_CONSTANT,
        fill=BG_COLOR,
        p=0.85,
    ),
    A.HorizontalFlip(p=0.5),
    A.Perspective(scale=(0.03, 0.09), p=0.50),

    # Pixel — applied to image only (mask is untouched by these)
    A.GaussianBlur(blur_limit=(3, 9), p=0.60),
    A.GaussNoise(std_range=(0.04, 0.24), p=0.60),
    A.RandomBrightnessContrast(brightness_limit=0.30, contrast_limit=0.30, p=0.80),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=20,
        p=0.40,
    ),
    A.ImageCompression(quality_range=(60, 95), p=0.30),
])


def augment_all() -> None:
    raw_images = sorted(RAW_IMG_DIR.glob("*_img.png"))
    if not raw_images:
        print(f"No images found in {RAW_IMG_DIR}. Run generate_synthetic.py first.")
        return

    total = len(raw_images) * AUGMENTATIONS_PER_BASE
    print(f"Augmenting {len(raw_images)} base images × {AUGMENTATIONS_PER_BASE} = {total} pairs...")

    with tqdm(total=total) as pbar:
        for img_path in raw_images:
            base_idx = int(img_path.stem.split("_")[0])
            mask_path = RAW_MASK_DIR / f"{base_idx:03d}_mask.png"

            image = cv2.imread(str(img_path))
            mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"Warning: could not load {img_path} or {mask_path}, skipping.")
                continue

            for aug_idx in range(AUGMENTATIONS_PER_BASE):
                result    = TRANSFORM(image=image, mask=mask)
                aug_image = result["image"]
                aug_mask  = result["mask"]

                # Ensure mask stays binary after spatial transforms
                aug_mask = (aug_mask > 127).astype(np.uint8) * 255

                out_stem = f"{base_idx:03d}_{aug_idx:02d}"
                cv2.imwrite(str(AUG_IMG_DIR  / f"{out_stem}_img.png"),  aug_image)
                cv2.imwrite(str(AUG_MASK_DIR / f"{out_stem}_mask.png"), aug_mask)
                pbar.update(1)

    n_imgs  = len(list(AUG_IMG_DIR.glob("*.png")))
    n_masks = len(list(AUG_MASK_DIR.glob("*.png")))
    print(f"\nDone. Saved to:")
    print(f"  {AUG_IMG_DIR}  ({n_imgs} images)")
    print(f"  {AUG_MASK_DIR} ({n_masks} masks)")


if __name__ == "__main__":
    augment_all()
