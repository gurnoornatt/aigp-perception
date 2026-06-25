"""
Albumentations augmentation pipelines for gate segmentation training.

Why albumentations?
    It applies the same random transform to both image and mask, which is
    required for geometric augmentations — if you flip the image, the mask
    must flip identically.

Training augmentations address two challenges specific to FPV drone footage:

    Geometric variation — gates appear at arbitrary angles, distances, and
    aspect ratios depending on the drone's trajectory and camera pose.
    → HorizontalFlip, VerticalFlip, Rotate, RandomScale

    Photometric variation — lighting changes rapidly at speed; motion blur
    is common at high throttle; cameras differ between competition setups.
    → RandomBrightnessContrast, HueSaturationValue, GaussianBlur

Validation uses only a deterministic resize — no augmentation — so metrics
are comparable across runs and reflect real-world performance.
"""

import albumentations as A


def get_train_transforms(size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.4),
        # Photometric
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        # Regularization — randomly blanks small regions to prevent over-reliance
        # on any single visual cue (e.g., gate color alone).
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.2),
    ])


def get_val_transforms(size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
    ])
