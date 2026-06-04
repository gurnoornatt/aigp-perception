"""
GateDataset — loads synthetic gate image+mask pairs for training.

Split strategy: 120 train bases / 15 val bases / 15 test bases (seed=42).
Train set uses all 20 augmented variants per base (2400 pairs).
Val and test use raw images only (15 pairs each) → no data leakage.

Images resize to 384×384 (MonoRace's input resolution).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


RAW_IMG_DIR  = Path("data/raw/images")
RAW_MASK_DIR = Path("data/raw/masks")
AUG_IMG_DIR  = Path("data/augmented/images")
AUG_MASK_DIR = Path("data/augmented/masks")

INPUT_SIZE = 384   # MonoRace uses 384×384

# ImageNet stats (for pretrained encoder normalisation)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_mask(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


class GateDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        normalize_imagenet: bool = False,
    ):
        self.pairs = pairs
        self.normalize_imagenet = normalize_imagenet

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        img  = _load_rgb(img_path)
        mask = _load_mask(mask_path)

        # Resize to 384×384
        img  = cv2.resize(img,  (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)

        # Image → float32 tensor (3, H, W) in [0, 1]
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.normalize_imagenet:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            img_t = (img_t - mean) / std

        # Mask → float32 tensor (1, H, W) binary in {0, 1}
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        return img_t, mask_t


def get_split_indices(n_bases: int = 150, seed: int = 42):
    rng   = np.random.default_rng(seed)
    idxs  = rng.permutation(n_bases)
    n_val  = 15
    n_test = 15
    test_idxs  = set(idxs[:n_test].tolist())
    val_idxs   = set(idxs[n_test:n_test + n_val].tolist())
    train_idxs = set(idxs[n_test + n_val:].tolist())
    return train_idxs, val_idxs, test_idxs


def build_datasets(normalize_imagenet: bool = False):
    train_idxs, val_idxs, test_idxs = get_split_indices()

    train_pairs, val_pairs, test_pairs = [], [], []

    # Train: all 20 augmented variants for each train base
    for base_idx in train_idxs:
        for aug_idx in range(20):
            img_path  = AUG_IMG_DIR  / f"{base_idx:03d}_{aug_idx:02d}_img.png"
            mask_path = AUG_MASK_DIR / f"{base_idx:03d}_{aug_idx:02d}_mask.png"
            if img_path.exists() and mask_path.exists():
                train_pairs.append((img_path, mask_path))

    # Val/test: raw images only (no augmentation → unseen gate positions)
    for base_idx in val_idxs:
        img_path  = RAW_IMG_DIR  / f"{base_idx:03d}_img.png"
        mask_path = RAW_MASK_DIR / f"{base_idx:03d}_mask.png"
        if img_path.exists() and mask_path.exists():
            val_pairs.append((img_path, mask_path))

    for base_idx in test_idxs:
        img_path  = RAW_IMG_DIR  / f"{base_idx:03d}_img.png"
        mask_path = RAW_MASK_DIR / f"{base_idx:03d}_mask.png"
        if img_path.exists() and mask_path.exists():
            test_pairs.append((img_path, mask_path))

    return (
        GateDataset(train_pairs, normalize_imagenet=normalize_imagenet),
        GateDataset(val_pairs,   normalize_imagenet=normalize_imagenet),
        GateDataset(test_pairs,  normalize_imagenet=normalize_imagenet),
    )


if __name__ == "__main__":
    train_ds, val_ds, test_ds = build_datasets()
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    img, mask = train_ds[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print(f"Image range: [{img.min():.2f}, {img.max():.2f}]")
    print(f"Mask unique values: {mask.unique().tolist()}")
