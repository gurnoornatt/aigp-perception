"""
GateDataset — loads gate image+mask pairs for training.

Datasets:
    synthetic  data/raw + data/augmented  (150 bases from generate_synthetic.py)
    sim        data/sim/raw + data/sim/augmented  (labelled video frames)

Split: ~80% train / 10% val / 10% test by base index (seed=42).
Train uses all augmented variants per train base; val/test use raw only.

Images resize to 384x384 (MonoRace's input resolution).
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

SIM_RAW_IMG_DIR  = Path("data/sim/raw/images")
SIM_RAW_MASK_DIR = Path("data/sim/raw/masks")
SIM_AUG_IMG_DIR  = Path("data/sim/augmented/images")
SIM_AUG_MASK_DIR = Path("data/sim/augmented/masks")

AUGS_PER_BASE = 20

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
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(n_bases)
    if n_bases <= 3:
        n_test, n_val = 1, 1
    else:
        n_test = max(1, round(n_bases * 0.1))
        n_val = max(1, round(n_bases * 0.1))
        if n_test + n_val >= n_bases:
            n_test, n_val = 1, 1
    test_idxs = set(idxs[:n_test].tolist())
    val_idxs = set(idxs[n_test:n_test + n_val].tolist())
    train_idxs = set(idxs[n_test + n_val:].tolist())
    return train_idxs, val_idxs, test_idxs


def _count_bases(raw_img_dir: Path) -> int:
    return len(list(raw_img_dir.glob("*_img.png")))


def build_datasets(
    normalize_imagenet: bool = False,
    dataset: str = "sim",
    raw_img_dir: Path | None = None,
    raw_mask_dir: Path | None = None,
    aug_img_dir: Path | None = None,
    aug_mask_dir: Path | None = None,
    augs_per_base: int = AUGS_PER_BASE,
):
    if dataset == "sim":
        raw_img_dir = raw_img_dir or SIM_RAW_IMG_DIR
        raw_mask_dir = raw_mask_dir or SIM_RAW_MASK_DIR
        aug_img_dir = aug_img_dir or SIM_AUG_IMG_DIR
        aug_mask_dir = aug_mask_dir or SIM_AUG_MASK_DIR
    elif dataset == "synthetic":
        raw_img_dir = raw_img_dir or RAW_IMG_DIR
        raw_mask_dir = raw_mask_dir or RAW_MASK_DIR
        aug_img_dir = aug_img_dir or AUG_IMG_DIR
        aug_mask_dir = aug_mask_dir or AUG_MASK_DIR
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_bases = _count_bases(raw_img_dir)
    if n_bases == 0:
        raise FileNotFoundError(
            f"No images in {raw_img_dir}. "
            f"Run scripts/augment_video_frames.py for sim, or scripts/generate_synthetic.py + augment_dataset.py."
        )

    train_idxs, val_idxs, test_idxs = get_split_indices(n_bases)

    train_pairs, val_pairs, test_pairs = [], [], []

    for base_idx in train_idxs:
        for aug_idx in range(augs_per_base):
            img_path = aug_img_dir / f"{base_idx:03d}_{aug_idx:02d}_img.png"
            mask_path = aug_mask_dir / f"{base_idx:03d}_{aug_idx:02d}_mask.png"
            if img_path.exists() and mask_path.exists():
                train_pairs.append((img_path, mask_path))

    for base_idx in val_idxs:
        img_path = raw_img_dir / f"{base_idx:03d}_img.png"
        mask_path = raw_mask_dir / f"{base_idx:03d}_mask.png"
        if img_path.exists() and mask_path.exists():
            val_pairs.append((img_path, mask_path))

    for base_idx in test_idxs:
        img_path = raw_img_dir / f"{base_idx:03d}_img.png"
        mask_path = raw_mask_dir / f"{base_idx:03d}_mask.png"
        if img_path.exists() and mask_path.exists():
            test_pairs.append((img_path, mask_path))

    return (
        GateDataset(train_pairs, normalize_imagenet=normalize_imagenet),
        GateDataset(val_pairs, normalize_imagenet=normalize_imagenet),
        GateDataset(test_pairs, normalize_imagenet=normalize_imagenet),
        {"n_bases": n_bases, "train": sorted(train_idxs), "val": sorted(val_idxs), "test": sorted(test_idxs)},
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sim", "synthetic"], default="sim")
    args = parser.parse_args()

    train_ds, val_ds, test_ds, split = build_datasets(dataset=args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Split: {json.dumps(split)}")
    img, mask = train_ds[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print(f"Image range: [{img.min():.2f}, {img.max():.2f}]")
    print(f"Mask unique values: {mask.unique().tolist()}")
