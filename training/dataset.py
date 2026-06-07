"""
GateDataset — loads synthetic gate image+mask pairs for training.

Split strategy: 120 train bases / 15 val bases / 15 test bases (seed=42).
Train set uses all 20 augmented variants per base (2400 pairs).
Val and test use raw images only (15 pairs each) → no data leakage.

Images resize to 384×384 (MonoRace's input resolution).

FDA augmentation: pass fda_target_dir to build_datasets() once real sim
captures exist (data/sim_captures/). No-op when the directory is empty or
not provided.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset


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


# ── FDA helper ────────────────────────────────────────────────────────────────

def _fda_source_to_target(
    src: np.ndarray,
    tgt: np.ndarray,
    beta: float = 0.01,
) -> np.ndarray:
    """
    Fourier Domain Adaptation: swap the low-frequency amplitude of src toward
    tgt while preserving the semantic phase of src.

    src, tgt: (H, W, 3) float32 in [0, 1].
    beta:     controls the size of the low-frequency window to swap.
              beta=0.01 → ~8×8 central region for 384×384 input.

    Returns float32 adapted image in [0, 1] (clipped).

    Key invariant: phase is extracted from the *non-shifted* FFT so that the
    reconstruction uses unshifted amplitude + unshifted phase, which gives a
    correct inverse transform. fftshift is only used to identify the central
    region for the amplitude swap.
    """
    H, W = src.shape[:2]
    h_half = max(1, int(H * beta))
    w_half = max(1, int(W * beta))
    cy, cx = H // 2, W // 2

    result = np.empty_like(src)
    for c in range(3):
        fft_src = np.fft.fft2(src[:, :, c])
        fft_tgt = np.fft.fft2(tgt[:, :, c])

        amp_src = np.fft.fftshift(np.abs(fft_src))
        amp_tgt = np.fft.fftshift(np.abs(fft_tgt))
        pha_src = np.angle(fft_src)  # non-shifted — critical for correct ifft

        amp_src[cy - h_half:cy + h_half, cx - w_half:cx + w_half] = \
            amp_tgt[cy - h_half:cy + h_half, cx - w_half:cx + w_half]

        amp_new = np.fft.ifftshift(amp_src)
        fft_new = amp_new * np.exp(1j * pha_src)
        result[:, :, c] = np.real(np.fft.ifft2(fft_new))

    return np.clip(result, 0.0, 1.0)


# ── Dataset ───────────────────────────────────────────────────────────────────

class GateDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        normalize_imagenet: bool = False,
        fda_targets: list[Path] | None = None,
        fda_beta: float = 0.01,
        fda_prob: float = 0.5,
    ):
        self.pairs = pairs
        self.normalize_imagenet = normalize_imagenet
        self.fda_targets = fda_targets or []
        self.fda_beta = fda_beta
        self.fda_prob = fda_prob
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        img  = _load_rgb(img_path)
        mask = _load_mask(mask_path)

        # Resize to 384×384
        img  = cv2.resize(img,  (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)

        # FDA augmentation (only when target images are available)
        if self.fda_targets and self._rng.random() < self.fda_prob:
            tgt_path = self.fda_targets[self._rng.integers(len(self.fda_targets))]
            tgt_bgr  = cv2.imread(str(tgt_path))
            if tgt_bgr is not None:
                tgt_rgb    = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB)
                tgt_r      = cv2.resize(tgt_rgb, (INPUT_SIZE, INPUT_SIZE))
                src_f32    = img.astype(np.float32) / 255.0
                tgt_f32    = tgt_r.astype(np.float32) / 255.0
                adapted    = _fda_source_to_target(src_f32, tgt_f32, beta=self.fda_beta)
                img        = (adapted * 255).astype(np.uint8)

        # Image → float32 tensor (3, H, W) in [0, 1]
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.normalize_imagenet:
            mean  = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std   = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            img_t = (img_t - mean) / std

        # Mask → float32 tensor (1, H, W) binary in {0, 1}
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        return img_t, mask_t


def get_split_indices(n_bases: int = 150, seed: int = 42):
    rng       = np.random.default_rng(seed)
    idxs      = rng.permutation(n_bases)
    n_val     = 15
    n_test    = 15
    test_idxs  = set(idxs[:n_test].tolist())
    val_idxs   = set(idxs[n_test:n_test + n_val].tolist())
    train_idxs = set(idxs[n_test + n_val:].tolist())
    return train_idxs, val_idxs, test_idxs


def build_datasets(
    normalize_imagenet: bool = False,
    fda_target_dir: Path | None = None,
    fda_beta: float = 0.01,
) -> tuple[GateDataset, GateDataset, GateDataset]:
    """
    Build train/val/test datasets.

    fda_target_dir: directory of real images to use as FDA style targets.
                    Pass Path("data/sim_captures/") once you have captured
                    real sim frames with the 'S' key in sim_inference.py.
                    No-op when None or the directory contains no images.
    """
    train_idxs, val_idxs, test_idxs = get_split_indices()

    # Collect FDA target images (PNGs and JPEGs)
    fda_targets: list[Path] = []
    if fda_target_dir is not None and fda_target_dir.exists():
        fda_targets = sorted(
            list(fda_target_dir.glob("*.png")) + list(fda_target_dir.glob("*.jpg"))
        )

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

    if fda_targets:
        print(f"FDA: {len(fda_targets)} target images from {fda_target_dir}")

    return (
        GateDataset(train_pairs, normalize_imagenet=normalize_imagenet,
                    fda_targets=fda_targets, fda_beta=fda_beta),
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
