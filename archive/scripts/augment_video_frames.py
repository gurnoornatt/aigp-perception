"""
Augment labelled sim frames from experiments/video/ into data/sim/.

Expects:
    experiments/video/frames/frame_XX.png
    experiments/video/masks/frame_XX_mask.png

Writes:
    data/sim/raw/images/{idx:03d}_img.png
    data/sim/raw/masks/{idx:03d}_mask.png
    data/sim/augmented/images/{idx:03d}_{aug:02d}_img.png
    data/sim/augmented/masks/{idx:03d}_{aug:02d}_mask.png

Run:
    python scripts/augment_video_frames.py
    python scripts/augment_video_frames.py --per-base 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from scripts.augment_dataset import AUGMENTATIONS_PER_BASE, TRANSFORM

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = REPO_ROOT / "experiments" / "video" / "frames"
MASKS_DIR = REPO_ROOT / "experiments" / "video" / "masks"

RAW_IMG_DIR = REPO_ROOT / "data" / "sim" / "raw" / "images"
RAW_MASK_DIR = REPO_ROOT / "data" / "sim" / "raw" / "masks"
AUG_IMG_DIR = REPO_ROOT / "data" / "sim" / "augmented" / "images"
AUG_MASK_DIR = REPO_ROOT / "data" / "sim" / "augmented" / "masks"

FRAME_RE = re.compile(r"^frame_(\d{2})\.png$")
TARGET_W, TARGET_H = 640, 360


def list_pairs() -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    for frame_path in sorted(FRAMES_DIR.glob("frame_*.png")):
        m = FRAME_RE.match(frame_path.name)
        if not m:
            continue
        stem = frame_path.stem
        mask_path = MASKS_DIR / f"{stem}_mask.png"
        if mask_path.exists():
            pairs.append((frame_path, mask_path, stem))
    return pairs


def _resize_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.resize(image, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8) * 255
    return image, mask


def augment_video_frames(per_base: int = AUGMENTATIONS_PER_BASE) -> None:
    pairs = list_pairs()
    if not pairs:
        print("No frame/mask pairs found. Label masks in experiments/video/masks/ first.")
        return

    for d in (RAW_IMG_DIR, RAW_MASK_DIR, AUG_IMG_DIR, AUG_MASK_DIR):
        d.mkdir(parents=True, exist_ok=True)

    all_frames = sorted(
        p.name for p in FRAMES_DIR.glob("frame_*.png") if FRAME_RE.match(p.name)
    )
    missing = [f for f in all_frames if not (MASKS_DIR / f"{Path(f).stem}_mask.png").exists()]
    if missing:
        print(f"Using {len(pairs)}/{len(all_frames)} labelled frames.")
        print(f"Missing masks for: {', '.join(missing[:8])}" + (" ..." if len(missing) > 8 else ""))

    total = len(pairs) * per_base
    print(f"Augmenting {len(pairs)} sim frames x {per_base} = {total} pairs -> data/sim/")

    manifest = []
    with tqdm(total=total) as pbar:
        for base_idx, (img_path, mask_path, stem) in enumerate(pairs):
            manifest.append({"base_idx": base_idx, "source_frame": stem})
            image = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                print(f"Warning: could not load {img_path} or {mask_path}, skipping.")
                continue

            image, mask = _resize_pair(image, mask)
            cv2.imwrite(str(RAW_IMG_DIR / f"{base_idx:03d}_img.png"), image)
            cv2.imwrite(str(RAW_MASK_DIR / f"{base_idx:03d}_mask.png"), mask)

            for aug_idx in range(per_base):
                result = TRANSFORM(image=image, mask=mask)
                aug_image = result["image"]
                aug_mask = (result["mask"] > 127).astype(np.uint8) * 255

                out_stem = f"{base_idx:03d}_{aug_idx:02d}"
                cv2.imwrite(str(AUG_IMG_DIR / f"{out_stem}_img.png"), aug_image)
                cv2.imwrite(str(AUG_MASK_DIR / f"{out_stem}_mask.png"), aug_mask)
                pbar.update(1)

    from training.dataset import get_split_indices

    n_bases = len(manifest)
    train_idxs, val_idxs, test_idxs = get_split_indices(n_bases)
    meta = {
        "n_bases": n_bases,
        "augs_per_base": per_base,
        "manifest": manifest,
        "split": {
            "train": sorted(train_idxs),
            "val": sorted(val_idxs),
            "test": sorted(test_idxs),
        },
    }
    meta_path = REPO_ROOT / "data" / "sim" / "dataset.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"  {RAW_IMG_DIR}  ({len(list(RAW_IMG_DIR.glob('*.png')))} images)")
    print(f"  {RAW_MASK_DIR} ({len(list(RAW_MASK_DIR.glob('*.png')))} masks)")
    print(f"  {AUG_IMG_DIR}  ({len(list(AUG_IMG_DIR.glob('*.png')))} images)")
    print(f"  {AUG_MASK_DIR} ({len(list(AUG_MASK_DIR.glob('*.png')))} masks)")
    print(f"  {meta_path}")
    print(f"  Split: train={len(train_idxs)} val={len(val_idxs)} test={len(test_idxs)} bases")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-base", type=int, default=AUGMENTATIONS_PER_BASE)
    args = parser.parse_args()
    augment_video_frames(per_base=max(1, args.per_base))


if __name__ == "__main__":
    main()
