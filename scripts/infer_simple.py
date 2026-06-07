"""
Run the trained simple UNet on a single image.

Usage:
    python scripts/infer_simple.py
    python scripts/infer_simple.py path/to/image.png
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from training.dataset import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "image.png"
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "simple_sim_scratch.pt"
OUT_DIR = REPO_ROOT / "experiments" / "unet"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt


def preprocess(img_bgr: np.ndarray, normalize_imagenet: bool) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    if normalize_imagenet:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor.unsqueeze(0), (h, w)


@torch.no_grad()
def predict_mask(model: torch.nn.Module, img_bgr: np.ndarray, normalize_imagenet: bool) -> np.ndarray:
    batch, (h, w) = preprocess(img_bgr, normalize_imagenet)
    logits = model(batch.to(DEVICE))
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def save_outputs(img_bgr: np.ndarray, mask: np.ndarray, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mask_path = OUT_DIR / f"{stem}_mask.png"
    overlay_path = OUT_DIR / f"{stem}_overlay.png"
    compare_path = OUT_DIR / f"{stem}_compare.png"

    cv2.imwrite(str(mask_path), mask)

    overlay = img_bgr.copy()
    tint = np.zeros_like(img_bgr)
    tint[:, :, 2] = mask  # red tint on predicted gate pixels
    overlay = cv2.addWeighted(overlay, 0.7, tint, 0.3, 0)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    compare = np.hstack([img_bgr, mask_bgr, overlay])

    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(compare_path), compare)

    print(f"Mask:    {mask_path}")
    print(f"Overlay: {overlay_path}")
    print(f"Compare: {compare_path}")
    print(f"Gate pixels: {int((mask > 0).sum())} / {mask.size} ({100 * (mask > 0).mean():.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--from-scratch", action="store_true",
                        help="No ImageNet normalization (auto-detected from checkpoint if omitted)")
    args = parser.parse_args()

    image_path = Path(args.image)
    ckpt_path = Path(args.ckpt)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise SystemExit(f"Could not read image: {image_path}")

    model, ckpt = load_model(ckpt_path)
    from_scratch = args.from_scratch or ckpt.get("from_scratch", False)

    print(f"Device: {DEVICE}")
    print(f"Image:  {image_path}")
    print(f"Ckpt:   {ckpt_path} (epoch {ckpt.get('epoch', '?')}, val IoU {ckpt.get('val_iou', 0):.4f})")
    print(f"Norm:   {'[0,1] only' if from_scratch else 'ImageNet'}")

    mask = predict_mask(model, img_bgr, normalize_imagenet=not from_scratch)
    save_outputs(img_bgr, mask, image_path.stem)


if __name__ == "__main__":
    main()
