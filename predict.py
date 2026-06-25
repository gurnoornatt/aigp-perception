"""
predict.py — Run inference on a single image and save the predicted mask.

Usage:
    python predict.py --model checkpoints/best.pt --input image.png --output mask.png

The output is a grayscale PNG where 255 = gate, 0 = background.
The mask is resized back to the original image resolution.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import UNet


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="U-Net inference on a single image")
    p.add_argument("--model",        type=Path,  required=True, help="Path to best.pt checkpoint")
    p.add_argument("--input",        type=Path,  required=True, help="Input image path")
    p.add_argument("--output",       type=Path,  default=Path("prediction.png"))
    p.add_argument("--img-size",     type=int,   default=512,   help="Must match training resolution")
    p.add_argument("--base-filters", type=int,   default=64)
    p.add_argument("--bilinear",     action="store_true")
    p.add_argument("--threshold",    type=float, default=0.5,   help="Sigmoid threshold for binarizing")
    p.add_argument("--device",       type=str,   default="cpu")
    return p.parse_args()


def load_model(path: Path, base_filters: int, bilinear: bool, device: torch.device) -> UNet:
    model = UNet(n_channels=3, n_classes=1, bilinear=bilinear, base_filters=base_filters)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval().to(device)


def preprocess(image_path: Path, size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load image, resize to model input size, return tensor and original (H, W)."""
    img = Image.open(image_path).convert("RGB")
    orig_size = (img.height, img.width)
    img = img.resize((size, size), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), orig_size  # (1, 3, H, W)


def postprocess(logits: torch.Tensor, orig_size: tuple[int, int], threshold: float) -> Image.Image:
    """Sigmoid → threshold → resize to original resolution → PIL mask."""
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W) in [0, 1]
    mask = (prob > threshold).astype(np.uint8) * 255
    return Image.fromarray(mask).resize((orig_size[1], orig_size[0]), Image.NEAREST)


def main() -> None:
    args   = get_args()
    device = torch.device(args.device)

    model  = load_model(args.model, args.base_filters, args.bilinear, device)
    tensor, orig_size = preprocess(args.input, args.img_size)
    tensor = tensor.to(device)

    with torch.inference_mode():
        logits = model(tensor)

    mask = postprocess(logits, orig_size, args.threshold)
    mask.save(args.output)
    print(f"Saved mask → {args.output}")


if __name__ == "__main__":
    main()
