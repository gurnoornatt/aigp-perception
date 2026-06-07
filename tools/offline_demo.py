"""
Offline demo — runs GateNet (or simple UNet) on raw dataset images.

Shows a side-by-side window:  left = original  |  right = gate mask overlay
Press SPACE to advance, Q to quit, S to save current frame.

Run from aigp-perception/:
    python tools/offline_demo.py
    python tools/offline_demo.py --model simple
    python tools/offline_demo.py --delay 500    # ms between frames (0 = manual)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from training.gatenet import GateNet

INPUT_SIZE = 384
DEVICE     = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR   = Path("checkpoints")
SAVE_DIR   = Path("data/sim_captures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RAW_IMG_DIR  = Path("data/raw/images")
RAW_MASK_DIR = Path("data/raw/masks")


def load_gatenet():
    ckpt_path = CKPT_DIR / "gatenet_best.pt"
    model = GateNet(f=4).to(DEVICE)
    ckpt  = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"GateNet loaded  val_iou={ckpt.get('val_iou', 0):.4f}  device={DEVICE}")
    return model, False


def load_simple():
    import segmentation_models_pytorch as smp
    ckpt_path = CKPT_DIR / "simple_best.pt"
    model = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1, activation=None).to(DEVICE)
    ckpt  = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Simple UNet loaded  val_iou={ckpt.get('val_iou', 0):.4f}  device={DEVICE}")
    return model, True


def run_inference(model, bgr, use_imagenet=False):
    orig_h, orig_w = bgr.shape[:2]
    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    img_t  = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    if use_imagenet:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std
    img_t = img_t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out    = model(img_t)
        logits = out[-1] if isinstance(out, list) else out
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8) * 255
    return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def overlay_mask(bgr, mask, alpha=0.45):
    overlay  = bgr.copy()
    green    = np.zeros_like(bgr)
    green[:, :, 1] = mask
    cv2.addWeighted(green, alpha, overlay, 1.0, 0, overlay)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


def main(use_simple=False, delay_ms=0):
    model, use_imagenet = load_simple() if use_simple else load_gatenet()
    model_name = "Simple UNet" if use_simple else "GateNet"

    img_paths = sorted(RAW_IMG_DIR.glob("*_img.png"))
    if not img_paths:
        print("No images found in data/raw/images/"); sys.exit(1)

    print(f"\n{len(img_paths)} images  |  SPACE=next  Q=quit  S=save")
    print(f"delay={delay_ms}ms (0 = manual advance)\n")

    save_count = 0
    for i, img_path in enumerate(img_paths):
        bgr = cv2.imread(str(img_path))

        t0   = time.perf_counter()
        mask = run_inference(model, bgr, use_imagenet)
        ms   = (time.perf_counter() - t0) * 1000

        overlay = overlay_mask(bgr, mask)

        # Coverage %
        cov = (mask > 0).mean() * 100
        cv2.putText(overlay, f"{model_name} | {ms:.1f}ms | gate={cov:.1f}%",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(bgr, f"{img_path.name}  [{i+1}/{len(img_paths)}]",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

        combined = np.hstack([bgr, overlay])
        cv2.imshow("Gate Segmentation — offline demo", combined)

        wait = delay_ms if delay_ms > 0 else 0
        key  = cv2.waitKey(wait) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = SAVE_DIR / f"offline_{save_count:04d}.png"
            cv2.imwrite(str(fname), combined)
            print(f"Saved {fname}")
            save_count += 1
        elif key == ord(" ") or delay_ms > 0:
            continue
        else:
            cv2.waitKey(0)  # wait for any key

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gatenet", "simple"], default="gatenet")
    parser.add_argument("--delay", type=int, default=0,
                        help="ms between frames; 0 = press SPACE to advance")
    args = parser.parse_args()
    main(use_simple=(args.model == "simple"), delay_ms=args.delay)
