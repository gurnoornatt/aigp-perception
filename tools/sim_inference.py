"""
Live gate segmentation on the sim camera feed.

Listens on UDP port 5601 (same port the sim streams to), runs GateNet
on every frame, and shows a live window:
    Left:  original sim frame
    Right: frame with gate mask overlaid in green

Press Q to quit. Press S to save the current frame.

Run from aigp-perception/:
    python tools/sim_inference.py
    python tools/sim_inference.py --model simple   # use smp resnet18 instead
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../aigp-pyaipilot-remote"))

import argparse
import socket
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from training.gatenet import GateNet

VISION_PORT   = 5601
VISION_BIND   = "0.0.0.0"
INPUT_SIZE    = 384
DEVICE        = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR      = Path("checkpoints")
SAVE_DIR      = Path("data/sim_captures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ── Minimal UDP frame assembler (mirrors pyaipilot/vision_stream.py) ──────────
import struct

HEADER_FMT  = "<IHHIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

class _Partial:
    def __init__(self, jpeg_size, sim_time_ns, total_chunks, created_at):
        self.jpeg_size   = jpeg_size
        self.sim_time_ns = sim_time_ns
        self.total_chunks = total_chunks
        self.created_at  = created_at
        self.chunks      = {}

class FrameAssembler:
    def __init__(self):
        self._frames = {}

    def add(self, packet: bytes):
        if len(packet) < HEADER_SIZE:
            return None
        frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = \
            struct.unpack(HEADER_FMT, packet[:HEADER_SIZE])
        if total_chunks == 0 or chunk_id >= total_chunks:
            return None
        payload = packet[HEADER_SIZE: HEADER_SIZE + payload_size]

        pf = self._frames.get(frame_id)
        if pf is None:
            pf = _Partial(jpeg_size, sim_time_ns, total_chunks, time.time())
            self._frames[frame_id] = pf
        pf.chunks[chunk_id] = payload

        if len(pf.chunks) < pf.total_chunks:
            return None

        jpeg = b"".join(pf.chunks[i] for i in range(pf.total_chunks))
        del self._frames[frame_id]
        arr = np.frombuffer(jpeg[:pf.jpeg_size], dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img  # BGR numpy array, or None if decode failed


# ── Model loader ──────────────────────────────────────────────────────────────

def load_gatenet() -> GateNet:
    ckpt_path = CKPT_DIR / "gatenet_best.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}. Run: python training/train_gatenet.py --epochs 5")
        sys.exit(1)
    model = GateNet(f=4).to(DEVICE)
    ckpt  = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded GateNet from {ckpt_path} (val IoU={ckpt.get('val_iou', '?'):.4f})")
    return model


def load_simple():
    import segmentation_models_pytorch as smp
    ckpt_path = CKPT_DIR / "simple_best.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}. Run: python training/train_simple.py --epochs 5")
        sys.exit(1)
    model = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1, activation=None).to(DEVICE)
    ckpt  = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded Simple UNet from {ckpt_path} (val IoU={ckpt.get('val_iou', '?'):.4f})")
    return model


# ── Inference on a single BGR frame ──────────────────────────────────────────

def run_inference(model, bgr_frame: np.ndarray, use_imagenet_norm: bool = False) -> np.ndarray:
    """Returns predicted binary mask (H, W) uint8 {0, 255}."""
    orig_h, orig_w = bgr_frame.shape[:2]

    # Preprocess: BGR → RGB → resize → tensor
    rgb   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    img_t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

    if use_imagenet_norm:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

    img_t = img_t.unsqueeze(0).to(DEVICE)  # (1, 3, 384, 384)

    with torch.no_grad():
        if isinstance(model, GateNet):
            outputs = model(img_t)
            logits  = outputs[-1]       # y4 = final full-res output
        else:
            logits  = model(img_t)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (384, 384)

    # Resize mask back to original frame size
    mask = (prob > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask


def overlay_mask(bgr_frame: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Draw gate mask as green overlay on the frame."""
    overlay = bgr_frame.copy()
    green   = np.zeros_like(bgr_frame)
    green[:, :, 1] = mask   # green channel = mask
    cv2.addWeighted(green, alpha, overlay, 1.0, 0, overlay)
    # Also draw contour for clarity
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


# ── Main ──────────────────────────────────────────────────────────────────────

def main(use_simple: bool = False) -> None:
    model = load_simple() if use_simple else load_gatenet()
    use_imagenet = use_simple

    assembler   = FrameAssembler()
    latest_frame = {"img": None, "count": 0}
    lock         = threading.Lock()

    def recv_loop():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        sock.bind((VISION_BIND, VISION_PORT))
        print(f"Listening for sim camera on port {VISION_PORT}...")
        while True:
            try:
                pkt, _ = sock.recvfrom(65536)
                img = assembler.add(pkt)
                if img is not None:
                    with lock:
                        latest_frame["img"]   = img
                        latest_frame["count"] += 1
            except socket.timeout:
                continue

    t = threading.Thread(target=recv_loop, daemon=True)
    t.start()

    print("Waiting for first frame... (make sure sim is running)")
    print("Controls: Q = quit | S = save current frame")

    model_name  = "Simple UNet" if use_simple else "GateNet"
    frame_count = 0
    fps_t0      = time.time()
    fps_display = 0.0
    save_count  = 0

    while True:
        with lock:
            raw = latest_frame["img"]
            cur = latest_frame["count"]

        if raw is None:
            cv2.waitKey(50)
            continue

        if cur == frame_count:
            cv2.waitKey(10)
            continue

        frame_count = cur

        # Inference
        t0   = time.perf_counter()
        mask = run_inference(model, raw, use_imagenet_norm=use_imagenet)
        dt   = (time.perf_counter() - t0) * 1000

        # FPS
        elapsed = time.time() - fps_t0
        if elapsed > 1.0:
            fps_display = frame_count / elapsed

        overlay = overlay_mask(raw, mask)

        # Labels
        cv2.putText(overlay, f"{model_name} | {dt:.1f}ms | {fps_display:.1f} fps",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(raw, "Original", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        combined = np.hstack([raw, overlay])
        cv2.imshow("Gate Segmentation — sim live", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = SAVE_DIR / f"capture_{save_count:04d}.png"
            cv2.imwrite(str(fname), combined)
            print(f"Saved {fname}")
            save_count += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gatenet", "simple"], default="gatenet")
    args = parser.parse_args()
    main(use_simple=(args.model == "simple"))
