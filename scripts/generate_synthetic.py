"""
Synthetic gate image + mask generator.

Produces 150 paired (image, mask) PNG files covering diverse distances,
positions, and perspective angles — no sim required, no hand labeling.

Output:
    data/raw/images/{idx:03d}_img.png   — 640x360 BGR gate image
    data/raw/masks/{idx:03d}_mask.png   — 640x360 grayscale binary mask

The gate is a RED square frame (outer ~2.7m, inner ~1.5m) on a dark
desaturated background, matching what the sim camera sees.
The mask is WHITE where the gate frame body is, BLACK everywhere else.

Run:
    python scripts/generate_synthetic.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import random
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from perception.config import GATE_INNER_SIZE_M, GATE_OUTER_SIZE_M

# ── Output directories ──────────────────────────────────────────────────────
IMG_DIR  = Path("data/raw/images")
MASK_DIR = Path("data/raw/masks")
IMG_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

# ── Image dimensions ─────────────────────────────────────────────────────────
W, H = 640, 360

# ── Background ───────────────────────────────────────────────────────────────
BG_COLOR   = (15, 20, 30)    # dark desaturated blue-black (BGR)
BG_NOISE_S = 8               # Gaussian noise sigma on background

# ── Gate colours (BGR) ───────────────────────────────────────────────────────
GATE_RED_MAIN  = (0,  30, 210)   # primary red fill
GATE_RED_DARK  = (0,  15, 160)   # darker red for checkered alternate
GATE_EDGE_GLOW = (40, 60, 240)   # thin bright highlight at outer edge

# ── Inner/outer ratio from real geometry ─────────────────────────────────────
INNER_OUTER_RATIO = GATE_INNER_SIZE_M / GATE_OUTER_SIZE_M   # 1.5 / 2.7 ≈ 0.556

# ── Parameter grid  (5 × 5 × 3 × 2 = 150) ──────────────────────────────────
OUTER_PX_TIERS = [60, 110, 170, 240, 320]        # gate apparent size (pixels)
CX_OFFSETS     = [-200, -100, 0, 100, 200]        # horizontal offset from centre
CY_OFFSETS     = [-80, 0, 80]                     # vertical offset from centre
TILT_STRENGTHS = [0.08, 0.20]                     # mild / moderate perspective tilt

JITTER = 0.20    # ±20% random jitter per parameter


# ── Helpers ──────────────────────────────────────────────────────────────────

def _jitter(value: float) -> float:
    return value * (1.0 + random.uniform(-JITTER, JITTER))


def _square_corners(cx: float, cy: float, half: float) -> np.ndarray:
    """Return 4 corners of an axis-aligned square as int32 array (TL, TR, BR, BL)."""
    return np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ], dtype=np.float32)


def _draw_checkered_strip(canvas: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> None:
    """Draw alternating red/dark-red strips on each side of the frame body."""
    for side in range(4):
        o0, o1 = outer[side], outer[(side + 1) % 4]
        i0, i1 = inner[side], inner[(side + 1) % 4]
        n_checks = 6
        for k in range(n_checks):
            t0, t1 = k / n_checks, (k + 1) / n_checks
            p = np.array([
                o0 + t0 * (o1 - o0),
                o0 + t1 * (o1 - o0),
                i0 + t1 * (i1 - i0),
                i0 + t0 * (i1 - i0),
            ], dtype=np.float32)
            color = GATE_RED_MAIN if k % 2 == 0 else GATE_RED_DARK
            cv2.fillConvexPoly(canvas, p.astype(np.int32), color)


def _build_perspective_matrix(outer: np.ndarray, tilt: float) -> np.ndarray:
    """
    Build a perspective warp matrix that simulates viewing the gate from an angle.
    tilt controls how much the top edge shifts left/right relative to the bottom.
    """
    shift_x = (outer[1][0] - outer[0][0]) * tilt   # fraction of gate width
    shift_y = (outer[2][1] - outer[0][1]) * tilt * 0.3

    src = outer.astype(np.float32)
    dst = np.array([
        [src[0][0] + shift_x,  src[0][1] - shift_y],
        [src[1][0] + shift_x,  src[1][1] - shift_y],
        [src[2][0] - shift_x,  src[2][1] + shift_y],
        [src[3][0] - shift_x,  src[3][1] + shift_y],
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(src, dst)


def generate_sample(idx: int, outer_px: float, cx: float, cy: float, tilt: float) -> None:
    """Render one (image, mask) pair and save it."""
    inner_px = outer_px * INNER_OUTER_RATIO
    outer_half = outer_px / 2.0
    inner_half = inner_px / 2.0

    outer = _square_corners(cx, cy, outer_half)
    inner = _square_corners(cx, cy, inner_half)

    # ── Canvas with noisy background ─────────────────────────────────────────
    canvas = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
    noise  = np.random.normal(0, BG_NOISE_S, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Optional faint wireframe grid on background (simulates sim aesthetic)
    for x in range(0, W, 60):
        cv2.line(canvas, (x, 0), (x, H), (25, 35, 45), 1)
    for y in range(0, H, 40):
        cv2.line(canvas, (0, y), (W, y), (25, 35, 45), 1)

    # ── Draw gate frame ───────────────────────────────────────────────────────
    cv2.fillConvexPoly(canvas, outer.astype(np.int32), GATE_RED_MAIN)
    _draw_checkered_strip(canvas, outer, inner)
    cv2.fillConvexPoly(canvas, inner.astype(np.int32), BG_COLOR)          # punch inner hole
    cv2.polylines(canvas, [outer.astype(np.int32)], True, GATE_EDGE_GLOW, 2)  # bright edge

    # ── Binary mask: outer=255, inner=0 ──────────────────────────────────────
    mask_canvas = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask_canvas, outer.astype(np.int32), 255)
    cv2.fillConvexPoly(mask_canvas, inner.astype(np.int32), 0)

    # ── Apply perspective warp (same matrix to both) ──────────────────────────
    M = _build_perspective_matrix(outer, tilt)
    canvas      = cv2.warpPerspective(canvas, M, (W, H),
                                      borderValue=BG_COLOR)
    mask_canvas = cv2.warpPerspective(mask_canvas, M, (W, H),
                                      flags=cv2.INTER_NEAREST,
                                      borderValue=0)

    # ── Save ──────────────────────────────────────────────────────────────────
    cv2.imwrite(str(IMG_DIR  / f"{idx:03d}_img.png"),  canvas)
    cv2.imwrite(str(MASK_DIR / f"{idx:03d}_mask.png"), mask_canvas)


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    samples = []
    for outer_px in OUTER_PX_TIERS:
        for cx_off in CX_OFFSETS:
            for cy_off in CY_OFFSETS:
                for tilt in TILT_STRENGTHS:
                    cx = W / 2 + _jitter(cx_off) if cx_off != 0 else W / 2 + random.uniform(-20, 20)
                    cy = H / 2 + _jitter(cy_off) if cy_off != 0 else H / 2 + random.uniform(-15, 15)
                    op = _jitter(outer_px)
                    t  = _jitter(tilt) * random.choice([-1, 1])

                    # Clamp gate inside image bounds
                    half = op / 2
                    cx = float(np.clip(cx, half + 5, W - half - 5))
                    cy = float(np.clip(cy, half + 5, H - half - 5))

                    # Skip if gate would be too small to be useful
                    if op < 30:
                        continue

                    samples.append((op, cx, cy, t))

    print(f"Generating {len(samples)} synthetic gate images...")
    for idx, (op, cx, cy, t) in enumerate(tqdm(samples)):
        generate_sample(idx, op, cx, cy, t)

    print(f"\nDone. Saved to:")
    print(f"  {IMG_DIR}  ({len(list(IMG_DIR.glob('*.png')))} images)")
    print(f"  {MASK_DIR} ({len(list(MASK_DIR.glob('*.png')))} masks)")


if __name__ == "__main__":
    main()
