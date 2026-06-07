"""
Experiment: isolate red racing gates as white on black.

Run from repo root:
    python scripts/experiment_gate_mask.py [path/to/image.png]

Outputs comparison masks under experiments/gate_mask/
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "image.png"
OUT_DIR = REPO_ROOT / "experiments" / "gate_mask"


def mask_hsv_red(img_bgr: np.ndarray, s_min: int, v_min: int) -> np.ndarray:
    """Red spans low and high hue in OpenCV HSV (H in 0..179)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, s_min, v_min], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([168, s_min, v_min], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(m1, m2)


def mask_excess_red(img_bgr: np.ndarray, margin: int) -> np.ndarray:
    """R channel dominates G and B (works when gates are saturated red)."""
    b, g, r = cv2.split(img_bgr)
    rg = cv2.subtract(r, g)
    rb = cv2.subtract(r, b)
    dom = cv2.min(rg, rb)
    _, mask = cv2.threshold(dom, margin, 255, cv2.THRESH_BINARY)
    return mask


def mask_lab_red(img_bgr: np.ndarray, a_min: int) -> np.ndarray:
    """Red gates have high a* in LAB."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    _, a, _ = cv2.split(lab)
    return cv2.inRange(a, a_min, 255)


def mask_ycrcb_red(img_bgr: np.ndarray, cr_min: int) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)
    return cv2.inRange(cr, cr_min, 255)


def clean_mask(mask: np.ndarray, close_k: int, open_k: int) -> np.ndarray:
    """Fill white text holes on gates; drop small cyan/red speckles."""
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def mask_hsv_red_tuned(img_bgr: np.ndarray) -> np.ndarray:
    """Best-effort tuned HSV + morphology for AIGP-style red gates."""
    raw = mask_hsv_red(img_bgr, s_min=80, v_min=60)
    return clean_mask(raw, close_k=9, open_k=3)


def mask_combined(img_bgr: np.ndarray) -> np.ndarray:
    """HSV red AND excess-red to suppress cyan/blue highlights."""
    hsv = mask_hsv_red(img_bgr, s_min=70, v_min=50)
    ex = mask_excess_red(img_bgr, margin=25)
    raw = cv2.bitwise_and(hsv, ex)
    return clean_mask(raw, close_k=11, open_k=5)


def filter_by_contour_area(mask: np.ndarray, min_area: float) -> np.ndarray:
    """Drop small noise blobs; keep gate-sized connected components."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(out, [cnt], -1, 255, thickness=cv2.FILLED)
    return out


def mask_gates_recommended(img_bgr: np.ndarray) -> np.ndarray:
    """
    Recommended pipeline for AIGP red gates on dark/cyan backgrounds.

    1. HSV dual-range red (hue wraps at 0/179)
    2. Excess-red AND to reject cyan/teal (high G/B vs R)
    3. Close to fill white text/icons inside gate frames
    4. Open lightly, then drop tiny contours
    """
    hsv = mask_hsv_red(img_bgr, s_min=80, v_min=55)
    ex = mask_excess_red(img_bgr, margin=28)
    raw = cv2.bitwise_and(hsv, ex)
    morph = clean_mask(raw, close_k=13, open_k=5)
    # ~0.02% of 778x872 ≈ 135 px; gates are thousands of px
    h, w = morph.shape[:2]
    min_area = max(150.0, 0.00015 * h * w)
    return filter_by_contour_area(morph, min_area=min_area)


def to_bw(mask: np.ndarray) -> np.ndarray:
    return mask  # already 0 / 255


def save_label(path: Path, mask: np.ndarray, label: str, preview_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)
    # overlay: gates white, dim original elsewhere
    overlay = preview_bgr.copy()
    overlay[mask == 0] = (overlay[mask == 0] * 0.25).astype(np.uint8)
    overlay[mask > 0] = (255, 255, 255)
    panel = np.hstack([preview_bgr, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), overlay])
    cv2.putText(
        panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
    )
    cv2.imwrite(str(path.with_name(path.stem + "_panel.png")), panel)


def build_comparison_grid(panels: list[np.ndarray], cols: int = 2) -> np.ndarray:
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)
    norm = []
    for p in panels:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[: p.shape[0], : p.shape[1]] = p
        norm.append(canvas)
    rows = []
    for i in range(0, len(norm), cols):
        row = norm[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(norm[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main() -> None:
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not img_path.is_file():
        print(f"Image not found: {img_path}")
        sys.exit(1)

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"Failed to read: {img_path}")
        sys.exit(1)

    h, w = bgr.shape[:2]
    print(f"Input: {img_path} ({w}x{h})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, np.ndarray]] = [
        ("hsv_s70_v50", mask_hsv_red(bgr, s_min=70, v_min=50)),
        ("hsv_s80_v60", mask_hsv_red(bgr, s_min=80, v_min=60)),
        ("hsv_s100_v80", mask_hsv_red(bgr, s_min=100, v_min=80)),
        ("excess_red_m15", mask_excess_red(bgr, margin=15)),
        ("excess_red_m25", mask_excess_red(bgr, margin=25)),
        ("excess_red_m40", mask_excess_red(bgr, margin=40)),
        ("lab_a150", mask_lab_red(bgr, a_min=150)),
        ("lab_a165", mask_lab_red(bgr, a_min=165)),
        ("ycrcb_cr150", mask_ycrcb_red(bgr, cr_min=150)),
        ("ycrcb_cr165", mask_ycrcb_red(bgr, cr_min=165)),
        ("hsv_tuned+morph", mask_hsv_red_tuned(bgr)),
        ("combined_hsv+excess+morph", mask_combined(bgr)),
        ("recommended", mask_gates_recommended(bgr)),
    ]

    panels = []
    for name, mask in variants:
        out = OUT_DIR / f"{name}.png"
        save_label(out, to_bw(mask), name, bgr)
        panels.append(cv2.imread(str(out.with_name(out.stem + "_panel.png"))))
        white_px = int(np.count_nonzero(mask))
        print(f"  {name}: {white_px} white pixels ({100 * white_px / (h * w):.2f}%)")

    grid_path = OUT_DIR / "comparison_grid.png"
    grid = build_comparison_grid(panels, cols=2)
    cv2.imwrite(str(grid_path), grid)
    print(f"\nWrote masks and panels to {OUT_DIR}")
    print(f"Comparison grid: {grid_path}")
    print("\nRecommended starting point: recommended.png")
    print("Tune in scripts/experiment_gate_mask.py (HSV bounds, margin, morph kernel sizes).")


if __name__ == "__main__":
    main()
