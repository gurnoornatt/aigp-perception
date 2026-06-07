from __future__ import annotations

import cv2
import numpy as np


def extract_corners_from_mask(
    binary_mask: np.ndarray,
    extend_factor: float = 5 / 3,
    bbox_margin_ratio: float = 0.20,
    min_area_px: float = 500.0,
) -> np.ndarray | None:
    """
    Extract 4 gate corners from a binary segmentation mask.

    Primary path: LSD line segments → extend → pairwise intersections → bbox filter.
    Fallback: cv2.minAreaRect when LSD finds fewer than 4 lines.

    Args:
        binary_mask: uint8 (H, W) with values 0 or 255.
        extend_factor: how much to extend each LSD segment from its midpoint.
                       5/3 ensures segments cross at corners even when LSD stops short.
        bbox_margin_ratio: fraction of gate bbox side to expand before filtering
                           intersection candidates.
        min_area_px: minimum contour area in pixels — gates smaller than this are
                     too far away / noisy to solve PnP reliably.

    Returns:
        (4, 2) float32 array in TL, TR, BR, BL order, or None if extraction failed.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_px:
        return None

    corners = _lsd_corners(binary_mask, largest, extend_factor, bbox_margin_ratio)
    if corners is None:
        corners = _minAreaRect_corners(largest)
    return corners


# ── LSD path ──────────────────────────────────────────────────────────────────

def _lsd_corners(
    mask: np.ndarray,
    largest_contour: np.ndarray,
    extend_factor: float,
    bbox_margin_ratio: float,
) -> np.ndarray | None:
    lsd = cv2.createLineSegmentDetector(0)  # LSD_REFINE_NONE — fastest
    lines, _, _, _ = lsd.detect(mask)
    if lines is None or len(lines) < 4:
        return None

    segs = lines.reshape(-1, 4)  # (N, 4): [x1, y1, x2, y2]

    # Extend each segment by extend_factor from its midpoint
    extended = np.empty_like(segs)
    for i, (x1, y1, x2, y2) in enumerate(segs):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = (x2 - x1) * extend_factor / 2, (y2 - y1) * extend_factor / 2
        extended[i] = [cx - dx, cy - dy, cx + dx, cy + dy]

    # Compute all pairwise intersections
    H, W = mask.shape[:2]
    candidates = []
    for i in range(len(extended)):
        for j in range(i + 1, len(extended)):
            pt = _intersect(extended[i], extended[j])
            if pt is not None and 0 <= pt[0] < W and 0 <= pt[1] < H:
                candidates.append(pt)

    if len(candidates) < 4:
        return None

    pts = np.array(candidates, dtype=np.float32)

    # Filter to expanded gate bounding box
    x, y, bw, bh = cv2.boundingRect(largest_contour)
    margin = bbox_margin_ratio * max(bw, bh)
    x0, y0 = x - margin, y - margin
    x1_, y1_ = x + bw + margin, y + bh + margin
    mask_in = (pts[:, 0] >= x0) & (pts[:, 0] <= x1_) & \
              (pts[:, 1] >= y0) & (pts[:, 1] <= y1_)
    pts = pts[mask_in]

    if len(pts) < 4:
        return None

    return _order_corners_tl_tr_br_bl(pts)


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    """Cramer's rule intersection of two line segments (extended to infinite lines)."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None  # parallel
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return np.array([x, y], dtype=np.float32)


# ── minAreaRect fallback ───────────────────────────────────────────────────────

def _minAreaRect_corners(contour: np.ndarray) -> np.ndarray | None:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)  # (4, 2)
    return _order_corners_tl_tr_br_bl(box)


# ── Corner ordering ────────────────────────────────────────────────────────────

def _order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """
    Given N >= 4 candidate points, return the 4 extreme corners in TL, TR, BR, BL order.

    Uses coordinate sum/difference:
        TL = min(x + y)   BR = max(x + y)
        TR = max(x - y)   BL = min(x - y)
    """
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)
