"""
Extract ordered gate corners from a binary segmentation mask.

UNet produces a pixel mask. PnP needs exactly 4 image-plane points in a
specific order: top-left, top-right, bottom-right, bottom-left. This module
bridges the two using OpenCV contour analysis.

Gate masks come in two shapes depending on what was segmented:

    Ring (frame) mask — the physical gate structure is a hollow square.
        The outer boundary is the outside of the frame tube.
        The inner boundary (hole) is the gate's inner aperture.
        PnP needs the INNER corners (matching GATE_INNER_SIZE_M).
        → Use RETR_CCOMP to see both outer and inner contours.
        → If a hole exists, extract corners from the INNER contour.

    Solid mask — the gate aperture is filled as a solid region.
        The outer boundary IS the gate's inner aperture.
        → Extract corners from the outer contour directly.

Strategy (in order):
    1. Find outer contour + detect if it has a child hole (ring mask).
    2. If ring: use inner hole contour for corners → matches inner gate size.
    3. If solid: use outer contour with approxPolyDP → 4 corners directly.
    4. Fallback: minAreaRect on whichever contour we're working with.
"""

import cv2
import numpy as np


def extract_gate_corners(
    mask: np.ndarray,
    min_area_px: int   = 500,
    poly_epsilon: float = 0.05,
) -> np.ndarray | None:
    """
    Find the gate in a binary mask and return 4 ordered inner-aperture corners.

    Args:
        mask:         Binary mask (H, W) uint8 — any nonzero pixel = gate.
        min_area_px:  Ignore contours with area below this (noise filter).
        poly_epsilon: approxPolyDP precision as a fraction of contour perimeter.

    Returns:
        np.ndarray of shape (4, 2) float64 in pixel coordinates, ordered:
            [top-left, top-right, bottom-right, bottom-left]
        None if no valid gate contour is found.

    Note:
        Pixel coordinates must be in the same space as the camera matrix used
        by PnP (i.e. at the original camera image resolution, not model input).
    """
    binary = (mask > 0).astype(np.uint8) * 255

    # RETR_CCOMP: two-level hierarchy — outer contours at level 0,
    # inner holes at level 1 (child index in hierarchy[:,2]).
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find the largest level-0 (outer) contour.
    outer_indices = [i for i, h in enumerate(hierarchy[0]) if h[3] == -1]  # no parent
    if not outer_indices:
        return None

    best_outer = max(outer_indices, key=lambda i: cv2.contourArea(contours[i]))
    if cv2.contourArea(contours[best_outer]) < min_area_px:
        return None

    # Check whether the outer contour has a child hole (ring-shaped mask).
    child_idx = hierarchy[0][best_outer][2]
    if child_idx != -1 and cv2.contourArea(contours[child_idx]) >= min_area_px:
        # Ring mask: use the INNER contour (the gate aperture hole).
        # These corners correspond to GATE_INNER_SIZE_M in the 3D model.
        target = contours[child_idx]
    else:
        # Solid mask: the outer contour is the aperture directly.
        target = contours[best_outer]

    pts = _contour_to_quad(target, poly_epsilon)
    return _order_corners(pts)


def _contour_to_quad(contour: np.ndarray, poly_epsilon: float) -> np.ndarray:
    """Reduce a contour to 4 points via approxPolyDP or minAreaRect fallback."""
    perimeter = cv2.arcLength(contour, closed=True)
    approx    = cv2.approxPolyDP(contour, epsilon=poly_epsilon * perimeter, closed=True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float64)
    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype(np.float64)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Sort 4 unordered points into (top-left, top-right, bottom-right, bottom-left).

    In image coordinates y increases downward, so "top" means smaller y.
    Sort by y → smallest two y values are top pair; sort each pair by x.
    """
    pts    = pts[np.argsort(pts[:, 1])]
    top    = pts[:2][np.argsort(pts[:2, 0])]     # TL, TR
    bottom = pts[2:][np.argsort(pts[2:, 0])]     # BL, BR
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float64)
    #                 TL       TR       BR         BL
