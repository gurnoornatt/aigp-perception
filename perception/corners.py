"""
Extract ordered gate corners from a binary segmentation mask.

UNet produces a pixel mask. PnP needs exactly 4 image-plane points in a
specific order: top-left, top-right, bottom-right, bottom-left. This module
bridges the two using OpenCV contour analysis.

Two-step strategy:
    1. approxPolyDP — tries to simplify the largest contour to a quadrilateral.
       Works well when the gate is a clean filled rectangle in the mask.
    2. minAreaRect  — fallback for ring-shaped masks (physical gate frame) or
       non-square approximations. Fits the tightest rotated rectangle around
       all contour points and returns its four corners.
"""

import cv2
import numpy as np


def extract_gate_corners(
    mask: np.ndarray,
    min_area_px: int   = 500,
    poly_epsilon: float = 0.05,
) -> np.ndarray | None:
    """
    Find the largest gate-like region in a binary mask and return 4 corners.

    Args:
        mask:         Binary mask (H, W) uint8 — any nonzero pixel = gate.
        min_area_px:  Ignore contours smaller than this. Filters sensor noise
                      and stray activations from the segmentation model.
        poly_epsilon: approxPolyDP precision as a fraction of contour perimeter.
                      Larger → coarser approximation (fewer vertices).

    Returns:
        np.ndarray of shape (4, 2) float64 in pixel coordinates, ordered:
            [top-left, top-right, bottom-right, bottom-left]
        None if no valid gate contour is found.

    Note:
        Pixel coordinates returned here must be in the same coordinate space
        as the camera matrix used by PnP. If the mask was produced at a
        different resolution than the camera image, resize before calling.
    """
    binary = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_px:
        return None

    perimeter = cv2.arcLength(largest, closed=True)
    approx    = cv2.approxPolyDP(largest, epsilon=poly_epsilon * perimeter, closed=True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float64)
    else:
        # minAreaRect always gives exactly 4 corners for any convex shape.
        rect = cv2.minAreaRect(largest)
        pts  = cv2.boxPoints(rect).astype(np.float64)

    return _order_corners(pts)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Sort 4 unordered points into (top-left, top-right, bottom-right, bottom-left).

    In image coordinates, y increases downward, so "top" means smaller y.

    Sort by y → the two points with the smallest y values are the top pair.
    Within each pair sort by x → smaller x is left.
    """
    pts    = pts[np.argsort(pts[:, 1])]               # ascending y
    top    = pts[:2][np.argsort(pts[:2, 0])]           # TL (small x), TR (large x)
    bottom = pts[2:][np.argsort(pts[2:, 0])]           # BL (small x), BR (large x)
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float64)
    #                 TL       TR       BR         BL
