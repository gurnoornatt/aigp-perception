from __future__ import annotations

import unittest
import numpy as np
import cv2

from perception.corners import extract_corners_from_mask


def make_gate_mask(
    x1: int, y1: int, x2: int, y2: int,
    thickness: int = 8,
    w: int = 640,
    h: int = 360,
) -> np.ndarray:
    """Hollow rectangle gate mask at 640×360 (matches sim resolution)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness)
    return mask


class TestCornersLSD(unittest.TestCase):

    def test_returns_four_corners(self):
        mask = make_gate_mask(260, 130, 380, 230)
        corners = extract_corners_from_mask(mask)
        self.assertIsNotNone(corners)
        self.assertEqual(corners.shape, (4, 2))

    def test_corner_dtype_is_float32(self):
        mask = make_gate_mask(260, 130, 380, 230)
        corners = extract_corners_from_mask(mask)
        self.assertIsNotNone(corners)
        self.assertEqual(corners.dtype, np.float32)

    def test_corner_order_tl_tr_br_bl(self):
        mask = make_gate_mask(260, 130, 380, 230)
        corners = extract_corners_from_mask(mask)
        self.assertIsNotNone(corners)
        tl, tr, br, bl = corners
        self.assertLess(tl[0], tr[0],    "TL.x should be left of TR.x")
        self.assertLess(tl[1], bl[1],    "TL.y should be above BL.y")
        self.assertGreater(br[0], bl[0], "BR.x should be right of BL.x")
        self.assertGreater(br[1], tr[1], "BR.y should be below TR.y")

    def test_corners_approximate_rect_bounds(self):
        mask = make_gate_mask(260, 130, 380, 230)
        corners = extract_corners_from_mask(mask)
        self.assertIsNotNone(corners)
        tl, tr, br, bl = corners
        # TL should be near (260, 130) within 15px tolerance
        self.assertAlmostEqual(float(tl[0]), 260, delta=15)
        self.assertAlmostEqual(float(tl[1]), 130, delta=15)
        # BR should be near (380, 230)
        self.assertAlmostEqual(float(br[0]), 380, delta=15)
        self.assertAlmostEqual(float(br[1]), 230, delta=15)

    def test_returns_none_for_empty_mask(self):
        mask = np.zeros((360, 640), dtype=np.uint8)
        corners = extract_corners_from_mask(mask)
        self.assertIsNone(corners)

    def test_returns_none_for_tiny_mask(self):
        # Gate smaller than min_area_px=500 (e.g. 10×10 = 100 px²)
        mask = np.zeros((360, 640), dtype=np.uint8)
        cv2.rectangle(mask, (315, 175), (325, 185), 255, 2)
        corners = extract_corners_from_mask(mask)
        self.assertIsNone(corners)

    def test_small_gate_detected(self):
        # 40×40 px gate — small but above area threshold with thickness=5
        mask = make_gate_mask(300, 160, 340, 200, thickness=5)
        corners = extract_corners_from_mask(mask)
        self.assertIsNotNone(corners)
        self.assertEqual(corners.shape, (4, 2))

    def test_fallback_triggers_for_solid_fill(self):
        # Solid filled rectangle — LSD finds no internal lines, fallback to minAreaRect
        mask = np.zeros((360, 640), dtype=np.uint8)
        cv2.rectangle(mask, (260, 130), (380, 230), 255, -1)
        corners = extract_corners_from_mask(mask)
        self.assertIsNotNone(corners)
        self.assertEqual(corners.shape, (4, 2))

    def test_large_gate_detected(self):
        # Large gate near full frame
        mask = make_gate_mask(50, 30, 590, 330, thickness=12)
        corners = extract_corners_from_mask(mask)
        self.assertIsNotNone(corners)

    def test_off_center_gate(self):
        # Gate shifted to left side of frame
        mask = make_gate_mask(30, 80, 200, 280)
        corners = extract_corners_from_mask(mask)
        self.assertIsNotNone(corners)
        tl, tr, br, bl = corners
        # All corners should be in the left half
        self.assertLess(float(tl[0]), 320)
        self.assertLess(float(tr[0]), 320)


if __name__ == "__main__":
    unittest.main()
