"""
Tests for perception.corners — mask-to-corners extraction.

We test with synthetic masks (drawn rectangles) so we know the exact expected
corner locations and can verify the ordering contract (TL, TR, BR, BL).
"""

import unittest
import numpy as np
import cv2

from perception.corners import extract_gate_corners, _order_corners


def make_rect_mask(h: int, w: int, top: int, left: int, bottom: int, right: int) -> np.ndarray:
    """Draw a filled white rectangle on a black mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return mask


class TestExtractGateCorners(unittest.TestCase):
    def test_returns_none_for_empty_mask(self):
        mask = np.zeros((360, 640), dtype=np.uint8)
        self.assertIsNone(extract_gate_corners(mask))

    def test_returns_none_for_tiny_contour(self):
        mask = np.zeros((360, 640), dtype=np.uint8)
        mask[100:105, 100:105] = 255   # 25px² — below min_area_px=500
        self.assertIsNone(extract_gate_corners(mask))

    def test_returns_four_points(self):
        mask = make_rect_mask(360, 640, 80, 200, 280, 440)
        corners = extract_gate_corners(mask)
        self.assertIsNotNone(corners)
        self.assertEqual(corners.shape, (4, 2))

    def test_ordering_tl_tr_br_bl(self):
        # Rectangle with known corners: TL=(200,80), TR=(440,80), BR=(440,280), BL=(200,280)
        mask = make_rect_mask(360, 640, top=80, left=200, bottom=280, right=440)
        corners = extract_gate_corners(mask)
        self.assertIsNotNone(corners)
        tl, tr, br, bl = corners

        # Top row should have smaller y than bottom row
        self.assertLess(tl[1], bl[1])
        self.assertLess(tr[1], br[1])

        # Left column should have smaller x than right column
        self.assertLess(tl[0], tr[0])
        self.assertLess(bl[0], br[0])

    def test_approximate_corner_locations(self):
        """Extracted corners should be within a few pixels of the true rect corners."""
        mask = make_rect_mask(360, 640, top=80, left=200, bottom=280, right=440)
        corners = extract_gate_corners(mask)
        self.assertIsNotNone(corners)
        tl, tr, br, bl = corners

        tol = 5  # pixels
        self.assertAlmostEqual(tl[0], 200, delta=tol)
        self.assertAlmostEqual(tl[1],  80, delta=tol)
        self.assertAlmostEqual(tr[0], 440, delta=tol)
        self.assertAlmostEqual(tr[1],  80, delta=tol)
        self.assertAlmostEqual(br[0], 440, delta=tol)
        self.assertAlmostEqual(br[1], 280, delta=tol)
        self.assertAlmostEqual(bl[0], 200, delta=tol)
        self.assertAlmostEqual(bl[1], 280, delta=tol)


class TestOrderCorners(unittest.TestCase):
    def test_already_ordered(self):
        pts = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float64)
        result = _order_corners(pts)
        np.testing.assert_array_equal(result, pts)

    def test_shuffled_input(self):
        tl = [10.0, 10.0]
        tr = [50.0, 10.0]
        br = [50.0, 50.0]
        bl = [10.0, 50.0]
        shuffled = np.array([br, tl, bl, tr], dtype=np.float64)
        result = _order_corners(shuffled)
        np.testing.assert_array_almost_equal(result[0], tl)
        np.testing.assert_array_almost_equal(result[1], tr)
        np.testing.assert_array_almost_equal(result[2], br)
        np.testing.assert_array_almost_equal(result[3], bl)


if __name__ == "__main__":
    unittest.main()
