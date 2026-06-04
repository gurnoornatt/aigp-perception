import unittest
import numpy as np
import cv2

from perception.pnp import solve_gate_pnp, GATE_CORNERS_3D
from perception.config import CAMERA_MATRIX, DIST_COEFFS


def make_fake_corners(distance_m: float = 5.0, offset_x: float = 0.0) -> np.ndarray:
    """
    Project the real 3D gate corners onto the image from a known position.
    This gives us perfect 2D corners we can use to test PnP.
    """
    # Put the gate 'distance_m' meters in front of the camera
    rvec = np.zeros((3, 1))
    tvec = np.array([[offset_x], [0.0], [distance_m]])

    projected, _ = cv2.projectPoints(
        GATE_CORNERS_3D, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS
    )
    return projected.reshape(4, 2)


class TestPnP(unittest.TestCase):
    def test_pnp_returns_result_for_valid_corners(self):
        corners = make_fake_corners(distance_m=5.0)
        result = solve_gate_pnp(corners)
        self.assertIsNotNone(result)

    def test_pnp_reprojection_error_is_low(self):
        corners = make_fake_corners(distance_m=5.0)
        result = solve_gate_pnp(corners)
        # Perfect synthetic corners should reproject with near-zero error
        self.assertLess(result["reprojection_error_px"], 1.0)

    def test_pnp_distance_is_approximately_correct(self):
        true_distance = 5.0
        corners = make_fake_corners(distance_m=true_distance)
        result = solve_gate_pnp(corners)
        # tvec_camera[2] is the forward (Z) distance to the gate
        estimated_distance = result["tvec_camera"][2]
        self.assertAlmostEqual(estimated_distance, true_distance, delta=0.1)

    def test_pnp_closer_gate_gives_smaller_z(self):
        corners_near = make_fake_corners(distance_m=3.0)
        corners_far  = make_fake_corners(distance_m=8.0)
        result_near = solve_gate_pnp(corners_near)
        result_far  = solve_gate_pnp(corners_far)
        self.assertLess(result_near["tvec_camera"][2], result_far["tvec_camera"][2])


if __name__ == "__main__":
    unittest.main()
