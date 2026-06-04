import unittest
import numpy as np

from perception.ekf import DroneEKF


class TestEKF(unittest.TestCase):
    def test_predict_moves_position_with_velocity(self):
        ekf = DroneEKF()
        ekf.state[3] = 1.0  # vx = 1 m/s
        ekf.predict(np.zeros(3), dt=1.0)
        # After 1 second at 1 m/s, x should be ~1.0
        self.assertAlmostEqual(ekf.position[0], 1.0, delta=0.01)

    def test_update_corrects_position(self):
        ekf = DroneEKF()
        # Start at (0,0,0), then tell it we're at (3,0,0)
        ekf.update(np.array([3.0, 0.0, 0.0]))
        # State should move toward (3,0,0)
        self.assertGreater(ekf.position[0], 0.0)

    def test_outlier_rejected(self):
        ekf = DroneEKF()
        # State is at (0,0,0) and certain (small P)
        ekf.P = np.eye(6) * 0.0001
        # Feed a measurement that is 1000m away — should be rejected
        accepted = ekf.update(np.array([1000.0, 0.0, 0.0]))
        self.assertFalse(accepted)
        # Position should barely move
        self.assertLess(ekf.position[0], 1.0)

    def test_predict_then_update_converges(self):
        ekf = DroneEKF()
        true_pos = np.array([2.0, 1.0, -1.5])
        # Run several predict+update cycles
        for _ in range(20):
            ekf.predict(np.zeros(3), dt=0.033)  # ~30 Hz
            ekf.update(true_pos + np.random.normal(0, 0.05, 3))
        # After 20 updates toward (2, 1, -1.5), should be close
        np.testing.assert_allclose(ekf.position, true_pos, atol=0.3)


if __name__ == "__main__":
    unittest.main()
