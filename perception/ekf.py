import numpy as np


class DroneEKF:
    """
    Extended Kalman Filter fusing PnP position (30 Hz) with IMU (high rate).

    State vector (6 values):
        [x, y, z, vx, vy, vz]
         position (m)  velocity (m/s)   — all in body/course frame

    Usage:
        ekf = DroneEKF()
        # Call every IMU message (~200 Hz):
        ekf.predict(accel_xyz, dt)
        # Call every time PnP gives a new position (~30 Hz):
        ekf.update(pnp_position_xyz)
        # Read current best estimate:
        pos = ekf.position
        vel = ekf.velocity
    """

    def __init__(self):
        self.state = np.zeros(6)       # [x, y, z, vx, vy, vz]

        # P: how uncertain we are about the state (start uncertain)
        self.P = np.eye(6) * 1.0

        # Q: how much we trust the IMU prediction (bigger = trust less)
        self.Q = np.diag([0.01, 0.01, 0.01,    # position noise
                          0.05, 0.05, 0.05])   # velocity noise — tightened for real IMU

        # R: how much we trust PnP measurements (bigger = trust less)
        self.R = np.eye(3) * 0.25

        # Outlier rejection threshold (chi-squared, 3 DOF, 99% = 11.3)
        self.chi2_threshold = 11.3

    def predict(self, accel_body: np.ndarray, dt: float) -> None:
        """
        Run every IMU message. Propagates state forward using acceleration.

        accel_body: [ax, ay, az] in m/s², body frame
        dt: time since last predict call in seconds
        """
        x, y, z, vx, vy, vz = self.state

        # State transition matrix F (how state evolves over dt)
        F = np.eye(6)
        F[0, 3] = dt   # x  += vx * dt
        F[1, 4] = dt   # y  += vy * dt
        F[2, 5] = dt   # z  += vz * dt

        # Update position and velocity
        new_pos = np.array([x, y, z]) + np.array([vx, vy, vz]) * dt
        new_vel = np.array([vx, vy, vz]) + accel_body * dt
        self.state = np.concatenate([new_pos, new_vel])

        # Grow uncertainty
        self.P = F @ self.P @ F.T + self.Q

    def update(self, pnp_position: np.ndarray) -> bool:
        """
        Correct state when PnP gives a new position measurement.

        pnp_position: [x, y, z] in meters
        Returns True if measurement was accepted, False if rejected as outlier.
        """
        # H: observation matrix — we only directly observe position, not velocity
        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        # Innovation: difference between what PnP says and what we predicted
        innovation = pnp_position - H @ self.state

        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        S_inv = np.linalg.inv(S)

        # Outlier rejection: if PnP is way off from prediction, ignore it
        mahalanobis = float(innovation.T @ S_inv @ innovation)
        if mahalanobis > self.chi2_threshold:
            return False   # bad measurement, skip

        # Kalman gain: how much to trust PnP vs our prediction
        K = self.P @ H.T @ S_inv

        # Correct state and shrink uncertainty
        self.state = self.state + K @ innovation
        self.P = (np.eye(6) - K @ H) @ self.P
        return True

    @property
    def position(self) -> np.ndarray:
        return self.state[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:].copy()
