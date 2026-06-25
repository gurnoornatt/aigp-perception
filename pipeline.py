"""
GatePerceptionPipeline

Full inference pipeline from raw camera frame to EKF-fused drone position.

    Frame (RGB)
        │
        ▼
    UNet  →  binary mask  (512×512 → resized to camera resolution)
        │
        ▼
    extract_gate_corners  →  4 pixel corners [TL, TR, BR, BL]
        │
        ▼
    solve_gate_pnp  →  tvec_body (gate position in drone body frame)
        │
        ▼
    DroneEKF.update  →  smoothed position + velocity estimate

Usage:
    pipe = GatePerceptionPipeline("checkpoints/best.pt", device="cuda")

    # Camera loop at 30 Hz:
    result = pipe.step(frame_rgb)
    if result:
        print("position:", result["position"])   # [x, y, z] body frame, metres
        print("velocity:", result["velocity"])   # [vx, vy, vz] m/s

    # IMU loop at 200 Hz (can be called between camera frames):
    pipe.imu_predict(accel_xyz_body, dt_seconds)
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import UNet
from perception.corners import extract_gate_corners
from perception.ekf import DroneEKF
from perception.pnp import solve_gate_pnp


class GatePerceptionPipeline:
    """
    Args:
        model_path:    Path to a trained UNet checkpoint (best.pt).
        device:        "cuda" or "cpu".
        img_size:      Model input resolution. Must match training (default 512).
        threshold:     Sigmoid cutoff for mask binarization (default 0.5).
        base_filters:  Must match the value used during training (default 64).
        bilinear:      Must match the value used during training (default True).
        max_reproj_px: Discard PnP solutions with reprojection error above this.
                       Keeps bad detections from corrupting the EKF state.
    """

    def __init__(
        self,
        model_path:    str | Path,
        device:        str   = "cpu",
        img_size:      int   = 512,
        threshold:     float = 0.5,
        base_filters:  int   = 64,
        bilinear:      bool  = True,
        max_reproj_px: float = 10.0,
    ):
        self.device        = torch.device(device)
        self.img_size      = img_size
        self.threshold     = threshold
        self.max_reproj_px = max_reproj_px

        self.model = UNet(
            n_channels=3, n_classes=1,
            bilinear=bilinear, base_filters=base_filters,
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval().to(self.device)

        self.ekf = DroneEKF()

    # ── Public API ────────────────────────────────────────────────────────

    def imu_predict(self, accel_body: np.ndarray, dt: float) -> None:
        """
        Propagate the EKF with IMU data.

        Call at IMU rate (~200 Hz) even between camera frames. The EKF uses
        acceleration to keep the position estimate fresh between PnP updates.

        accel_body: [ax, ay, az] in m/s², drone body frame.
        dt:         Seconds since last call.
        """
        self.ekf.predict(accel_body, dt)

    def step(self, frame: np.ndarray) -> dict | None:
        """
        Process one camera frame through the full pipeline.

        frame: (H, W, 3) uint8 RGB image at the camera's native resolution.
               Must match the resolution the camera matrix was calibrated for.

        Returns a dict on success:
            mask          — (H, W) uint8 binary mask at frame resolution
            corners_px    — (4, 2) float64: [TL, TR, BR, BL] in pixels
            pnp           — full output of solve_gate_pnp (tvec, rvec, error)
            ekf_accepted  — True if EKF accepted the measurement (not an outlier)
            position      — (3,) float64 EKF position [x, y, z] body frame, m
            velocity      — (3,) float64 EKF velocity [vx, vy, vz] body frame, m/s

        Returns None when:
            - No gate is visible in the frame
            - Corner extraction fails (mask too small or noisy)
            - PnP solver does not converge
            - PnP reprojection error exceeds max_reproj_px
        """
        # 1. Segment the gate
        mask = self._infer_mask(frame)

        # 2. Fit corners to the mask
        corners_px = extract_gate_corners(mask)
        if corners_px is None:
            return None

        # 3. Solve pose from corners
        pnp = solve_gate_pnp(corners_px)
        if pnp is None:
            return None
        if pnp["reprojection_error_px"] > self.max_reproj_px:
            return None

        # 4. Fuse with EKF
        ekf_accepted = self.ekf.update(pnp["tvec_body"])

        return {
            "mask":         mask,
            "corners_px":   corners_px,
            "pnp":          pnp,
            "ekf_accepted": ekf_accepted,
            "position":     self.ekf.position,
            "velocity":     self.ekf.velocity,
        }

    # ── Private ───────────────────────────────────────────────────────────

    def _infer_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Run UNet inference and return a binary mask at the original frame resolution.

        The model runs at img_size × img_size internally. The result is scaled
        back to the original (H, W) using nearest-neighbour interpolation so
        the mask corners align with the camera matrix pixel coordinates.
        """
        orig_h, orig_w = frame.shape[:2]

        img = Image.fromarray(frame).resize((self.img_size, self.img_size), Image.BILINEAR)
        tensor = (
            torch.from_numpy(np.array(img))
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .div(255.0)
            .to(self.device)
        )

        with torch.inference_mode():
            prob = torch.sigmoid(self.model(tensor)).squeeze().cpu().numpy()

        mask_small = (prob > self.threshold).astype(np.uint8) * 255
        return np.array(
            Image.fromarray(mask_small).resize((orig_w, orig_h), Image.NEAREST)
        )
