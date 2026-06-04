import numpy as np
import cv2

from .config import CAMERA_MATRIX, DIST_COEFFS, GATE_INNER_SIZE_M, CAMERA_TILT_DEG

# Gate corners in the gate's own 3D frame, centered at origin, Z=0 plane.
# Order matches what Faadil's corner detector will give us:
#   0: top-left   1: top-right   2: bottom-right   3: bottom-left
_h = GATE_INNER_SIZE_M / 2.0
GATE_CORNERS_3D = np.array([
    [-_h,  _h, 0.0],
    [ _h,  _h, 0.0],
    [ _h, -_h, 0.0],
    [-_h, -_h, 0.0],
], dtype=np.float64)

# Rotation to go from camera-frame vector → body-frame vector.
# Camera is tilted up 20° from body (pitched up), so we undo that tilt.
_t = np.radians(CAMERA_TILT_DEG)
_R_body_from_cam = np.array([
    [ np.cos(_t), 0.0, np.sin(_t)],
    [        0.0, 1.0,        0.0],
    [-np.sin(_t), 0.0, np.cos(_t)],
], dtype=np.float64)


def solve_gate_pnp(corners_2d: np.ndarray) -> dict | None:
    """
    Solve drone pose from 4 gate corners.

    corners_2d: shape (4, 2) pixel coordinates in image.
                Order must be: top-left, top-right, bottom-right, bottom-left.

    Returns a dict with:
        tvec_camera  - gate origin in camera frame (3,)
        rvec         - rotation vector camera->gate (3,)
        tvec_body    - gate origin in body frame (3,) [forward, right, down]
        reprojection_error_px - mean pixel error (lower = better)

    Returns None if the solve failed.
    """
    ok, rvec, tvec = cv2.solvePnP(
        GATE_CORNERS_3D,
        corners_2d.astype(np.float64),
        CAMERA_MATRIX,
        DIST_COEFFS,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None

    tvec_camera = tvec.flatten()
    rvec_flat   = rvec.flatten()

    # Rotate into body frame to account for the 20° camera tilt
    tvec_body = _R_body_from_cam @ tvec_camera

    # Reprojection: re-project 3D corners back to pixels and measure error
    projected, _ = cv2.projectPoints(
        GATE_CORNERS_3D, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS
    )
    projected = projected.reshape(4, 2)
    reprojection_error = float(np.linalg.norm(corners_2d - projected, axis=1).mean())

    return {
        "tvec_camera":           tvec_camera,
        "rvec":                  rvec_flat,
        "tvec_body":             tvec_body,
        "reprojection_error_px": reprojection_error,
    }


def draw_reprojection(image: np.ndarray, corners_2d: np.ndarray, pnp_result: dict) -> np.ndarray:
    """
    Draw detected corners (green) and reprojected corners (red) on a copy of image.
    Use this to visually verify PnP is working before plugging into EKF.
    """
    img = image.copy()
    rvec = pnp_result["rvec"].reshape(3, 1)
    tvec = pnp_result["tvec_camera"].reshape(3, 1)

    projected, _ = cv2.projectPoints(
        GATE_CORNERS_3D, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS
    )
    projected = projected.reshape(4, 2)

    for pt in corners_2d:
        cv2.circle(img, tuple(pt.astype(int)), 6, (0, 255, 0), -1)   # detected = green

    for pt in projected:
        cv2.circle(img, tuple(pt.astype(int)), 4, (0, 0, 255), -1)   # reprojected = red

    err = pnp_result["reprojection_error_px"]
    cv2.putText(img, f"reproj err: {err:.1f}px", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img
