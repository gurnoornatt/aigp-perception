import numpy as np

# Camera intrinsics — from VADR-TS-002 spec, exact values
CAMERA_MATRIX = np.array([
    [320.0,   0.0, 320.0],
    [  0.0, 320.0, 180.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

DIST_COEFFS = np.zeros(5)  # pinhole, no distortion (spec confirmed)

# Gate geometry
GATE_INNER_SIZE_M = 1.5    # 1500mm inner square (the one PnP uses)
GATE_OUTER_SIZE_M = 2.7    # 2700mm outer square (for reference)

# Camera mount
CAMERA_TILT_DEG = 20.0     # camera tilts upward 20 degrees from body

# Image dimensions
IMAGE_WIDTH  = 640
IMAGE_HEIGHT = 360
IMAGE_FPS    = 30
