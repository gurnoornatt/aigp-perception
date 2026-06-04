# aigp-perception

Perception stack for AI Grand Prix — gate detection, PnP pose estimation, and EKF state estimation.

## What's in here

| File | What it does |
|------|-------------|
| `perception/config.py` | Camera intrinsics, gate size, constants from spec |
| `perception/pnp.py` | Takes 4 gate corners → drone position via solvePnP |
| `perception/ekf.py` | Fuses PnP (30 Hz) with IMU (200 Hz) for smooth position |
| `perception/result.py` | `MultiGatePerceptionResult` — the contract handed to planning |
| `tests/test_pnp.py` | Tests for PnP solver |
| `tests/test_ekf.py` | Tests for EKF |

## Team ownership

- **Faadil** — U-Net gate segmentation, corner extraction → outputs 4 pixel corners
- **Noor** — PnP solver, EKF, packages `MultiGatePerceptionResult`

## Setup

```bash
pip install numpy opencv-python
```

## Run tests

```bash
python -m unittest discover tests
```
