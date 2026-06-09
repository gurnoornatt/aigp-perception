from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

from .corners import extract_corners_from_mask
from .ekf import DroneEKF
from .pnp import solve_gate_pnp
from .result import MultiGatePerceptionResult

# ── Module-level EKF + IMU state (persists across frames within a session) ────

_ekf: DroneEKF | None = None
_last_time_ns: int = 0
_imu_accel: np.ndarray = np.zeros(3)   # latest body-frame accel from HIGHRES_IMU


def reset_ekf() -> None:
    """Call at flight start, after a long frame gap, or between separate flights."""
    global _ekf, _last_time_ns
    _ekf = DroneEKF()
    _last_time_ns = 0


def update_imu(accel_body: np.ndarray) -> None:
    """Feed latest IMU acceleration. Called from MAVLink thread at ~200 Hz."""
    global _imu_accel
    _imu_accel = accel_body.copy()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def detect_gates(
    frame_bgr: np.ndarray,
    frame_id: int,
    sim_time_ns: int,
    model: nn.Module,
    device: str,
    input_size: int = 384,
    threshold: float = 0.5,
) -> MultiGatePerceptionResult | None:
    """
    Full perception pipeline: BGR frame → MultiGatePerceptionResult.

    Returns None when no gate is confidently detected (empty mask, corner
    extraction failed, or PnP solve failed).

    Args:
        frame_bgr: uint8 (H, W, 3) BGR frame — nominally 640×360.
        frame_id:  monotonic frame counter from the UDP stream.
        sim_time_ns: simulator timestamp in nanoseconds.
        model:     GateNet or simple UNet already on `device` and in eval mode.
        device:    'mps', 'cuda', or 'cpu'.
        input_size: model input resolution (384 for GateNet).
        threshold:  sigmoid threshold for binary mask binarisation.
    """
    global _ekf, _last_time_ns, _imu_accel

    # ── 0. Alpha-channel guard ────────────────────────────────────────────────
    if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        frame_bgr = frame_bgr[:, :, :3]

    H, W = frame_bgr.shape[:2]

    # ── 1. Inference ──────────────────────────────────────────────────────────
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img_t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0).to(device)

    is_half = next(model.parameters()).dtype == torch.float16
    if is_half:
        img_t = img_t.half()

    with torch.no_grad():
        output = model(img_t)
        logits = output[-1] if isinstance(output, list) else output
        # Always cast to float32 before numpy — mandatory on MPS with fp16 models
        prob_small = torch.sigmoid(logits).squeeze().cpu().float().numpy()

    # ── 2. Resize mask + prob back to frame dimensions ────────────────────────
    binary_small = (prob_small > threshold).astype(np.uint8) * 255
    binary_mask = cv2.resize(binary_small, (W, H), interpolation=cv2.INTER_NEAREST)
    prob_map = cv2.resize(prob_small, (W, H), interpolation=cv2.INTER_LINEAR)

    # ── 3. Find all gate candidates (sorted largest → smallest = closest → farthest)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gate_contours = [c for c in contours if cv2.contourArea(c) > 500]
    if not gate_contours:
        return None
    gate_contours.sort(key=cv2.contourArea, reverse=True)

    # ── 4 & 5. Corners + PnP for each candidate (up to 5 gates) ─────────────
    gate_results: list[tuple[int, dict, np.ndarray]] = []
    for i, contour in enumerate(gate_contours[:5]):
        single_mask = np.zeros_like(binary_mask)
        cv2.drawContours(single_mask, [contour], -1, 255, -1)
        corners = extract_corners_from_mask(single_mask)
        if corners is None:
            continue
        pnp = solve_gate_pnp(corners)
        if pnp is None:
            continue
        gate_results.append((i, pnp, corners))

    if not gate_results:
        return None

    # Nearest gate (largest contour = closest) drives EKF and result
    _, nearest_pnp, _ = gate_results[0]

    # ── 6. EKF update — IMU predict + PnP correct ─────────────────────────────
    if _ekf is None:
        _ekf = DroneEKF()

    # Reset if there has been a long gap (> 500 ms) between frames
    if _last_time_ns > 0 and (sim_time_ns - _last_time_ns) > 500_000_000:
        reset_ekf()

    dt = (sim_time_ns - _last_time_ns) / 1e9 if _last_time_ns > 0 else 1.0 / 30.0
    dt = float(np.clip(dt, 0.001, 0.5))
    _ekf.predict(_imu_accel, dt)          # real IMU accel (zeros until MAVLink wired)
    _ekf.update(nearest_pnp["tvec_body"])
    _last_time_ns = sim_time_ns

    smoothed_pos = _ekf.position  # (3,) in body frame

    # ── 7. Confidence scoring ─────────────────────────────────────────────────
    confidence = _compute_confidence(prob_map, binary_mask)

    # ── 8. Assemble result ────────────────────────────────────────────────────
    tvec = nearest_pnp["tvec_body"]
    distance = float(np.linalg.norm(tvec))
    unit_vec = tvec / (distance + 1e-9)
    visible_ids = tuple(f"gate_{i}" for i, _, _ in gate_results)

    return MultiGatePerceptionResult(
        frame_id=frame_id,
        sim_time_ns=sim_time_ns,
        visible_gate_ids=visible_ids,
        used_corner_count=4,
        camera_course_position_m=(0.0, 0.0, 0.0),  # no world-frame odometry in v1
        camera_course_rvec=tuple(float(v) for v in nearest_pnp["rvec"]),
        next_gate_id=visible_ids[0],
        next_gate_course_position_m=tuple(float(v) for v in smoothed_pos),
        vector_to_next_gate_m=tuple(float(v) for v in unit_vec),
        distance_to_next_gate_m=distance,
        reprojection_error_px=nearest_pnp["reprojection_error_px"],
        confidence=confidence,
    )


# ── Confidence ────────────────────────────────────────────────────────────────

def _compute_confidence(prob_map: np.ndarray, binary_mask: np.ndarray) -> float:
    """
    Composite confidence score in [0, 1] from four independent signals:
      0.4 × mean sigmoid probability inside the gate mask
      0.3 × shape regularity (aspect ratio of bounding box — square gate = 1.0)
      0.2 × single connected component (1 gate, not fragmented noise)
      0.1 × area ratio in expected range (gate should be 3–60 % of frame)
    """
    gate_pixels = prob_map[binary_mask > 0]
    mean_sig = float(gate_pixels.mean()) if len(gate_pixels) > 0 else 0.0

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        _, _, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        aspect = min(bw, bh) / (max(bw, bh) + 1e-6)
        shape_reg = float(np.clip(aspect, 0.0, 1.0))
    else:
        shape_reg = 0.0

    n_labels, _ = cv2.connectedComponents(binary_mask)
    single_comp = 1.0 if n_labels == 2 else 0.0  # 1 background + 1 gate

    area_ratio = float((binary_mask > 0).mean())
    area_ok = 1.0 if 0.03 <= area_ratio <= 0.60 else 0.0

    return float(
        0.4 * mean_sig
        + 0.3 * shape_reg
        + 0.2 * single_comp
        + 0.1 * area_ok
    )
