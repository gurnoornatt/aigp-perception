from dataclasses import dataclass


@dataclass(frozen=True)
class MultiGatePerceptionResult:
    frame_id: int
    sim_time_ns: int

    # What perception saw
    visible_gate_ids: tuple          # e.g. ("gate_1",) or ("gate_1", "gate_2")
    used_corner_count: int           # corners used in PnP solve (max 8 if 2 gates)

    # Pose solved from all visible corners
    camera_course_position_m: tuple  # (x, y, z) drone position in meters
    camera_course_rvec: tuple        # (rx, ry, rz) rotation vector

    # Planning target — next gate to fly through
    next_gate_id: str
    next_gate_course_position_m: tuple   # where the gate is in the world
    vector_to_next_gate_m: tuple         # direction from drone to gate
    distance_to_next_gate_m: float       # how far away in meters

    # Quality metrics — planning uses these to decide how much to trust us
    reprojection_error_px: float     # lower is better, < 5px is good
    confidence: float                # 0.0 to 1.0
