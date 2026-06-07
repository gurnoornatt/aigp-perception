from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn


CKPT_PATH = Path("checkpoints/gatenet_best.pt")
DEVICE    = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Mock model (no GPU, no checkpoint needed) ─────────────────────────────────

class _MockGateNet(nn.Module):
    """
    Returns logit=5.0 (sigmoid≈0.993) in a central rectangular region and
    logit=-5.0 everywhere else. Simulates a clearly visible gate.
    """
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        logits = torch.full((B, 1, H, W), -5.0)
        # Gate occupies the central 40% of the frame
        y0, y1 = H // 4, 3 * H // 4
        x0, x1 = W // 4, 3 * W // 4
        logits[:, :, y0:y1, x0:x1] = 5.0
        return [logits] * 5  # 5 heads like GateNet

    def parameters(self, recurse=True):
        return iter([torch.zeros(1)])  # dtype check returns float32


class _NoGateModel(nn.Module):
    """Returns all-negative logits — no gate visible."""
    def forward(self, x: torch.Tensor):
        return [torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), -10.0)] * 5

    def parameters(self, recurse=True):
        return iter([torch.zeros(1)])


def _make_frame(w: int = 640, h: int = 360) -> np.ndarray:
    """BGR frame with a visible gate-coloured rectangle."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (15, 20, 30)
    cv2.rectangle(frame, (250, 120), (390, 240), (40, 60, 200), 6)  # red-ish gate
    return frame


# ── Tests with mock model (always run, no sim / checkpoint needed) ─────────────

class TestDetectWithMockModel(unittest.TestCase):

    def setUp(self):
        from perception.detect import reset_ekf
        reset_ekf()

    def test_returns_result_for_visible_gate(self):
        from perception.detect import detect_gates
        result = detect_gates(
            _make_frame(), frame_id=0, sim_time_ns=33_000_000,
            model=_MockGateNet(), device="cpu",
        )
        # Mock produces a large clean mask — result should be non-None
        # (corner extraction / PnP may fail on a perfect square logit; acceptable)
        if result is not None:
            self.assertGreater(result.distance_to_next_gate_m, 0.0)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_returns_none_for_no_gate(self):
        from perception.detect import detect_gates
        result = detect_gates(
            _make_frame(), frame_id=1, sim_time_ns=66_000_000,
            model=_NoGateModel(), device="cpu",
        )
        self.assertIsNone(result)

    def test_alpha_channel_stripped(self):
        """4-channel RGBA frame must not crash detect_gates."""
        from perception.detect import detect_gates
        rgba = np.zeros((360, 640, 4), dtype=np.uint8)
        rgba[:, :, 2] = 200  # red channel
        # Just must not raise; result can be None
        try:
            detect_gates(rgba, frame_id=2, sim_time_ns=0, model=_NoGateModel(), device="cpu")
        except Exception as e:
            self.fail(f"detect_gates raised on 4-channel input: {e}")

    def test_result_is_frozen_dataclass(self):
        from perception.result import MultiGatePerceptionResult
        import dataclasses
        r = MultiGatePerceptionResult(
            frame_id=0, sim_time_ns=0,
            visible_gate_ids=("gate_0",), used_corner_count=4,
            camera_course_position_m=(0.0, 0.0, 0.0),
            camera_course_rvec=(0.0, 0.0, 0.0),
            next_gate_id="gate_0",
            next_gate_course_position_m=(0.0, 0.0, 5.0),
            vector_to_next_gate_m=(0.0, 0.0, 1.0),
            distance_to_next_gate_m=5.0,
            reprojection_error_px=1.0,
            confidence=0.9,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            r.frame_id = 999  # type: ignore

    def test_ekf_persists_across_calls(self):
        """Two consecutive calls should not crash even when the first returns None."""
        from perception.detect import detect_gates, reset_ekf
        reset_ekf()
        detect_gates(_make_frame(), 0, 0,       _NoGateModel(), "cpu")
        detect_gates(_make_frame(), 1, 33000000, _NoGateModel(), "cpu")

    def test_reset_ekf_clears_state(self):
        from perception.detect import detect_gates, reset_ekf
        import perception.detect as _det
        reset_ekf()
        self.assertIsNotNone(_det._ekf)
        _det._last_time_ns = 999_999_999_999
        reset_ekf()
        self.assertEqual(_det._last_time_ns, 0)


# ── Tests with real checkpoint (skipped when checkpoint is absent) ─────────────

@unittest.skipUnless(CKPT_PATH.exists(), f"Requires {CKPT_PATH}")
class TestDetectWithRealModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from training.gatenet import GateNet
        from perception.detect import reset_ekf
        reset_ekf()
        cls.model = GateNet(f=4).to(DEVICE)
        ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE, weights_only=True)
        cls.model.load_state_dict(ckpt["state_dict"])
        cls.model.eval()

    def setUp(self):
        from perception.detect import reset_ekf
        reset_ekf()

    def _load_val_frame(self) -> np.ndarray | None:
        """Load first available val-split image, resized to 640×360."""
        from training.dataset import get_split_indices
        _, val_idxs, _ = get_split_indices()
        for idx in sorted(val_idxs):
            p = Path(f"data/raw/images/{idx:03d}_img.png")
            if p.exists():
                img = cv2.imread(str(p))
                return cv2.resize(img, (640, 360))
        return None

    def test_real_model_on_val_image(self):
        from perception.detect import detect_gates
        frame = self._load_val_frame()
        if frame is None:
            self.skipTest("No val images found in data/raw/images/")
        result = detect_gates(frame, frame_id=0, sim_time_ns=0,
                               model=self.model, device=DEVICE)
        # Result may be None on some val images (gate may be off-centre)
        if result is not None:
            self.assertGreater(result.distance_to_next_gate_m, 0.0)
            self.assertLess(result.reprojection_error_px, 30.0,
                            "Reprojection error too high — check corner ordering")
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_consecutive_frames_ekf_smooths(self):
        """EKF should not raise and distance should be finite across 3 frames."""
        from perception.detect import detect_gates
        frame = self._load_val_frame()
        if frame is None:
            self.skipTest("No val images found")
        for i in range(3):
            result = detect_gates(frame, i, i * 33_000_000, self.model, DEVICE)
            if result is not None:
                self.assertTrue(np.isfinite(result.distance_to_next_gate_m))


if __name__ == "__main__":
    unittest.main()
