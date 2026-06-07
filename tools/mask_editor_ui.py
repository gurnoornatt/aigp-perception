"""
Interactive mask editor for sim frame images.

- Starts from the recommended red-gate filter (experiment_gate_mask.py)
- Reference image shown under the mask while you draw
- Polygon click-to-fill (white = gate, black = background)
- Brush painting on the mask layer

Run from repo root:
    python tools/mask_editor_ui.py
Then open http://127.0.0.1:7861
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import re
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np

from scripts.experiment_gate_mask import mask_gates_recommended

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = REPO_ROOT / "experiments" / "video" / "frames"
MASKS_DIR = REPO_ROOT / "experiments" / "video" / "masks"
MASKS_DIR.mkdir(parents=True, exist_ok=True)

FRAME_RE = re.compile(r"^frame_\d{2}\.png$")
EditorValue = dict[str, Any]


def list_frames() -> list[str]:
    if not FRAMES_DIR.exists():
        return []
    return sorted(
        p.name for p in FRAMES_DIR.glob("frame_*.png") if FRAME_RE.match(p.name)
    )


def _mask_path(frame_name: str) -> Path:
    stem = Path(frame_name).stem
    return MASKS_DIR / f"{stem}_mask.png"


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def _as_uint8_image(img: Any) -> np.ndarray | None:
    if img is None:
        return None
    if isinstance(img, dict):
        for key in ("composite", "image", "background"):
            if key in img:
                return _as_uint8_image(img[key])
        return None
    if isinstance(img, (str, bytes)):
        return None
    arr = np.asarray(img)
    if not isinstance(arr, np.ndarray) or arr.size == 0 or arr.ndim < 2:
        return None
    if arr.dtype == np.object_:
        return None
    if arr.dtype in (np.float32, np.float64):
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _get_layers_list(value: dict) -> list[Any]:
    layers = value.get("layers")
    if layers is None:
        return []
    if isinstance(layers, (list, tuple)):
        return list(layers)
    if isinstance(layers, np.ndarray):
        return [layers] if layers.ndim >= 2 else []
    return [layers]


def _layer_to_mask(layer: Any, fallback: np.ndarray) -> np.ndarray:
    arr = _as_uint8_image(layer)
    if arr is None:
        return fallback.copy()
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
        painted = alpha > 8
        out = fallback.copy() if fallback.shape == gray.shape else np.zeros(gray.shape, np.uint8)
        out[painted & (gray > 127)] = 255
        out[painted & (gray <= 127)] = 0
        return out
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    return ((gray > 127).astype(np.uint8)) * 255


def _from_editor(
    value: EditorValue | np.ndarray | None,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Read mask from editor layers only (background is the reference image)."""
    fb = fallback.copy() if fallback is not None else np.zeros((1, 1), dtype=np.uint8)
    if value is None:
        return fb
    if not isinstance(value, dict):
        return _layer_to_mask(value, fb)
    for layer in reversed(_get_layers_list(value)):
        if layer is None:
            continue
        mask = _layer_to_mask(layer, fb)
        if mask.size > 1:
            return mask
    bg = _as_uint8_image(value.get("background"))
    if bg is not None and fb.shape[0] <= 1:
        h, w = bg.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)
    return fb


def _to_editor_with_ref(bgr: np.ndarray, mask: np.ndarray, alpha: float) -> EditorValue:
    """Reference image as background; mask on an editable layer on top."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    ref = _bgr_to_rgb(bgr)
    background = (ref.astype(np.float32) * alpha).astype(np.uint8)
    mask_rgb = _mask_to_rgb(mask)
    composite = overlay_reference(bgr, mask, alpha)
    return {"background": background, "layers": [mask_rgb], "composite": composite}


def _load_bgr(frame_name: str) -> np.ndarray:
    path = FRAMES_DIR / frame_name
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read frame: {path}")
    return bgr


def _load_mask(frame_name: str, bgr: np.ndarray) -> np.ndarray:
    path = _mask_path(frame_name)
    if path.exists():
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return mask
    return mask_gates_recommended(bgr)


def overlay_reference(bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """RGB preview: reference dimmed with white gate regions highlighted."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    rgb = _bgr_to_rgb(bgr).astype(np.float32)
    gate = mask > 127
    out = rgb * alpha
    out[gate] = rgb[gate] * alpha + np.array([255.0, 255.0, 255.0]) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_polygon_preview(
    bgr: np.ndarray,
    mask: np.ndarray,
    points: list[list[int]],
    alpha: float,
) -> np.ndarray:
    preview = overlay_reference(bgr, mask, alpha)
    if len(points) >= 2:
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(preview, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
    for x, y in points:
        cv2.circle(preview, (int(x), int(y)), 5, (255, 255, 0), -1)
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(preview, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    return preview


def apply_polygon_to_editor(
    bgr: np.ndarray,
    mask: np.ndarray,
    points: list[list[int]],
    fill_white: bool,
    alpha: float,
) -> tuple[EditorValue, np.ndarray]:
    out = mask.copy()
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32)
        color = 255 if fill_white else 0
        cv2.fillPoly(out, [pts], color)
    return _to_editor_with_ref(bgr, out, alpha), out


def load_frame(frame_name: str, overlay_alpha: float):
    if not frame_name:
        empty = np.zeros((64, 64, 3), dtype=np.uint8)
        zm = np.zeros((64, 64), dtype=np.uint8)
        return empty, _to_editor_with_ref(empty, zm, overlay_alpha), empty, [], zm, "No frames found."

    bgr = _load_bgr(frame_name)
    mask = _load_mask(frame_name, bgr)
    ref = _bgr_to_rgb(bgr)
    editor = _to_editor_with_ref(bgr, mask, overlay_alpha)
    preview = overlay_reference(bgr, mask, overlay_alpha)
    saved = " (saved)" if _mask_path(frame_name).exists() else " (auto filter)"
    return ref, editor, preview, [], mask, f"Loaded {frame_name}{saved}"


def apply_recommended(frame_name: str, overlay_alpha: float):
    if not frame_name:
        z = np.zeros((64, 64), dtype=np.uint8)
        return _to_editor_with_ref(np.zeros((64, 64, 3), dtype=np.uint8), z, overlay_alpha), z, z, "No frame selected."
    bgr = _load_bgr(frame_name)
    mask = mask_gates_recommended(bgr)
    editor = _to_editor_with_ref(bgr, mask, overlay_alpha)
    preview = overlay_reference(bgr, mask, overlay_alpha)
    return editor, preview, mask, "Applied recommended filter."


def apply_recommended_all():
    frames = list_frames()
    if not frames:
        return "No frames to process."
    for name in frames:
        bgr = _load_bgr(name)
        mask = mask_gates_recommended(bgr)
        cv2.imwrite(str(_mask_path(name)), mask)
    return f"Applied recommended filter to {len(frames)} frames."


def on_reference_click(
    evt: gr.SelectData,
    frame_name: str,
    mask_state: np.ndarray,
    points: list[list[int]],
    overlay_alpha: float,
):
    if not frame_name:
        return points, mask_state, "No frame selected."
    x, y = int(evt.index[0]), int(evt.index[1])
    points = list(points) + [[x, y]]
    bgr = _load_bgr(frame_name)
    preview = draw_polygon_preview(bgr, mask_state, points, overlay_alpha)
    return points, preview, f"Polygon vertex {len(points)}: ({x}, {y})"


def fill_polygon(
    frame_name: str,
    mask_state: np.ndarray,
    points: list[list[int]],
    fill_white: bool,
    overlay_alpha: float,
):
    if not frame_name:
        z = np.zeros((64, 64, 3), dtype=np.uint8)
        return mask_state, mask_state, [], z, "No frame selected."
    bgr = _load_bgr(frame_name)
    editor, mask = apply_polygon_to_editor(bgr, mask_state, points, fill_white, overlay_alpha)
    preview = overlay_reference(bgr, mask, overlay_alpha)
    label = "white (gate)" if fill_white else "black (background)"
    return editor, mask, [], preview, f"Filled polygon with {label}."


def on_editor_change(
    frame_name: str,
    editor_val: EditorValue,
    mask_state: np.ndarray,
    overlay_alpha: float,
):
    if not frame_name:
        z = np.zeros((64, 64, 3), dtype=np.uint8)
        return z, mask_state
    bgr = _load_bgr(frame_name)
    fb = mask_state if mask_state is not None else np.zeros(bgr.shape[:2], dtype=np.uint8)
    mask = _from_editor(editor_val, fb)
    preview = overlay_reference(bgr, mask, overlay_alpha)
    return preview, mask


def refresh_editor_background(frame_name: str, mask_state: np.ndarray, overlay_alpha: float):
    if not frame_name:
        z = np.zeros((64, 64, 3), dtype=np.uint8)
        zm = np.zeros((64, 64), dtype=np.uint8)
        return _to_editor_with_ref(z, zm, overlay_alpha), z, zm
    bgr = _load_bgr(frame_name)
    mask = mask_state if mask_state is not None else np.zeros(bgr.shape[:2], dtype=np.uint8)
    editor = _to_editor_with_ref(bgr, mask, overlay_alpha)
    preview = overlay_reference(bgr, mask, overlay_alpha)
    return editor, preview, mask


def save_mask(frame_name: str, mask_state: np.ndarray):
    if not frame_name or mask_state is None:
        return "No frame selected."
    path = _mask_path(frame_name)
    cv2.imwrite(str(path), mask_state)
    white_pct = 100.0 * (mask_state > 0).mean()
    return f"Saved {path.name} ({white_pct:.2f}% gate pixels)"


def save_and_next(frame_name: str, mask_state: np.ndarray, overlay_alpha: float):
    status = save_mask(frame_name, mask_state)
    frames = list_frames()
    if frame_name not in frames:
        z = np.zeros((1, 1), dtype=np.uint8)
        return status, None, None, _to_editor_with_ref(np.zeros((1, 1, 3), dtype=np.uint8), z, overlay_alpha), z, None
    idx = frames.index(frame_name)
    next_name = frames[(idx + 1) % len(frames)]
    ref, editor, preview, _, mask, load_msg = load_frame(next_name, overlay_alpha)
    return f"{status} → {load_msg}", next_name, ref, editor, mask, preview


def navigate(frame_name: str, step: int, overlay_alpha: float):
    frames = list_frames()
    if not frames:
        empty = np.zeros((64, 64, 3), dtype=np.uint8)
        zm = np.zeros((64, 64), dtype=np.uint8)
        editor = _to_editor_with_ref(empty, zm, overlay_alpha)
        return None, empty, editor, empty, [], zm, "No frames."
    if frame_name not in frames:
        idx = 0
    else:
        idx = (frames.index(frame_name) + step) % len(frames)
    name = frames[idx]
    ref, editor, preview, _, mask, msg = load_frame(name, overlay_alpha)
    return name, ref, editor, preview, [], mask, msg


def clear_polygon(frame_name: str, mask_state: np.ndarray, overlay_alpha: float):
    if not frame_name:
        return [], "Cleared polygon points."
    bgr = _load_bgr(frame_name)
    preview = draw_polygon_preview(bgr, mask_state, [], overlay_alpha)
    return [], preview, "Cleared polygon points."


def build_ui() -> gr.Blocks:
    frames = list_frames()
    default = frames[0] if frames else None
    white_brush = gr.Brush(
        colors=["#FFFFFF", "#000000"],
        default_color="#FFFFFF",
        color_mode="fixed",
        default_size=24,
    )

    with gr.Blocks(title="Gate Mask Editor") as demo:
        gr.Markdown(
            "## Gate mask editor\n"
            "1. **Recommended filter** seeds each mask.\n"
            "2. **Mask editor** shows the **reference image underneath** while you brush white (gate) / black (background).\n"
            "3. **Click the reference panel** to place polygon vertices → fill white or black.\n"
            "4. **Preview** shows the final mask blended over the frame.\n"
            "5. **Save** writes PNGs to `experiments/video/masks/`."
        )

        poly_state = gr.State([])
        mask_state = gr.State(None)

        with gr.Row():
            frame_dd = gr.Dropdown(choices=frames, value=default, label="Frame", scale=3)
            prev_btn = gr.Button("← Prev", scale=1)
            next_btn = gr.Button("Next →", scale=1)
            overlay_slider = gr.Slider(
                0.2, 1.0, value=0.65, step=0.05, label="Reference overlay strength", scale=2
            )

        with gr.Row():
            ref_img = gr.Image(label="Reference (click to add polygon points)", type="numpy")
            mask_img = gr.ImageEditor(
                label="Mask editor (reference underlay + brush)",
                type="numpy",
                sources=[],
                transforms=[],
                brush=white_brush,
                eraser=gr.Eraser(default_size=24),
                layers=gr.LayerOptions(allow_additional_layers=False),
            )
            preview_img = gr.Image(label="Preview (reference + mask)", type="numpy")

        with gr.Row():
            apply_btn = gr.Button("Apply recommended filter", variant="secondary")
            apply_all_btn = gr.Button("Apply recommended to ALL frames", variant="secondary")
            clear_poly_btn = gr.Button("Clear polygon points")
            fill_white_btn = gr.Button("Fill polygon WHITE", variant="primary")
            fill_black_btn = gr.Button("Fill polygon BLACK")
            save_btn = gr.Button("Save mask")
            save_next_btn = gr.Button("Save & next", variant="primary")

        status = gr.Textbox(label="Status", interactive=False)

        load_inputs = [frame_dd, overlay_slider]
        load_outputs = [ref_img, mask_img, preview_img, poly_state, mask_state, status]

        demo.load(fn=load_frame, inputs=load_inputs, outputs=load_outputs)
        frame_dd.change(fn=load_frame, inputs=load_inputs, outputs=load_outputs)

        prev_btn.click(
            fn=lambda f, a: navigate(f, -1, a),
            inputs=[frame_dd, overlay_slider],
            outputs=[frame_dd, ref_img, mask_img, preview_img, poly_state, mask_state, status],
        )
        next_btn.click(
            fn=lambda f, a: navigate(f, 1, a),
            inputs=[frame_dd, overlay_slider],
            outputs=[frame_dd, ref_img, mask_img, preview_img, poly_state, mask_state, status],
        )

        apply_btn.click(
            fn=apply_recommended,
            inputs=[frame_dd, overlay_slider],
            outputs=[mask_img, preview_img, mask_state, status],
        )
        apply_all_btn.click(fn=apply_recommended_all, outputs=[status])

        ref_img.select(
            fn=on_reference_click,
            inputs=[frame_dd, mask_state, poly_state, overlay_slider],
            outputs=[poly_state, preview_img, status],
        )

        clear_poly_btn.click(
            fn=clear_polygon,
            inputs=[frame_dd, mask_state, overlay_slider],
            outputs=[poly_state, preview_img, status],
        )

        for btn, white in [(fill_white_btn, True), (fill_black_btn, False)]:
            btn.click(
                fn=lambda f, m, p, a, w=white: fill_polygon(f, m, p, w, a),
                inputs=[frame_dd, mask_state, poly_state, overlay_slider],
                outputs=[mask_img, mask_state, poly_state, preview_img, status],
            )

        mask_img.change(
            fn=on_editor_change,
            inputs=[frame_dd, mask_img, mask_state, overlay_slider],
            outputs=[preview_img, mask_state],
        )
        overlay_slider.change(
            fn=refresh_editor_background,
            inputs=[frame_dd, mask_state, overlay_slider],
            outputs=[mask_img, preview_img, mask_state],
        )

        save_btn.click(fn=save_mask, inputs=[frame_dd, mask_state], outputs=[status])
        save_next_btn.click(
            fn=save_and_next,
            inputs=[frame_dd, mask_state, overlay_slider],
            outputs=[status, frame_dd, ref_img, mask_img, mask_state, preview_img],
        )

    return demo


def launch_demo(demo: gr.Blocks) -> None:
    for port in (7861, 7862, 7863, 7860):
        try:
            demo.launch(server_port=port)
            return
        except OSError:
            continue
    demo.launch()


if __name__ == "__main__":
    launch_demo(build_ui())
