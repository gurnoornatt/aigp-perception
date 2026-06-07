"""
Interactive augmentation UI — Gradio.

Lets you pick any base image and tune augmentation parameters with sliders
in real time. See original, augmented, and mask side-by-side. Save individual
results or batch-save with current settings across all base images.

Run:
    python tools/augmentation_ui.py
Then open http://127.0.0.1:7860 in your browser.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import albumentations as A
import gradio as gr

RAW_IMG_DIR  = Path("data/raw/images")
RAW_MASK_DIR = Path("data/raw/masks")
UI_SAVE_DIR  = Path("data/ui_saved")
UI_SAVE_DIR.mkdir(parents=True, exist_ok=True)

BG_COLOR = (15, 20, 30)


# ── Image loading ─────────────────────────────────────────────────────────────

def _list_base_images() -> list[str]:
    if not RAW_IMG_DIR.exists():
        return []
    return sorted(p.name for p in RAW_IMG_DIR.glob("*_img.png"))


def _load_pair(filename: str):
    img_path  = RAW_IMG_DIR  / filename
    base_idx  = filename.split("_")[0]
    mask_path = RAW_MASK_DIR / f"{base_idx}_mask.png"

    image = cv2.imread(str(img_path))
    mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    return image, mask


# ── Build transform from current slider values ────────────────────────────────

def _build_transform(
    blur: int,
    brightness: float,
    contrast: float,
    noise: float,
    rotation: float,
    perspective: float,
    flip_h: bool,
    flip_v: bool,
) -> A.Compose:
    transforms = []

    if rotation != 0 or perspective != 0:
        transforms.append(
            A.Affine(
                rotate=(-abs(rotation), abs(rotation)) if rotation != 0 else (0, 0),
                scale=(1.0, 1.0),
                translate_percent={"x": 0.0, "y": 0.0},
                border_mode=cv2.BORDER_CONSTANT,
                fill=BG_COLOR,
                p=1.0,
            )
        )

    if perspective > 0:
        transforms.append(
            A.Perspective(scale=(perspective, perspective), p=1.0)
        )

    if flip_h:
        transforms.append(A.HorizontalFlip(p=1.0))

    if flip_v:
        transforms.append(A.VerticalFlip(p=1.0))

    if blur > 0:
        k = blur * 2 + 1  # odd kernel size
        transforms.append(A.GaussianBlur(blur_limit=(k, k), p=1.0))

    if brightness != 0 or contrast != 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=(brightness, brightness),
                contrast_limit=(contrast, contrast),
                p=1.0,
            )
        )

    if noise > 0:
        std = noise / 255.0
        transforms.append(
            A.GaussNoise(std_range=(std, std), p=1.0)
        )

    return A.Compose(transforms)


def _apply_transform(image, mask, transform: A.Compose):
    result   = transform(image=image, mask=mask)
    aug_img  = result["image"]
    aug_mask = (result["mask"] > 127).astype(np.uint8) * 255
    return aug_img, aug_mask


# ── Main preview function (called on every slider change) ─────────────────────

def update_preview(
    filename,
    blur, brightness, contrast, noise, rotation, perspective,
    flip_h, flip_v,
):
    if not filename:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        return blank, blank, blank

    try:
        image, mask = _load_pair(filename)
    except FileNotFoundError as e:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, str(e), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return blank, blank, blank

    transform = _build_transform(blur, brightness, contrast, noise, rotation, perspective, flip_h, flip_v)
    aug_img, aug_mask = _apply_transform(image, mask, transform)

    orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aug_rgb  = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(aug_mask, cv2.COLOR_GRAY2RGB)

    return orig_rgb, aug_rgb, mask_rgb


# ── Save functions ─────────────────────────────────────────────────────────────

def save_current(filename, blur, brightness, contrast, noise, rotation, perspective, flip_h, flip_v):
    if not filename:
        return "No image selected."
    try:
        image, mask = _load_pair(filename)
    except FileNotFoundError as e:
        return str(e)

    transform = _build_transform(blur, brightness, contrast, noise, rotation, perspective, flip_h, flip_v)
    aug_img, aug_mask = _apply_transform(image, mask, transform)

    ts = datetime.now().strftime("%H%M%S")
    img_out  = UI_SAVE_DIR / f"ui_{ts}_img.png"
    mask_out = UI_SAVE_DIR / f"ui_{ts}_mask.png"
    cv2.imwrite(str(img_out),  aug_img)
    cv2.imwrite(str(mask_out), aug_mask)
    return f"Saved: {img_out.name} + {mask_out.name}"


def batch_save(filename, blur, brightness, contrast, noise, rotation, perspective, flip_h, flip_v):
    all_images = _list_base_images()
    if not all_images:
        return "No base images found. Run generate_synthetic.py first."

    transform = _build_transform(blur, brightness, contrast, noise, rotation, perspective, flip_h, flip_v)
    count = 0
    for fname in all_images:
        try:
            image, mask = _load_pair(fname)
            aug_img, aug_mask = _apply_transform(image, mask, transform)
            base = fname.replace("_img.png", "")
            cv2.imwrite(str(UI_SAVE_DIR / f"batch_{base}_img.png"),  aug_img)
            cv2.imwrite(str(UI_SAVE_DIR / f"batch_{base}_mask.png"), aug_mask)
            count += 1
        except Exception:
            continue

    return f"Batch saved {count} pairs to {UI_SAVE_DIR}/"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    images = _list_base_images()
    default_img = images[0] if images else None

    with gr.Blocks(title="Gate Augmentation Studio") as demo:
        gr.Markdown("## Gate Augmentation Studio\nAdjust sliders to augment in real time. Changes apply to both image and mask.")

        # State for flip toggles
        flip_h_state = gr.State(False)
        flip_v_state = gr.State(False)

        # ── File picker ───────────────────────────────────────────────────────
        with gr.Row():
            img_dropdown = gr.Dropdown(
                choices=images,
                value=default_img,
                label="Base image",
                scale=3,
            )
            refresh_btn = gr.Button("Refresh list", scale=1)

        # ── Image display ─────────────────────────────────────────────────────
        with gr.Row():
            orig_display = gr.Image(label="Original", height=240)
            aug_display  = gr.Image(label="Augmented (live)", height=240)
            mask_display = gr.Image(label="Mask", height=240)

        # ── Sliders row 1 ─────────────────────────────────────────────────────
        with gr.Row():
            blur_slider = gr.Slider(
                0, 10, value=0, step=1,
                label="Blur (0=off, higher=more blur)",
            )
            brightness_slider = gr.Slider(
                -0.5, 0.5, value=0, step=0.05,
                label="Brightness",
            )
            contrast_slider = gr.Slider(
                -0.5, 0.5, value=0, step=0.05,
                label="Contrast",
            )
            noise_slider = gr.Slider(
                0, 50, value=0, step=1,
                label="Gaussian Noise (sigma)",
            )

        # ── Sliders row 2 ─────────────────────────────────────────────────────
        with gr.Row():
            rotation_slider = gr.Slider(
                -45, 45, value=0, step=1,
                label="Rotation (degrees)",
            )
            perspective_slider = gr.Slider(
                0.0, 0.15, value=0.0, step=0.01,
                label="Perspective Warp",
            )

        # ── Flip buttons ──────────────────────────────────────────────────────
        with gr.Row():
            flip_h_btn  = gr.Button("Flip Horizontal")
            flip_v_btn  = gr.Button("Flip Vertical")
            reset_btn   = gr.Button("Reset All")

        # ── Save buttons ──────────────────────────────────────────────────────
        with gr.Row():
            save_btn       = gr.Button("Save Current")
            batch_save_btn = gr.Button("Batch Save (all 150 base images)")
        status_box = gr.Textbox(label="Status", interactive=False)

        # ── Shared inputs/outputs ─────────────────────────────────────────────
        all_inputs = [
            img_dropdown,
            blur_slider, brightness_slider, contrast_slider, noise_slider,
            rotation_slider, perspective_slider,
            flip_h_state, flip_v_state,
        ]
        display_outputs = [orig_display, aug_display, mask_display]

        # ── Wire sliders ──────────────────────────────────────────────────────
        for component in [
            img_dropdown,
            blur_slider, brightness_slider, contrast_slider, noise_slider,
            rotation_slider, perspective_slider,
        ]:
            component.change(fn=update_preview, inputs=all_inputs, outputs=display_outputs)

        # ── Flip buttons ──────────────────────────────────────────────────────
        def toggle_h(state): return not state
        def toggle_v(state): return not state

        flip_h_btn.click(fn=toggle_h, inputs=[flip_h_state], outputs=[flip_h_state]).then(
            fn=update_preview, inputs=all_inputs, outputs=display_outputs
        )
        flip_v_btn.click(fn=toggle_v, inputs=[flip_v_state], outputs=[flip_v_state]).then(
            fn=update_preview, inputs=all_inputs, outputs=display_outputs
        )

        # ── Reset ─────────────────────────────────────────────────────────────
        def reset_all():
            return 0, 0.0, 0.0, 0, 0, 0.0, False, False

        reset_btn.click(
            fn=reset_all,
            inputs=[],
            outputs=[
                blur_slider, brightness_slider, contrast_slider, noise_slider,
                rotation_slider, perspective_slider,
                flip_h_state, flip_v_state,
            ],
        ).then(fn=update_preview, inputs=all_inputs, outputs=display_outputs)

        # ── Refresh file list ─────────────────────────────────────────────────
        def refresh():
            imgs = _list_base_images()
            return gr.Dropdown(choices=imgs, value=imgs[0] if imgs else None)

        refresh_btn.click(fn=refresh, outputs=[img_dropdown])

        # ── Save buttons ──────────────────────────────────────────────────────
        save_btn.click(fn=save_current, inputs=all_inputs, outputs=[status_box])
        batch_save_btn.click(fn=batch_save, inputs=all_inputs, outputs=[status_box])

        # ── Initial render ────────────────────────────────────────────────────
        demo.load(fn=update_preview, inputs=all_inputs, outputs=display_outputs)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True)
