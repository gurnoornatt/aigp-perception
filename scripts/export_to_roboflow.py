"""
Upload the augmented dataset to Roboflow for team sharing.

Converts binary PNG masks → COCO JSON format, then pushes to Roboflow
so Faadil can download in whatever format his training code needs.

Usage:
    export ROBOFLOW_API_KEY=<your key>
    python scripts/export_to_roboflow.py

Faadil downloads with:
    from roboflow import Roboflow
    rf = Roboflow(api_key="...")
    version = rf.workspace("gurnoors-workspace-r7c3n").project("gate-segmentation").version(1)
    version.download("coco-segmentation")
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from roboflow import Roboflow
except ImportError:
    print("roboflow not installed. Run: pip install roboflow")
    sys.exit(1)

try:
    from pycocotools import mask as coco_mask_util
except ImportError:
    print("pycocotools not installed. Run: pip install pycocotools")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
WORKSPACE_ID  = "gurnoors-workspace-r7c3n"
PROJECT_NAME  = "gate-segmentation"
CATEGORY_NAME = "gate"

AUG_IMG_DIR  = Path("data/augmented/images")
AUG_MASK_DIR = Path("data/augmented/masks")
EXPORT_DIR   = Path("data/coco_export")


# ── PNG mask → COCO RLE ───────────────────────────────────────────────────────

def mask_to_rle(mask_path: Path) -> dict:
    mask   = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    binary = (mask > 127).astype(np.uint8)
    rle    = coco_mask_util.encode(np.asfortranarray(binary))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_bbox(mask_path: Path) -> list[float]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    binary = (mask > 127).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(binary)
    return [float(x), float(y), float(w), float(h)]


# ── Build COCO JSON ────────────────────────────────────────────────────────────

def build_coco_json(img_paths: list[Path], mask_dir: Path) -> dict:
    coco = {
        "info":        {"description": "AI Grand Prix gate segmentation"},
        "categories":  [{"id": 1, "name": CATEGORY_NAME, "supercategory": "drone"}],
        "images":      [],
        "annotations": [],
    }

    ann_id = 1
    for img_id, img_path in enumerate(tqdm(img_paths, desc="Building COCO JSON"), start=1):
        mask_path = mask_dir / img_path.name.replace("_img.png", "_mask.png")
        if not mask_path.exists():
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        coco["images"].append({
            "id":        img_id,
            "file_name": img_path.name,
            "height":    h,
            "width":     w,
        })

        rle  = mask_to_rle(mask_path)
        bbox = mask_to_bbox(mask_path)
        area = int(coco_mask_util.area(coco_mask_util.encode(
            np.asfortranarray((cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8))
        )))

        coco["annotations"].append({
            "id":           ann_id,
            "image_id":     img_id,
            "category_id":  1,
            "segmentation": rle,
            "bbox":         bbox,
            "area":         area,
            "iscrowd":      1,   # RLE format requires iscrowd=1
        })
        ann_id += 1

    return coco


# ── Export ─────────────────────────────────────────────────────────────────────

def build_export_folder() -> Path:
    export_img_dir = EXPORT_DIR / "train" / "images"
    export_img_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(AUG_IMG_DIR.glob("*_img.png"))
    print(f"Copying {len(img_paths)} images to export folder...")
    for p in tqdm(img_paths):
        shutil.copy(p, export_img_dir / p.name)

    print("Building COCO JSON annotations...")
    coco = build_coco_json(img_paths, AUG_MASK_DIR)

    ann_path = EXPORT_DIR / "train" / "_annotations.coco.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f)
    print(f"Saved annotations: {ann_path} ({len(coco['annotations'])} annotations)")

    return EXPORT_DIR


def upload_to_roboflow(export_dir: Path) -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: Set ROBOFLOW_API_KEY environment variable first.")
        print("  export ROBOFLOW_API_KEY=DxmUkqHkBNh7Q8tWiJWt")
        sys.exit(1)

    print(f"Connecting to Roboflow workspace: {WORKSPACE_ID}")
    rf        = Roboflow(api_key=api_key)
    workspace = rf.workspace(WORKSPACE_ID)

    print(f"Uploading dataset to project: {PROJECT_NAME}")
    workspace.upload_dataset(
        dataset_path=str(export_dir),
        project_name=PROJECT_NAME,
        dataset_format="coco-segmentation",
        project_license="MIT",
        project_type="instance-segmentation",
        num_workers=4,
        batch_size=64,
    )
    print("Upload complete.")
    print(f"\nFaadil can download with:")
    print(f"  version = rf.workspace('{WORKSPACE_ID}').project('{PROJECT_NAME}').version(1)")
    print(f"  version.download('coco-segmentation')")


def main() -> None:
    if not AUG_IMG_DIR.exists():
        print("No augmented images found. Run augment_dataset.py first.")
        sys.exit(1)

    export_dir = build_export_folder()
    upload_to_roboflow(export_dir)


if __name__ == "__main__":
    main()
