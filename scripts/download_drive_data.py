"""
Download gate training data from Google Drive into data/.

Expected layout after download (matches training/dataset.py):
    data/raw/images/{idx:03d}_img.png
    data/raw/masks/{idx:03d}_mask.png
    data/augmented/images/{idx:03d}_{aug:02d}_img.png
    data/augmented/masks/{idx:03d}_{aug:02d}_mask.png

Usage:
    python scripts/download_drive_data.py
    python scripts/download_drive_data.py --folder-id 1uR1hxLsO6-wzdJpbYlL2kmdjzv5oRixc
    python scripts/download_drive_data.py --zip path/to/downloaded.zip
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DEFAULT_FOLDER_ID = "1uR1hxLsO6-wzdJpbYlL2kmdjzv5oRixc"
DATA_DIR = Path("data")

RAW_IMG = DATA_DIR / "raw" / "images"
RAW_MASK = DATA_DIR / "raw" / "masks"
AUG_IMG = DATA_DIR / "augmented" / "images"
AUG_MASK = DATA_DIR / "augmented" / "masks"


def _ensure_layout() -> None:
    for d in (RAW_IMG, RAW_MASK, AUG_IMG, AUG_MASK):
        d.mkdir(parents=True, exist_ok=True)


def _count_pairs() -> dict[str, int]:
    return {
        "raw": len(list(RAW_IMG.glob("*_img.png"))),
        "augmented": len(list(AUG_IMG.glob("*_img.png"))),
    }


def _relocate_tree(src: Path, dst_root: Path) -> None:
    """Move files from src into dst_root, preserving relative paths."""
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = dst_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        shutil.move(str(path), str(target))


def _normalize_downloaded_layout(download_root: Path) -> None:
    """
    Accept common Drive export layouts:
      - raw/images, raw/masks, augmented/images, augmented/masks
      - images/, masks/ at top level
      - nested duplicate data/ folder
    """
    candidates = [download_root, download_root / "data"]
    for root in candidates:
        if (root / "raw" / "images").exists():
            _relocate_tree(root / "raw" / "images", RAW_IMG)
            _relocate_tree(root / "raw" / "masks", RAW_MASK)
        if (root / "augmented" / "images").exists():
            _relocate_tree(root / "augmented" / "images", AUG_IMG)
            _relocate_tree(root / "augmented" / "masks", AUG_MASK)
        if (root / "images").exists() and (root / "masks").exists():
            # Heuristic: augmented files have two numeric groups (e.g. 042_03_img.png)
            for img in (root / "images").glob("*.png"):
                name = img.name
                parts = name.replace("_img.png", "").split("_")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    dst_img, dst_mask = AUG_IMG, AUG_MASK
                else:
                    dst_img, dst_mask = RAW_IMG, RAW_MASK
                mask_name = name.replace("_img.png", "_mask.png")
                mask = root / "masks" / mask_name
                shutil.copy2(img, dst_img / name)
                if mask.exists():
                    shutil.copy2(mask, dst_mask / mask_name)


def download_from_drive(folder_id: str, output_dir: Path) -> None:
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    cmd = [sys.executable, "-m", "gdown", "--folder", url, "-O", str(output_dir)]
    print(f"Downloading {url} ...")
    subprocess.run(cmd, check=True)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    print(f"Extracted {zip_path} -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download gate dataset into data/")
    parser.add_argument("--folder-id", default=DEFAULT_FOLDER_ID)
    parser.add_argument("--zip", type=Path, help="Local zip from manual Drive download")
    parser.add_argument("--staging", type=Path, default=Path("data/_drive_download"))
    args = parser.parse_args()

    _ensure_layout()

    if args.zip:
        staging = args.staging
        if staging.exists():
            shutil.rmtree(staging)
        extract_zip(args.zip, staging)
        _normalize_downloaded_layout(staging)
        shutil.rmtree(staging, ignore_errors=True)
    else:
        staging = args.staging
        if staging.exists():
            shutil.rmtree(staging)
        try:
            download_from_drive(args.folder_id, staging)
        except subprocess.CalledProcessError as exc:
            print(
                "\nDrive download failed. The folder may require sign-in.\n"
                "Fix options:\n"
                "  1. In Google Drive: Share -> General access -> 'Anyone with the link' (Viewer)\n"
                "  2. Manual: download the folder as a zip in your browser, then run:\n"
                f"       python scripts/download_drive_data.py --zip path/to/folder.zip\n"
                "  3. Generate locally:\n"
                "       python scripts/generate_synthetic.py\n"
                "       python scripts/augment_dataset.py\n"
            )
            raise SystemExit(exc.returncode) from exc
        _normalize_downloaded_layout(staging)
        shutil.rmtree(staging, ignore_errors=True)

    counts = _count_pairs()
    print(f"Dataset ready — raw: {counts['raw']} pairs, augmented: {counts['augmented']} pairs")
    if counts["raw"] < 150 or counts["augmented"] < 2400:
        print("Warning: counts are lower than expected (150 raw, ~2400 augmented).")


if __name__ == "__main__":
    main()
