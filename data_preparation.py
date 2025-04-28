#!/usr/bin/env python3
"""
data_preparation.py
===================

Down-sample 1080 p frames to 720 p with high-quality LANCZOS filtering.

Example
-------
python data_preparation.py \
    --src datasets/davis_trainval_2017_1080p \
    --dst datasets/davis_trainval_2017_720p
"""
from __future__ import annotations
import argparse
from pathlib import Path
from multiprocessing.pool import ThreadPool
from PIL import Image

TARGET_SIZE = (1280, 720)        # (width, height) for 720p
# TARGET_SIZE = (1920, 1080)        # (width, height) for 720p

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def convert_one(args):
    src_file, dst_file = args
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_file) as im:
        # Convert to RGB in case the file is PNG+alpha etc.
        im = im.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
        im.save(dst_file, quality=95, subsampling=0)   # high-quality JPEG

def list_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser("1080p → 720p frame down-sampler (LANCZOS)")
    ap.add_argument("--src", required=True, type=Path,
                    help="Folder with original 1080p frames")
    ap.add_argument("--dst", required=True, type=Path,
                    help="Output folder for 720p frames")
    ap.add_argument("--threads", type=int, default=8,
                    help="Parallelism (default 8)")
    args = ap.parse_args()

    src_files = list_images(args.src)
    if not src_files:
        raise SystemExit("No image files found in --src folder.")

    tasks = []
    for src_path in src_files:
        rel = src_path.relative_to(args.src)
        dst_path = args.dst / rel
        tasks.append((src_path, dst_path))

    print(f"Converting {len(tasks)} images with {args.threads} threads …")
    with ThreadPool(args.threads) as pool:
        for _ in pool.imap_unordered(convert_one, tasks):
            pass
    print("Done ✔")

if __name__ == "__main__":
    main()
