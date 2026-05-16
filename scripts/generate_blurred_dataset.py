#!/usr/bin/env python3
"""Generate a blurred version of an existing dataset while preserving labels.

Creates a parallel directory tree under the destination root containing
blurred images and copies corresponding YOLO `.txt` label files.

Usage:
  python3 scripts/generate_blurred_dataset.py --source yolo_specialist_datasets \
      --dest yolo_specialist_datasets/blurred_dataset
"""
from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path
import cv2

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def process_image(src_path: Path, dst_path: Path, jpeg_quality: int = 75, blur_ksize: int = 9):
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"WARN: failed to read {src_path}")
        return False
    h, w = img.shape[:2]
    # downscale + upscale to create mild blur (simulate low-res capture)
    ds = cv2.resize(img, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_LINEAR)
    us = cv2.resize(ds, (w, h), interpolation=cv2.INTER_LINEAR)
    # gaussian blur
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(us, (k, k), 0)
    # jpeg compression artifact
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    ok, encimg = cv2.imencode('.jpg', blurred, encode_param)
    if not ok:
        print(f"WARN: jpeg encode failed for {src_path}")
        return False
    img_out = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), img_out)
    return True


def find_label_for_image(img_path: Path, source_root: Path) -> Path | None:
    # Common layouts: images/... and labels/... with matching relative paths
    parts = img_path.parts
    try:
        idx = parts.index('images')
    except ValueError:
        idx = -1
    if idx != -1:
        rel = Path(*parts[idx + 1:])
        label_path = source_root.joinpath('labels').joinpath(rel).with_suffix('.txt')
        if label_path.exists():
            return label_path
    # fallback: same directory and same-stem .txt
    alt = img_path.with_suffix('.txt')
    if alt.exists():
        return alt
    # search nearby labels directory (sibling)
    lbl = img_path.parent.parent.joinpath('labels').joinpath(img_path.name).with_suffix('.txt')
    if lbl.exists():
        return lbl
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, help='Source dataset root (contains class dirs)')
    p.add_argument('--dest', required=True, help='Destination root for blurred dataset')
    p.add_argument('--jpeg-quality', type=int, default=70, help='JPEG quality for compression step')
    p.add_argument('--blur-ksize', type=int, default=9, help='Gaussian blur kernel size (odd)')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    src_root = Path(args.source)
    dst_root = Path(args.dest)
    if not src_root.exists():
        raise SystemExit(f"Source root not found: {src_root}")
    if args.dry_run:
        print("Dry run: no files will be written")

    total_images = 0
    written = 0
    for root, dirs, files in os.walk(src_root):
        root_path = Path(root)
        for name in files:
            src_path = root_path.joinpath(name)
            if not is_image(src_path):
                continue
            total_images += 1
            rel = src_path.relative_to(src_root)
            dst_path = dst_root.joinpath(rel)
            if args.dry_run:
                print(f"Would process: {src_path} -> {dst_path}")
                continue
            ok = process_image(src_path, dst_path, jpeg_quality=args.jpeg_quality, blur_ksize=args.blur_ksize)
            if ok:
                written += 1
                # copy label if available
                label = find_label_for_image(src_path, src_root)
                if label:
                    # construct destination label path by mirroring relative path under 'labels' if possible
                    try:
                        idx = src_path.parts.index('images')
                        rel_lbl = Path(*src_path.parts[idx + 1:]).with_suffix('.txt')
                        dst_label_dir = dst_root.joinpath('labels')
                        dst_label_dir.joinpath(rel_lbl).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(label, dst_label_dir.joinpath(rel_lbl))
                    except ValueError:
                        # fallback: place label beside image
                        shutil.copy(label, dst_path.with_suffix('.txt'))

    print(f"Processed {written}/{total_images} images into {dst_root}")


if __name__ == '__main__':
    main()
