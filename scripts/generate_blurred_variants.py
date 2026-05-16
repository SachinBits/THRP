#!/usr/bin/env python3
"""Generate multiple degraded variants per image (motion blur, heavy blur, downscale, noise, jpeg).

Saves variants alongside a mirrored directory structure under destination root.
Copies YOLO `.txt` labels next to each generated image.

Example:
  python3 scripts/generate_blurred_variants.py --source test --dest outputs/blurred_test --single test/f14_2.jpg 
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import shutil
import cv2
import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def gaussian_blur(img: np.ndarray, ksize: int = 15) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img: np.ndarray, degree: int = 20, angle: float = 0.0) -> np.ndarray:
    k = np.zeros((degree, degree), dtype=np.float32)
    k[int((degree - 1) / 2), :] = np.ones(degree, dtype=np.float32)
    M = cv2.getRotationMatrix2D((degree / 2 - 0.5, degree / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (degree, degree))
    k = k * (1.0 / np.sum(k))
    return cv2.filter2D(img, -1, k)


def downscale_upscale(img: np.ndarray, factor: int = 3) -> np.ndarray:
    h, w = img.shape[:2]
    ds = cv2.resize(img, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(ds, (w, h), interpolation=cv2.INTER_LINEAR)


def add_noise(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def jpeg_compress(img: np.ndarray, quality: int = 60) -> np.ndarray:
    ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


VARIANT_FUNCS = {
    'gaussian_blur': lambda img: gaussian_blur(img, ksize=15),
    'heavy_blur': lambda img: gaussian_blur(img, ksize=31),
    'motion_blur': lambda img: motion_blur(img, degree=25, angle=10.0),
    'downscale': lambda img: downscale_upscale(img, factor=3),
    'noise': lambda img: add_noise(img, sigma=12.0),
    'jpeg': lambda img: jpeg_compress(img, quality=55),
}


def copy_label_for_image(src_img: Path, src_root: Path, dst_img: Path):
    # try images/labels mirrored layout
    try:
        idx = src_img.parts.index('images')
    except ValueError:
        idx = -1
    if idx != -1:
        rel = Path(*src_img.parts[idx + 1:]).with_suffix('.txt')
        possible = src_root.joinpath('labels').joinpath(rel)
        if possible.exists():
            dst_label = dst_img.with_suffix('.txt')
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(possible, dst_label)
            return

    alt = src_img.with_suffix('.txt')
    if alt.exists():
        dst_label = dst_img.with_suffix('.txt')
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(alt, dst_label)


def process_file(src_path: Path, src_root: Path, dst_root: Path, variants: list[str], copies: int = 1):
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"WARN: failed to read {src_path}")
        return 0
    rel = src_path.relative_to(src_root)
    created = 0
    for v in variants:
        if v not in VARIANT_FUNCS:
            continue
        for i in range(copies):
            out_rel = rel.parent / f"{rel.stem}__{v}__{i}{rel.suffix}"
            out_path = dst_root.joinpath(out_rel)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_img = VARIANT_FUNCS[v](img)
            cv2.imwrite(str(out_path), out_img)
            copy_label_for_image(src_path, src_root, out_path)
            created += 1
    return created


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, help='Source dataset root')
    p.add_argument('--dest', required=True, help='Destination root for variants')
    p.add_argument('--variants', default=','.join(VARIANT_FUNCS.keys()), help='Comma-separated variants to generate')
    p.add_argument('--copies', type=int, default=1, help='Copies per variant (random seeds)')
    p.add_argument('--single', help='Process a single image file instead of walking tree')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    src_root = Path(args.source)
    dst_root = Path(args.dest)
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]

    if not src_root.exists():
        raise SystemExit(f"Source not found: {src_root}")
    if args.single:
        src_file = Path(args.single)
        if not src_file.exists():
            raise SystemExit(f"Single file not found: {src_file}")
        if args.dry_run:
            print(f"Would create variants {variants} for {src_file}")
            return
        n = process_file(src_file, src_root, dst_root, variants, copies=args.copies)
        print(f"Created {n} variant files")
        return

    total_created = 0
    total_images = 0
    for root, dirs, files in os.walk(src_root):
        root_path = Path(root)
        for name in files:
            src_path = root_path.joinpath(name)
            if not is_image(src_path):
                continue
            total_images += 1
            if args.dry_run:
                print(f"Would process: {src_path}")
                continue
            total_created += process_file(src_path, src_root, dst_root, variants, copies=args.copies)

    print(f"Processed {total_images} images, created {total_created} variant files under {dst_root}")


if __name__ == '__main__':
    main()
