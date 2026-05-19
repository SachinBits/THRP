"""Precompute and cache HOG+color features for classical baselines.

Saves per-superclass NumPy archives under the `--out` directory with files:
- features.npy  (N x D float32)
- labels.npy    (N,)
- image_paths.txt
- class_names.json

Usage:
  python3 scripts/precompute_features.py --dataset yolo_specialist_datasets --out results/features
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import cv2

from extract_features import extract_combined

# Keep a small default list here so the script doesn't import the full helper module
DEFAULT_SUPERCLASSES = ["fighter", "cargo", "helicopter", "bomber"]


def load_class_names_from_config(specialist_root: Path):
    yaml_candidates = [
        specialist_root / f"{specialist_root.name}_config.yaml",
        specialist_root / f"{specialist_root.name.lower()}_config.yaml",
        specialist_root / f"{specialist_root.name.upper()}_config.yaml",
    ]

    for yaml_path in yaml_candidates:
        if not yaml_path.exists():
            continue
        try:
            import yaml

            with open(yaml_path, "r", encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle)
            names = cfg.get("names", {})
            if isinstance(names, dict):
                return [str(names.get(index, f"Class_{index}")) for index in sorted(names.keys())]
            if isinstance(names, list):
                return [str(name) for name in names]
        except Exception:
            pass

    return []


def process_specialist(specialist_root: Path, out_dir: Path):
    images = []
    labels = []
    paths = []

    for split in ("train", "val"):
        img_dir = specialist_root / "images" / split
        lbl_dir = specialist_root / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            try:
                with open(lbl_path, "r", encoding="utf-8") as h:
                    line = h.readline().strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
            except Exception:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            try:
                feat = extract_combined(img)
            except Exception:
                continue

            images.append(feat.astype(np.float32))
            labels.append(class_id)
            paths.append(str(img_path))

    if not images:
        raise RuntimeError(f"No images found under {specialist_root}")

    x = np.stack(images)
    y = np.array(labels, dtype=np.int32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", x)
    np.save(out_dir / "labels.npy", y)
    with open(out_dir / "image_paths.txt", "w", encoding="utf-8") as h:
        for p in paths:
            h.write(p + "\n")

    class_names = load_class_names_from_config(specialist_root)
    with open(out_dir / "class_names.json", "w", encoding="utf-8") as h:
        json.dump(class_names, h, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to yolo_specialist_datasets root")
    parser.add_argument("--out", required=True, help="Output directory for cached features")
    parser.add_argument("--superclasses", nargs="*", help="Optional list of superclasses to process")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    out_root = Path(args.out)

    if args.superclasses and len(args.superclasses) > 0:
        targets = args.superclasses
    else:
        targets = DEFAULT_SUPERCLASSES

    for sc in targets:
        sc_root = dataset_root / sc
        if not sc_root.exists():
            print(f"Skipping {sc}: path does not exist: {sc_root}")
            continue

        out_dir = out_root / sc
        print(f"Processing {sc} -> {out_dir}")
        t0 = time.time()
        try:
            process_specialist(sc_root, out_dir)
        except Exception as exc:
            print(f"Failed {sc}: {exc}")
            continue
        dt = time.time() - t0
        print(f"Completed {sc} in {dt:.1f}s, saved features shape: ", end="")
        try:
            import numpy as _np
            feats = _np.load(out_dir / "features.npy")
            print(feats.shape)
        except Exception:
            print("(unknown)")


if __name__ == "__main__":
    main()
