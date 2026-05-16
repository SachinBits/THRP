"""CLI runner for aircraft preprocessing and quick benchmarking."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from aircraft_preprocessing import AircraftEnhancementConfig, AircraftPreprocessor


def _save_preview(original: np.ndarray, enhanced: np.ndarray, output_path: Path) -> None:
    h1, w1 = original.shape[:2]
    h2, w2 = enhanced.shape[:2]
    h = max(h1, h2)
    w = max(w1, w2)
    left = cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_AREA)
    preview = np.hstack([left, right])
    cv2.putText(preview, "Original", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(preview, "Enhanced", (w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imwrite(str(output_path), preview)


def process_image(preprocessor: AircraftPreprocessor, image_path: Path, output_dir: Path, enable_deblur: bool) -> float:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    started = time.perf_counter()
    result = preprocessor.preprocess(image, enable_deblur=enable_deblur)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    stem = image_path.stem
    cv2.imwrite(str(output_dir / f"{stem}_original.jpg"), result.original_image)
    cv2.imwrite(str(output_dir / f"{stem}_enhanced.jpg"), result.enhanced_image)
    np.save(str(output_dir / f"{stem}_normalized.npy"), result.normalized_image)
    _save_preview(result.original_image, result.enhanced_image, output_dir / f"{stem}_preview.jpg")

    print(
        f"{image_path.name}: blur_var={result.blur_variance:.2f}, contrast_std={result.contrast_std:.2f}, "
        f"blur_severity={result.blur_severity:.2f}, contrast_severity={result.contrast_severity:.2f}, "
        f"deblur={result.applied_deblur}, time={elapsed_ms:.1f}ms"
    )
    return elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--enable-deblur", action="store_true", help="Enable lightweight deblurring")
    parser.add_argument("--target-long-side", type=int, default=768)
    parser.add_argument("--benchmark", action="store_true", help="Print average runtime")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = AircraftPreprocessor(
        AircraftEnhancementConfig(
            target_long_side=args.target_long_side,
            enable_deblur=args.enable_deblur,
        )
    )

    image_paths = []
    if input_path.is_dir():
        image_paths = sorted([p for p in input_path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    else:
        image_paths = [input_path]

    if not image_paths:
        raise SystemExit("No images found")

    times = []
    for path in image_paths:
        times.append(process_image(preprocessor, path, output_dir, args.enable_deblur))

    if args.benchmark:
        print(f"Average preprocessing time: {sum(times) / len(times):.1f}ms over {len(times)} image(s)")


if __name__ == "__main__":
    main()
