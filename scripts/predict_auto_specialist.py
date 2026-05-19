#!/usr/bin/env python3
"""Run the generalist detector, resolve superclass, then run the specialist predictor automatically.

Usage:
  python3 scripts/predict_auto_specialist.py --image path/to/img.jpg
"""
from pathlib import Path
import argparse
import sys

from aircraft_superclass_map import AIRCRAFT_TO_SUPERCLASS


def resolve_superclass_from_name(aircraft_name: str) -> str | None:
    # The mapping keys may be uppercase; try exact and uppercase
    if not aircraft_name:
        return None
    if aircraft_name in AIRCRAFT_TO_SUPERCLASS:
        return AIRCRAFT_TO_SUPERCLASS[aircraft_name]
    key = aircraft_name.strip()
    if key.upper() in AIRCRAFT_TO_SUPERCLASS:
        return AIRCRAFT_TO_SUPERCLASS[key.upper()]
    # try various casings
    for k in AIRCRAFT_TO_SUPERCLASS:
        if k.lower() == key.lower():
            return AIRCRAFT_TO_SUPERCLASS[k]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Image to run through pipeline")
    parser.add_argument("--generalist-model", default="models/generalist_model.pt", help="YOLO generalist model path")
    parser.add_argument("--dataset-root", default="yolo_specialist_datasets")
    parser.add_argument("--out", default="results/baselines")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        raise SystemExit(2)

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print("Error importing ultralytics YOLO: ", exc)
        raise

    model = YOLO(str(args.generalist_model))
    # run inference
    results = model.predict(source=str(image_path), conf=args.conf, verbose=False)
    if not results:
        print("No detections from generalist model")
        raise SystemExit(1)

    res = results[0]
    # extract boxes, classes, confidences
    try:
        cls_array = res.boxes.cls.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
    except Exception:
        # fallback for older ultralytics versions
        cls_array = []
        confs = []
        for b in res.boxes:
            cls_array.append(int(b.cls))
            confs.append(float(b.conf))

    if len(cls_array) == 0:
        print("No boxes predicted")
        raise SystemExit(1)

    # pick highest-confidence detection
    best_idx = int(confs.argmax()) if hasattr(confs, "argmax") else max(range(len(confs)), key=lambda i: confs[i])
    best_cls = int(cls_array[best_idx])

    # map class index to name via model.names dict
    names = getattr(model, "names", None)
    predicted_name = None
    if names is not None and best_cls in names:
        predicted_name = names[best_cls]
    else:
        predicted_name = str(best_cls)

    superclass = resolve_superclass_from_name(predicted_name)
    if superclass is None:
        print(f"Could not resolve superclass for predicted: {predicted_name}")
        raise SystemExit(2)

    superclass_folder = superclass.lower()
    print(f"Generalist predicted: {predicted_name} -> superclass {superclass_folder}")

    # call specialist predictor
    from predict_decision_tree_baseline import predict_image

    model_path = Path(args.out) / "decision_tree" / f"decision_tree_{superclass_folder}.joblib"
    if not model_path.exists():
        print(f"Specialist model not found: {model_path}")
        raise SystemExit(2)

    dataset_root = Path(args.dataset_root)
    specialist_root = dataset_root / superclass_folder

    predicted_name, predicted_index, confidence = predict_image(model_path, image_path, specialist_root)
    print(f"Specialist prediction: {predicted_name} (index {predicted_index})", end="")
    if confidence is not None:
        print(f", confidence {confidence:.4f}")
    else:
        print("")


if __name__ == "__main__":
    main()
