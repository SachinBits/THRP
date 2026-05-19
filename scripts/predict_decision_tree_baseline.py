#!/usr/bin/env python3
"""Predict a single aircraft class using a trained Decision Tree baseline."""

import argparse
from pathlib import Path

import cv2
import joblib

from classical_baseline_common import extract_combined, load_class_names_from_config


def predict_image(model_path: Path, image_path: Path, specialist_root: Path):
    model = joblib.load(model_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    features = extract_combined(image).reshape(1, -1)
    prediction = int(model.predict(features)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(features)[0]
        except Exception:
            probabilities = None

    class_names = load_class_names_from_config(specialist_root)
    if class_names and prediction < len(class_names):
        predicted_name = class_names[prediction]
    else:
        predicted_name = str(prediction)

    confidence = None
    if probabilities is not None and prediction < len(probabilities):
        confidence = float(probabilities[prediction])

    return predicted_name, prediction, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one image with a trained Decision Tree baseline")
    parser.add_argument("--image", required=True, help="Path to the image to classify")
    parser.add_argument("--superclass", required=True, choices=["fighter", "cargo", "helicopter", "bomber"], help="Superclass the model was trained on")
    parser.add_argument("--model", help="Path to the saved .joblib model")
    parser.add_argument("--dataset-root", default="yolo_specialist_datasets", help="Root folder for specialist datasets")
    parser.add_argument("--out", default="results/baselines", help="Baseline output directory")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    specialist_root = dataset_root / args.superclass
    model_path = Path(args.model) if args.model else Path(args.out) / "decision_tree" / f"decision_tree_{args.superclass}.joblib"
    image_path = Path(args.image)

    predicted_name, predicted_index, confidence = predict_image(model_path, image_path, specialist_root)

    print(f"Superclass: {args.superclass}")
    print(f"Prediction: {predicted_name}")
    print(f"Class index: {predicted_index}")
    if confidence is not None:
        print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
