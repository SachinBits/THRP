#!/usr/bin/env python3
"""Run Gaussian Naive Bayes baselines for Fighter, Cargo, Helicopter, and Bomber datasets."""

import argparse
from pathlib import Path

from sklearn.naive_bayes import GaussianNB

from classical_baseline_common import DEFAULT_SUPERCLASSES, load_specialist_dataset, save_json, save_model, split_data, train_and_evaluate_classifier


def run_superclass(dataset_root: Path, out_dir: Path, superclass: str, test_size: float) -> Path:
    specialist_dir = dataset_root / superclass
    x, y, _ = load_specialist_dataset(specialist_dir)
    labels = sorted(list(set(int(value) for value in y)))
    x_train, x_test, y_train, y_test = split_data(x, y, test_size=test_size)

    model = GaussianNB()
    metrics = train_and_evaluate_classifier(model, x_train, x_test, y_train, y_test, labels)

    payload = {
        "superclass": superclass,
        "method": "Naive Bayes",
        "dataset_info": {
            "total_samples": int(len(x)),
            "num_classes": int(len(labels)),
            "train_size": int(len(x_train)),
            "test_size": int(len(x_test)),
        },
        "metrics": metrics,
    }

    out_path = out_dir / "bayes" / f"metrics_{superclass}.json"
    save_json(out_path, payload)
    save_model(model, out_dir / "bayes" / f"bayes_{superclass}.joblib")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gaussian Naive Bayes baselines for all aircraft superclasses")
    parser.add_argument("--dataset", default="yolo_specialist_datasets")
    parser.add_argument("--out", default="results/baselines")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--superclasses", nargs="*", default=DEFAULT_SUPERCLASSES)
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    out_dir = Path(args.out)
    for superclass in args.superclasses:
        run_superclass(dataset_root, out_dir, superclass, args.test_size)
        print(f"Saved Naive Bayes results for {superclass}")


if __name__ == "__main__":
    main()
