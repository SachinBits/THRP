#!/usr/bin/env python3
"""Run DBSCAN baselines for Fighter, Cargo, Helicopter, and Bomber datasets."""

import argparse
from pathlib import Path

from classical_baseline_common import DEFAULT_SUPERCLASSES, compute_metrics, dbscan_predictions, load_specialist_dataset, save_json


def run_superclass(dataset_root: Path, out_dir: Path, superclass: str, eps: float, min_samples: int) -> Path:
    specialist_dir = dataset_root / superclass
    x, y, _ = load_specialist_dataset(specialist_dir)
    labels = sorted(list(set(int(value) for value in y)))
    y_pred, cluster_ids, mapping = dbscan_predictions(x, y, eps=eps, min_samples=min_samples)
    metrics = compute_metrics(y, y_pred, labels=labels)

    payload = {
        "superclass": superclass,
        "method": "DBSCAN",
        "params": {
            "eps": eps,
            "min_samples": min_samples,
        },
        "dataset_info": {
            "total_samples": int(len(x)),
            "num_classes": int(len(labels)),
            "num_clusters": int(len(set(int(value) for value in cluster_ids)) - (1 if -1 in cluster_ids else 0)),
            "noise_samples": int((cluster_ids == -1).sum()),
        },
        "cluster_to_label_map": {str(key): int(value) for key, value in mapping.items()},
        "metrics": metrics,
    }

    out_path = out_dir / "dbscan" / f"metrics_{superclass}.json"
    save_json(out_path, payload)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DBSCAN baselines for all aircraft superclasses")
    parser.add_argument("--dataset", default="yolo_specialist_datasets")
    parser.add_argument("--out", default="results/baselines")
    parser.add_argument("--eps", type=float, default=0.7)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--superclasses", nargs="*", default=DEFAULT_SUPERCLASSES)
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    out_dir = Path(args.out)
    for superclass in args.superclasses:
        run_superclass(dataset_root, out_dir, superclass, args.eps, args.min_samples)
        print(f"Saved DBSCAN results for {superclass}")


if __name__ == "__main__":
    main()
