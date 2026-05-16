#!/usr/bin/env python3
"""Run YOLO validation and save per-class and summary metrics to logs/metrics/."""
import argparse
import json
from datetime import datetime
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as e:
    print("Please install ultralytics in your environment: pip install ultralytics")
    raise

import numpy as np


def save_metrics(model_path: Path, data_yaml: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "metrics.jsonl"
    csv_path = out_dir / "metrics_summary.csv"

    print(f"Running validation: model={model_path}, data={data_yaml}")
    m = YOLO(str(model_path))
    res = m.val(data=str(data_yaml), verbose=False)

    # Extract per-class map50 if available
    per_class_map = None
    overall_map50 = None
    try:
        arr = np.array(res.box.map50)
        per_class_map = arr.tolist()
        overall_map50 = float(np.nanmean(arr))
    except Exception:
        # fallback if attribute differs
        try:
            overall_map50 = float(res.metrics.get('map50', float('nan')))
        except Exception:
            overall_map50 = None

    # Map class names to values
    class_map = {}
    try:
        names = res.names
        if per_class_map is not None and len(per_class_map) == len(names):
            for i, name in names.items():
                class_map[name] = per_class_map[int(i)]
    except Exception:
        names = None

    record = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'model': str(model_path),
        'data': str(data_yaml),
        'overall_map50': overall_map50,
        'per_class_map50': class_map,
    }

    # Append JSONL
    with open(jsonl_path, 'a') as fh:
        fh.write(json.dumps(record) + "\n")

    # Update CSV summary (one-line per run)
    header = [
        'timestamp', 'model', 'data', 'overall_map50', 'num_classes', 'best_class', 'best_map50', 'worst_class', 'worst_map50'
    ]
    best_class = ''
    best_map = ''
    worst_class = ''
    worst_map = ''
    num_classes = 0
    if class_map:
        items = sorted(class_map.items(), key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
        num_classes = len(items)
        best_class, best_map = items[0]
        worst_class, worst_map = items[-1]
    row = [record['timestamp'], record['model'], record['data'], record['overall_map50'], num_classes, best_class, best_map, worst_class, worst_map]

    write_header = not csv_path.exists()
    with open(csv_path, 'a') as fh:
        if write_header:
            fh.write(','.join(header) + '\n')
        fh.write(','.join([str(x) for x in row]) + '\n')

    print(f"Saved metrics to: {jsonl_path}, {csv_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Validate YOLO model and save metrics')
    p.add_argument('--model', required=True, help='Path to .pt model')
    p.add_argument('--data', required=True, help='Path to data YAML')
    p.add_argument('--out', default='logs/metrics', help='Output directory to store metrics')
    args = p.parse_args()

    save_metrics(Path(args.model), Path(args.data), Path(args.out))
