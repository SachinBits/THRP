"""Shared helpers for classical baseline scripts."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from skimage.feature import hog


DEFAULT_SUPERCLASSES = ["fighter", "cargo", "helicopter", "bomber"]


def extract_hog(image: np.ndarray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations: int = 9) -> np.ndarray:
    # ensure a fixed input size so HOG produces consistent-length vectors
    target_size = (256, 256)
    if (image.shape[1], image.shape[0]) != target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
    )


def extract_color_hist(image: np.ndarray, bins: int = 32) -> np.ndarray:
    # resize to fixed size for consistency
    target_size = (256, 256)
    if (image.shape[1], image.shape[0]) != target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    chans = cv2.split(image)
    histograms = []
    for ch in chans:
        hist = cv2.calcHist([ch], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    return np.concatenate(histograms)


def extract_combined(image: np.ndarray) -> np.ndarray:
    return np.concatenate([extract_hog(image), extract_color_hist(image)])


def infer_class_names(specialist_root: Path, y: np.ndarray) -> List[str]:
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

    return [f"Class_{index}" for index in sorted(set(int(v) for v in y))]


def load_class_names_from_config(specialist_root: Path) -> List[str]:
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


def load_specialist_dataset(specialist_root: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    images = []
    labels = []

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
                with open(lbl_path, "r", encoding="utf-8") as handle:
                    line = handle.readline().strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
            except Exception:
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            try:
                images.append(extract_combined(image))
                labels.append(class_id)
            except Exception:
                continue

    if not images:
        raise RuntimeError(f"No images found under {specialist_root}")

    x = np.stack(images)
    y = np.array(labels)
    class_names = infer_class_names(specialist_root, y)
    return x, y, class_names


def split_data(x: np.ndarray, y: np.ndarray, test_size: float, random_state: int = 0):
    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_state)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score=None, labels: List[int] = None) -> Dict:
    if labels is None:
        labels = sorted(list(set(int(v) for v in y_true)))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    if y_score is not None:
        try:
            y_true_binarized = label_binarize(y_true, classes=labels)
            if y_score.ndim == 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true_binarized, y_score))
            elif y_score.shape[1] == len(labels):
                metrics["roc_auc"] = float(roc_auc_score(y_true_binarized, y_score, average="macro", multi_class="ovr"))
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def train_and_evaluate_classifier(model, x_train, x_test, y_train, y_test, labels: List[int]) -> Dict:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    y_score = None
    try:
        y_score = model.predict_proba(x_test)
    except Exception:
        try:
            y_score = model.decision_function(x_test)
        except Exception:
            y_score = None

    return compute_metrics(y_test, y_pred, y_score=y_score, labels=labels)


def save_model(model, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def cluster_majority_mapping(cluster_ids: np.ndarray, y_true: np.ndarray) -> Dict[int, int]:
    mapping = {}
    for cluster_id in sorted(set(int(v) for v in cluster_ids)):
        if cluster_id == -1:
            continue
        member_labels = y_true[cluster_ids == cluster_id]
        if len(member_labels) == 0:
            continue
        values, counts = np.unique(member_labels, return_counts=True)
        mapping[cluster_id] = int(values[np.argmax(counts)])
    return mapping


def dbscan_predictions(x: np.ndarray, y_true: np.ndarray, eps: float = 0.7, min_samples: int = 5):
    from sklearn.cluster import DBSCAN

    model = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_ids = model.fit_predict(x)
    mapping = cluster_majority_mapping(cluster_ids, y_true)
    default_label = int(np.bincount(y_true).argmax())
    y_pred = np.array([mapping.get(int(cluster_id), default_label) if cluster_id != -1 else default_label for cluster_id in cluster_ids])
    return y_pred, cluster_ids, mapping
