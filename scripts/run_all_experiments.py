"""Run comprehensive experiments: THRP + DT, KNN, Naive Bayes, SVM.

Trains classical ML methods on specialist datasets and evaluates against THRP.
Computes 5 evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
Generates confusion matrices and saves all results.

Usage:
    python3 scripts/run_all_experiments.py --superclass fighter --out results/comprehensive
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                              roc_auc_score, confusion_matrix, classification_report)
from sklearn.preprocessing import label_binarize
import joblib

from extract_features import extract_combined

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_specialist_dataset(specialist_root: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load images and labels from specialist dataset."""
    images = []
    labels = []
    class_names = []

    for split in ("train", "val"):
        img_dir = specialist_root / "images" / split
        lbl_dir = specialist_root / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            logger.warning(f"Split not found: {split} at {specialist_root}")
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            
            try:
                with open(lbl_path, "r") as f:
                    line = f.readline().strip()
                if not line:
                    continue
                cls = int(line.split()[0])
            except Exception:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            try:
                feat = extract_combined(img)
                images.append(feat)
                labels.append(cls)
            except Exception as e:
                logger.debug(f"Failed to extract features from {img_path}: {e}")
                continue

    if len(images) == 0:
        raise RuntimeError(f"No images found under {specialist_root}")

    X = np.stack(images)
    y = np.array(labels)
    
    # Get class names from model config if available
    yaml_path = specialist_root / f"{specialist_root.name}_config.yaml"
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            class_names = [str(cfg['names'].get(i, f"Class_{i}")) for i in sorted(cfg['names'].keys())]
        except Exception:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y)))]
    else:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y)))]
    
    return X, y, class_names


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score=None, 
                   labels: List[int] = None) -> Dict:
    """Compute all 5 evaluation metrics."""
    if labels is None:
        labels = sorted(list(set(y_true)))
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    roc = None
    if y_score is not None:
        try:
            y_true_b = label_binarize(y_true, classes=labels)
            if y_score.ndim == 1:
                roc = roc_auc_score(y_true_b, y_score)
            else:
                # Ensure y_score shape matches
                if y_score.shape[1] == len(labels):
                    roc = roc_auc_score(y_true_b, y_score, average='macro', multi_class='ovr')
        except Exception as e:
            logger.debug(f"ROC-AUC computation failed: {e}")
            roc = None
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc) if roc is not None else None,
        "confusion_matrix": conf_matrix.tolist()
    }


def train_classical_methods(X_train: np.ndarray, X_test: np.ndarray, 
                           y_train: np.ndarray, y_test: np.ndarray,
                           labels: List[int]) -> Dict:
    """Train and evaluate Decision Tree, KNN, Naive Bayes, SVM."""
    results = {}
    
    models_config = {
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=0),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=0, max_iter=1000)
    }
    
    for method_name, clf in models_config.items():
        logger.info(f"Training {method_name}...")
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # Get probability scores for ROC-AUC
            y_score = None
            try:
                y_score = clf.predict_proba(X_test)
            except Exception:
                try:
                    y_score = clf.decision_function(X_test)
                except Exception:
                    pass
            
            metrics = compute_metrics(y_test, y_pred, y_score, labels)
            results[method_name] = metrics
            logger.info(f"  ✓ {method_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        except Exception as e:
            logger.error(f"  ✗ {method_name} failed: {e}")
            results[method_name] = {"error": str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive experiments on specialist datasets")
    parser.add_argument("--dataset", default="yolo_specialist_datasets", help="Dataset root directory")
    parser.add_argument("--superclass", default="fighter", help="Superclass to run experiments on")
    parser.add_argument("--out", default="results/comprehensive", help="Output directory for results")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    args = parser.parse_args()

    root = Path(args.dataset)
    sc = args.superclass.lower()
    specialist_dir = root / sc
    
    if not specialist_dir.exists():
        logger.error(f"Specialist dataset not found: {specialist_dir}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {sc.upper()} specialist dataset and extracting features...")
    try:
        X, y, class_names = load_specialist_dataset(specialist_dir)
    except RuntimeError as e:
        logger.error(str(e))
        return

    logger.info(f"Loaded {len(X)} images with {len(np.unique(y))} classes")
    labels = sorted(list(set(y)))
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=0
    )

    # Train classical methods
    logger.info("\n" + "="*80)
    logger.info("CLASSICAL SUPERVISED LEARNING METHODS")
    logger.info("="*80)
    classical_results = train_classical_methods(X_train, X_test, y_train, y_test, labels)

    # Compile all results
    all_results = {
        "superclass": sc,
        "dataset_info": {
            "total_samples": len(X),
            "num_classes": len(labels),
            "class_names": class_names[:len(labels)],
            "train_size": len(X_train),
            "test_size": len(X_test)
        },
        "methods": classical_results
    }

    # Save results
    results_file = out_dir / f"metrics_{sc}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n✓ Results saved to {results_file}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    for method_name, metrics in classical_results.items():
        if "error" not in metrics:
            logger.info(f"\n{method_name}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
            if metrics['roc_auc'] is not None:
                logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        else:
            logger.error(f"\n{method_name}: FAILED - {metrics['error']}")


if __name__ == '__main__':
    main()
