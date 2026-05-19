"""Retrain classical baselines (Decision Tree, SVM, KNN, GaussianNB) using ResNet embeddings.

Usage:
  python3 scripts/retrain_with_embeddings.py --emb results/embeddings --out results/baselines --methods decision_tree svm knn bayes
"""
from pathlib import Path
import argparse
import json
import numpy as np

from classical_baseline_common import save_json, save_model, compute_metrics, split_data, train_and_evaluate_classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

DEFAULT_METHODS = ["decision_tree", "svm", "knn", "bayes"]


def load_embeddings_for(sc_root: Path):
    emb_path = sc_root / "embeddings.npy"
    labels_path = sc_root / "labels.npy"
    if not emb_path.exists() or not labels_path.exists():
        raise RuntimeError(f"Missing embeddings under {sc_root}")
    X = np.load(emb_path)
    y = np.load(labels_path)
    return X, y


def instantiate(method: str):
    if method == "decision_tree":
        # decision trees don't need scaling but class imbalance can be addressed
        return DecisionTreeClassifier(max_depth=15, random_state=0, class_weight="balanced")
    if method == "svm":
        # Use a scaler + SVC, and calibrate probabilities for better confidence estimates
        svc = SVC(kernel="rbf", probability=False, random_state=0, max_iter=10000)
        clf = make_pipeline(StandardScaler(), svc)
        # CalibratedClassifierCV expects an estimator supporting predict_proba or decision_function
        return CalibratedClassifierCV(clf, cv=3)
    if method == "knn":
        # KNN benefits from feature scaling
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if method == "bayes":
        # GaussianNB can work better with scaled inputs
        return make_pipeline(StandardScaler(), GaussianNB())
    raise ValueError("Unknown method")


def run_superclass(sc: str, emb_root: Path, out_root: Path, methods):
    sc_root = emb_root / sc
    X, y = load_embeddings_for(sc_root)
    labels = sorted(list(set(int(v) for v in y)))
    x_train, x_test, y_train, y_test = split_data(X, y, test_size=0.2)

    results = {}
    for method in methods:
        clf = instantiate(method)
        metrics = train_and_evaluate_classifier(clf, x_train, x_test, y_train, y_test, labels)
        payload = {
            "superclass": sc,
            "method": method,
            "dataset_info": {
                "total_samples": int(len(X)),
                "num_classes": int(len(labels)),
                "train_size": int(len(x_train)),
                "test_size": int(len(x_test)),
            },
            "metrics": metrics,
        }
        out_dir = out_root / method
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(out_dir / f"metrics_{sc}.json", payload)
        save_model(clf, out_dir / f"{method}_{sc}.joblib")
        results[method] = payload
        print(f"Saved {method} for {sc}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True, help="Root embeddings directory (results/embeddings)")
    parser.add_argument("--out", required=True, help="Output base for baselines")
    parser.add_argument("--superclasses", nargs="*", default=["fighter", "cargo", "helicopter", "bomber"])
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    args = parser.parse_args()

    emb_root = Path(args.emb)
    out_root = Path(args.out)

    all_results = {}
    for sc in args.superclasses:
        try:
            res = run_superclass(sc, emb_root, out_root, args.methods)
            all_results[sc] = res
        except Exception as exc:
            print(f"Failed {sc}: {exc}")

    # save summary
    save_json(out_root / "retrain_summary.json", all_results)


if __name__ == "__main__":
    main()
