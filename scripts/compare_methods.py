"""Run comparisons between the proposed two-stage THRP method and classical ML methods.

This script trains Decision Tree, K-NN, Naive Bayes, and SVM classifiers on a
specialist specialist dataset (default: fighter) using HOG+color features, then
evaluates them with Accuracy, Precision, Recall, F1, and ROC-AUC (macro).

Usage:
    python3 scripts/compare_methods.py --dataset yolo_specialist_datasets --superclass fighter --out results/experiments

Requirements: scikit-image, scikit-learn, opencv-python, joblib, numpy
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import joblib
from extract_features import extract_combined


def load_dataset(specialist_root: Path):
    images = []
    labels = []

    for split in ("train", "val"):
        img_dir = specialist_root / "images" / split
        lbl_dir = specialist_root / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                # try other extensions
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue
            # read first class id from label file
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

            feat = extract_combined(img)
            images.append(feat)
            labels.append(cls)

    if len(images) == 0:
        raise RuntimeError(f"No images found under {specialist_root}")

    X = np.stack(images)
    y = np.array(labels)
    return X, y


def evaluate_model(clf, X_test, y_test, labels):
    y_pred = clf.predict(X_test)
    y_score = None
    try:
        y_score = clf.predict_proba(X_test)
    except Exception:
        try:
            # decision function fallback
            y_score = clf.decision_function(X_test)
        except Exception:
            y_score = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    roc = None
    if y_score is not None:
        try:
            y_test_b = label_binarize(y_test, classes=labels)
            if y_score.ndim == 1:
                # binary
                roc = roc_auc_score(y_test_b, y_score)
            else:
                roc = roc_auc_score(y_test_b, y_score, average='macro', multi_class='ovr')
        except Exception:
            roc = None

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yolo_specialist_datasets")
    parser.add_argument("--superclass", default="fighter")
    parser.add_argument("--out", default="results/experiments")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    root = Path(args.dataset)
    sc = args.superclass.lower()
    specialist_dir = root / sc
    if not specialist_dir.exists():
        raise SystemExit(f"Specialist dataset not found: {specialist_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset and extracting features (this may take a while)...")
    X, y = load_dataset(specialist_dir)
    labels = sorted(list(set(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=0)

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=0),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NaiveBayes": GaussianNB(),
        "SVM": SVC(kernel='rbf', probability=True, random_state=0)
    }

    results = {}
    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        joblib.dump(clf, out_dir / f"{name}_{sc}.joblib")
        res = evaluate_model(clf, X_test, y_test, labels)
        results[name] = res
        print(f"  {name} results: {res}")

    # Save results
    import json
    with open(out_dir / f"results_{sc}.json", "w") as f:
        json.dump({"labels": labels, "results": results}, f, indent=2)

    print(f"Results saved to {out_dir}")


if __name__ == '__main__':
    main()
