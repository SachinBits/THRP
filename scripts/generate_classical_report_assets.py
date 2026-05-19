#!/usr/bin/env python3
"""Generate report assets for the classical baseline study.

Builds figures for Decision Tree, KNN, Naive Bayes, and SVM across the four
superclasses and stores them under outputs/classical_report_assets.
"""
from pathlib import Path
import json
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "baselines"
OUTPUTS_DIR = ROOT / "outputs"
ASSET_DIR = OUTPUTS_DIR / "classical_report_assets"
METHODS = OrderedDict([
    ("decision_tree", "Decision Tree"),
    ("knn", "K-Nearest Neighbors"),
    ("bayes", "Naive Bayes"),
    ("svm", "Support Vector Machine"),
])
SUPERCLASSES = ["fighter", "cargo", "helicopter", "bomber"]

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics():
    data = {}
    for sc in SUPERCLASSES:
        data[sc] = {}
        for method_key, method_label in METHODS.items():
            path = RESULTS_DIR / method_key / f"metrics_{sc}.json"
            if path.exists():
                data[sc][method_key] = load_json(path)
    return data


def class_names_for(sc: str):
    path = ROOT / "results" / "embeddings" / sc / "class_names.json"
    if path.exists():
        return load_json(path)
    path = ROOT / "results" / "features" / sc / "class_names.json"
    if path.exists():
        return load_json(path)
    return []


def plot_accuracy_by_superclass(metrics):
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(SUPERCLASSES))
    width = 0.18

    for idx, (method_key, method_label) in enumerate(METHODS.items()):
        values = [metrics[sc][method_key]["metrics"]["accuracy"] for sc in SUPERCLASSES]
        bars = ax.bar(x + idx * width, values, width, label=method_label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(METHODS) - 1) / 2)
    ax.set_xticklabels([sc.title() for sc in SUPERCLASSES])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Classical Method Accuracy by Superclass")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = ASSET_DIR / "accuracy_by_superclass.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_average_metrics(metrics):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    stat_keys = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1_score", "F1-Score")]
    colors = ["#2C7BE5", "#00A878", "#FF8C42", "#8E5CE6"]

    for ax, (stat_key, stat_label), color in zip(axes, stat_keys, colors):
        values = []
        for method_key in METHODS:
            per_sc = [metrics[sc][method_key]["metrics"][stat_key] for sc in SUPERCLASSES]
            values.append(float(np.mean(per_sc)))
        bars = ax.bar(list(METHODS.values()), values, color=color, alpha=0.9)
        ax.set_title(f"Mean {stat_label}")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=18)

    fig.suptitle("Average Classical Metrics Across Superclasses", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = ASSET_DIR / "average_metrics.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_matrix_page(sc: str, metrics_for_sc):
    class_names = class_names_for(sc)
    if not class_names:
        first_method = next(iter(metrics_for_sc.values()))
        class_names = [str(i) for i in range(len(first_method["metrics"]["confusion_matrix"]))]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f"{sc.title()} Superclass: Confusion Matrices", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for idx, (method_key, method_label) in enumerate(METHODS.items()):
        ax = axes[idx]
        blob = metrics_for_sc.get(method_key)
        if not blob:
            ax.axis("off")
            continue
        cm = np.array(blob["metrics"]["confusion_matrix"])
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums
        sns.heatmap(
            cm_norm,
            annot=cm,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar=True,
            square=True,
            linewidths=0.25,
            annot_kws={"size": 4},
        )
        ax.set_title(method_label, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=6)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)

    for idx in range(len(METHODS), 4):
        axes[idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = ASSET_DIR / f"confusion_{sc}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def build_summary(metrics):
    summary = {"superclasses": SUPERCLASSES, "methods": list(METHODS.keys()), "averages": {}}
    for method_key, method_label in METHODS.items():
        summary["averages"][method_key] = {}
        for stat in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            vals = [metrics[sc][method_key]["metrics"][stat] for sc in SUPERCLASSES]
            summary["averages"][method_key][stat] = float(np.mean(vals))
    return summary


def main():
    ensure_dir()
    metrics = load_metrics()
    if not metrics:
        raise SystemExit("No baseline metrics found under results/baselines")

    generated = []
    generated.append(plot_accuracy_by_superclass(metrics))
    generated.append(plot_average_metrics(metrics))
    for sc in SUPERCLASSES:
        generated.append(plot_confusion_matrix_page(sc, metrics.get(sc, {})))

    summary = build_summary(metrics)
    summary["generated_assets"] = [p.name for p in generated if p]
    with open(ASSET_DIR / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
