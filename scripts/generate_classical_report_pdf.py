#!/usr/bin/env python3
"""Generate the final PDF for the classical baseline report."""
from pathlib import Path
from datetime import datetime
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
ASSET_DIR = OUTPUTS / "classical_report_assets"
PDF_OUT = OUTPUTS / "thrp_report_final1.pdf"
GENERALIST_INFO = ROOT / "models" / "generalist_training_info_kaggle.json"

METHOD_LABELS = ["Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "SVM"]
SUPERCLASSES = ["fighter", "cargo", "helicopter", "bomber"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_generalist_info():
    if GENERALIST_INFO.exists():
        return load_json(GENERALIST_INFO)
    return {}


def add_text_page(pdf, title, lines, footer=None):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.05, 0.96, title, fontsize=18, fontweight="bold")
    y = 0.90
    for line in lines:
        ax.text(0.05, y, line, fontsize=10, va="top")
        y -= 0.045
    if footer:
        ax.text(0.05, 0.04, footer, fontsize=8, color="gray")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf, title, image_path, caption=None):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.05, 0.96, title, fontsize=16, fontweight="bold")
    img = Image.open(image_path)
    ax.imshow(img)
    ax.set_position([0.05, 0.10, 0.90, 0.78])
    if caption:
        ax.text(0.05, 0.04, caption, fontsize=9, color="dimgray")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_metrics_table_page(pdf, summary):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.05, 0.96, "Average Baseline Metrics", fontsize=18, fontweight="bold")

    headers = ["Method", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    rows = []
    for method_key, method_label in [("decision_tree", "Decision Tree"), ("knn", "KNN"), ("bayes", "Naive Bayes"), ("svm", "SVM")]:
        row = summary["averages"][method_key]
        rows.append([
            method_label,
            f"{row['accuracy']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1_score']:.4f}",
            f"{row['roc_auc']:.4f}",
        ])

    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#2C7BE5")
        table[(0, i)].set_text_props(color="white", weight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_thrp_summary_page(pdf, generalist):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.05, 0.96, "THRP Proposed System", fontsize=18, fontweight="bold")

    rows = [
        ["Model type", generalist.get("model_type", "THRP generalist")],
        ["Base model", generalist.get("base_model", "YOLOv8n")],
        ["Epochs", str(generalist.get("epochs", "N/A"))],
        ["Batch size", str(generalist.get("batch_size", "N/A"))],
        ["Device", str(generalist.get("device", "N/A"))],
        ["Saved model", generalist.get("model_path", "models/generalist_model.pt")],
        ["Superclass count", str(len(generalist.get("superclasses", [])))],
        ["Status", "Proposed hierarchical pipeline; no single scalar metric saved in the current artifacts"],
    ]

    table = ax.table(cellText=rows, colLabels=["Field", "Value"], cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    for i in range(2):
        table[(0, i)].set_facecolor("#1F77B4")
        table[(0, i)].set_text_props(color="white", weight="bold")
    ax.text(
        0.05,
        0.12,
        "THRP is the proposed two-stage pipeline: a generalist routes images to the correct superclass, then specialists classify the aircraft.",
        fontsize=9,
        color="dimgray",
        wrap=True,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf():
    summary = load_json(ASSET_DIR / "summary.json")
    generalist = load_generalist_info()

    with PdfPages(PDF_OUT) as pdf:
        add_text_page(
            pdf,
            "THRP Final Report",
            [
                "Two-Stage Hierarchical Recognition Pipeline",
                "",
                "THRP is the proposed system in this project.",
                "This PDF includes the proposed THRP hierarchical pipeline and compares",
                "its classical baseline methods: Decision Tree, K-Nearest Neighbors, Naive Bayes, and Support Vector Machine.",
                "",
                "SVM is the strongest baseline on average, while Decision Tree is weakest.",
            ],
            footer=f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )

        if generalist:
            add_thrp_summary_page(pdf, generalist)

        add_metrics_table_page(pdf, summary)
        add_image_page(pdf, "Accuracy by Superclass", ASSET_DIR / "accuracy_by_superclass.png")
        add_image_page(pdf, "Average Metrics Comparison", ASSET_DIR / "average_metrics.png")

        for sc in SUPERCLASSES:
            path = ASSET_DIR / f"confusion_{sc}.png"
            if path.exists():
                add_image_page(pdf, f"{sc.title()} Confusion Matrices", path)

        add_text_page(
            pdf,
            "Discussion",
            [
                "- SVM consistently leads the classical baselines across superclasses.",
                "- KNN is the second-strongest method overall.",
                "- Naive Bayes trades speed for lower accuracy.",
                "- Decision Tree is the easiest to interpret but has the weakest generalization.",
                "",
                "The confusion matrices show the fine-grained class confusion that motivates",
                "the hierarchical THRP design.",
            ],
            footer="Classical baseline study",
        )

        info = pdf.infodict()
        info["Title"] = "THRP Classical Baseline Report"
        info["Author"] = "Team THRP"
        info["Subject"] = "Decision Tree, KNN, Naive Bayes, and SVM comparison"
        info["CreationDate"] = datetime.now()

    print(f"Saved PDF report to {PDF_OUT}")


if __name__ == "__main__":
    generate_pdf()
