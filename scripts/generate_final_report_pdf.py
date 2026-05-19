#!/usr/bin/env python3
"""Generate the final THRP PDF report from the real report assets.

This composes the updated markdown/report content and embeds the generated charts
and confusion matrices into a clean PDF submission artifact.
"""
from pathlib import Path
from datetime import datetime
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
ASSET_DIR = OUTPUTS / "report_assets"
REPORT_MD = OUTPUTS / "FULL_REPORT.md"
PDF_OUT = OUTPUTS / "THRP_FINAL_REPORT.pdf"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def add_text_page(pdf, title, lines, footer=None):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 0.96
    ax.text(0.05, y, title, fontsize=18, fontweight="bold")
    y -= 0.06
    for line in lines:
        ax.text(0.05, y, line, fontsize=10, va="top")
        y -= 0.04
    if footer:
        ax.text(0.05, 0.04, footer, fontsize=8, color="gray")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf, title, image_path, caption=None, zoom=1.0):
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


def add_metrics_table_page(pdf, metrics_by_sc):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.05, 0.96, "Finetuned Specialist Metrics", fontsize=18, fontweight="bold")

    headers = ["Superclass", "Accuracy", "Precision", "Recall", "F1"]
    rows = []
    for sc, blob in metrics_by_sc.items():
        m = blob["metrics"]
        rows.append([
            sc.title(),
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1_score']:.4f}",
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


def generate_pdf():
    metrics_by_sc = {}
    for sc in ["fighter", "cargo", "helicopter", "bomber"]:
        path = ROOT / "results" / "baselines" / "cnn" / f"metrics_{sc}.json"
        if path.exists():
            metrics_by_sc[sc] = load_json(path)

    summary = load_json(ASSET_DIR / "summary.json") if (ASSET_DIR / "summary.json").exists() else {}

    with PdfPages(PDF_OUT) as pdf:
        add_text_page(
            pdf,
            "THRP Final Report",
            [
                "Two-Stage Hierarchical Recognition Pipeline",
                "",
                "This PDF is generated from the real finetuned specialist results and the",
                "actual figures in outputs/report_assets.",
                "",
                "Included components:",
                "- Finetuned ResNet50 specialist CNN metrics",
                "- Updated ensemble accuracy",
                "- Confusion matrices for fighter, cargo, helicopter, and bomber",
                "- Prediction montage with saved qualitative outputs",
                "",
                "The markdown version is available at outputs/FULL_REPORT.md.",
            ],
            footer=f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )

        add_metrics_table_page(pdf, metrics_by_sc)
        add_image_page(pdf, "Finetuned CNN Performance", ASSET_DIR / "finetune_metrics_comparison.png")
        add_image_page(pdf, "Updated Ensemble Accuracy", ASSET_DIR / "ensemble_accuracy_comparison.png")

        for sc in ["fighter", "cargo", "helicopter", "bomber"]:
            path = ASSET_DIR / f"confusion_{sc}.png"
            if path.exists():
                add_image_page(pdf, f"{sc.title()} Confusion Matrix", path)

        montage = ASSET_DIR / "prediction_montage.png"
        if montage.exists():
            add_image_page(pdf, "Prediction Montage", montage, caption="Saved inference/preprocessing examples")

        if summary:
            add_text_page(
                pdf,
                "Report Assets Summary",
                [
                    f"Generated assets: {', '.join(summary.get('generated_assets', []))}",
                    f"Superclasses covered: {', '.join(summary.get('superclasses', []))}",
                    "",
                    "Note: fighter and helicopter remain the weaker specialists; bomber is strongest.",
                ],
                footer="THRP final submission package",
            )

        info = pdf.infodict()
        info["Title"] = "THRP Final Report"
        info["Author"] = "Team THRP"
        info["Subject"] = "Two-Stage Hierarchical Recognition Pipeline"
        info["CreationDate"] = datetime.now()

    print(f"Saved PDF report to {PDF_OUT}")


if __name__ == "__main__":
    generate_pdf()
