#!/usr/bin/env python3
"""Generate actual report figures from the trained specialist CNN results.

Outputs:
- outputs/report_assets/finetune_metrics_comparison.png
- outputs/report_assets/ensemble_accuracy_comparison.png
- outputs/report_assets/confusion_<superclass>.png
- outputs/report_assets/prediction_montage.png
"""
from pathlib import Path
import json
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "baselines" / "cnn"
OUTPUTS = ROOT / "outputs"
ASSET_DIR = OUTPUTS / "report_assets"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def class_names_for(sc: str):
    path = ROOT / "results" / "embeddings" / sc / "class_names.json"
    if path.exists():
        return load_json(path)
    path = ROOT / "results" / "features" / sc / "class_names.json"
    if path.exists():
        return load_json(path)
    return []


def ensure_dir():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def plot_finetune_summary(metrics_by_sc):
    superclasses = list(metrics_by_sc.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    stats = ["accuracy", "precision", "recall", "f1_score"]
    colors = ["#2C7BE5", "#00A878", "#FF8C42", "#8E5CE6"]

    for ax, stat, color in zip(axes, stats, colors):
        values = [metrics_by_sc[sc]["metrics"][stat] for sc in superclasses]
        bars = ax.bar(superclasses, values, color=color, alpha=0.9)
        ax.set_title(stat.replace("_", " ").title())
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Finetuned ResNet50 Specialist Performance", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = ASSET_DIR / "finetune_metrics_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ensemble_accuracy(metrics_by_sc):
    superclasses = list(metrics_by_sc.keys())
    values = [metrics_by_sc[sc]["accuracy"] for sc in superclasses]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(superclasses, values, color="#1F77B4", alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy of Updated Ensemble")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = ASSET_DIR / "ensemble_accuracy_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_matrix(sc, metrics_blob):
    cm = np.array(metrics_blob["metrics"]["confusion_matrix"])
    class_names = class_names_for(sc)
    if not class_names:
        class_names = [str(i) for i in range(cm.shape[0])]
    fig, ax = plt.subplots(figsize=(11, 9))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True, square=True, linewidths=0.4)
    ax.set_title(f"{sc.title()} Specialist Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    out = ASSET_DIR / f"confusion_{sc}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _fit_text(draw, text, max_width, font):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def create_prediction_montage():
    pred_files = sorted(OUTPUTS.glob("pred_*.jpg")) + sorted(OUTPUTS.glob("pred_*.png"))
    if not pred_files:
        return None
    chosen = pred_files[:6]
    cols = 3
    rows = math.ceil(len(chosen) / cols)
    tile_w, tile_h = 420, 280
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h + 80), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default()
    caption_font = ImageFont.load_default()
    draw.text((20, 16), "Qualitative Prediction Examples", fill="black", font=title_font)
    draw.text((20, 40), "Sample outputs saved during inference and preprocessing", fill="#444444", font=caption_font)

    for idx, img_path in enumerate(chosen):
        row = idx // cols
        col = idx % cols
        x0 = col * tile_w + 10
        y0 = row * tile_h + 70
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        thumb = ImageOps.contain(img, (tile_w - 20, tile_h - 60))
        paste_x = x0 + (tile_w - 20 - thumb.width) // 2
        paste_y = y0 + 5
        canvas.paste(thumb, (paste_x, paste_y))
        caption = _fit_text(draw, img_path.name, tile_w - 20, caption_font)
        draw.text((x0 + 5, y0 + tile_h - 45), caption, fill="#222222", font=caption_font)
    out = ASSET_DIR / "prediction_montage.png"
    canvas.save(out)
    return out


def main():
    ensure_dir()
    metrics_by_sc = {}
    for sc in ["fighter", "cargo", "helicopter", "bomber"]:
        path = RESULTS / f"metrics_{sc}.json"
        if path.exists():
            metrics_by_sc[sc] = load_json(path)

    if not metrics_by_sc:
        raise SystemExit("No finetune metrics found in results/baselines/cnn")

    plot_finetune_summary(metrics_by_sc)
    ensemble_blob = {sc: load_json(RESULTS / f"ensemble_eval_{sc}.json") for sc in metrics_by_sc if (RESULTS / f"ensemble_eval_{sc}.json").exists()}
    if ensemble_blob:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = list(ensemble_blob.keys())
        values = [ensemble_blob[sc]["accuracy"] for sc in labels]
        bars = ax.bar(labels, values, color="#E76F51", alpha=0.9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Updated Ensemble Validation Accuracy")
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        fig.savefig(ASSET_DIR / "ensemble_accuracy_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    for sc, blob in metrics_by_sc.items():
        plot_confusion_matrix(sc, blob)

    create_prediction_montage()

    summary = {
        "generated_assets": sorted([p.name for p in ASSET_DIR.glob("*.png")]),
        "superclasses": sorted(metrics_by_sc.keys()),
    }
    with open(ASSET_DIR / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
