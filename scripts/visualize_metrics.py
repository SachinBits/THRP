"""Visualize experiment results: generate metric graphs, confusion matrices, and comparisons.

Usage:
    python3 scripts/visualize_metrics.py --results results/comprehensive/metrics_fighter.json --out results/comprehensive/visualizations
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_results(results_file: Path) -> Dict:
    """Load experiment results from JSON."""
    with open(results_file) as f:
        return json.load(f)


def plot_metrics_comparison(results: Dict, out_dir: Path):
    """Create bar plots comparing all metrics across methods."""
    methods = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for method_name, metrics in results['methods'].items():
        if 'error' not in metrics:
            methods.append(method_name)
            accuracies.append(metrics['accuracy'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1_score'])
    
    if not methods:
        print("No valid results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Evaluation Metrics Comparison - {results['superclass'].upper()} Specialist", 
                 fontsize=14, fontweight='bold')
    
    x_pos = np.arange(len(methods))
    width = 0.7
    
    # Accuracy
    axes[0, 0].bar(x_pos, accuracies, width, color='steelblue', alpha=0.8)
    axes[0, 0].set_ylabel('Accuracy', fontweight='bold')
    axes[0, 0].set_ylim([0, 1.0])
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(methods, rotation=15, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Precision
    axes[0, 1].bar(x_pos, precisions, width, color='forestgreen', alpha=0.8)
    axes[0, 1].set_ylabel('Precision', fontweight='bold')
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(methods, rotation=15, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(precisions):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Recall
    axes[1, 0].bar(x_pos, recalls, width, color='coral', alpha=0.8)
    axes[1, 0].set_ylabel('Recall', fontweight='bold')
    axes[1, 0].set_ylim([0, 1.0])
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(methods, rotation=15, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(recalls):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # F1-Score
    axes[1, 1].bar(x_pos, f1_scores, width, color='mediumpurple', alpha=0.8)
    axes[1, 1].set_ylabel('F1-Score', fontweight='bold')
    axes[1, 1].set_ylim([0, 1.0])
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(methods, rotation=15, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(f1_scores):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(out_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: metrics_comparison.png")
    plt.close(fig)


def plot_confusion_matrices(results: Dict, out_dir: Path):
    """Create confusion matrix heatmaps for each method."""
    superclass = results['superclass'].upper()
    class_names = results['dataset_info']['class_names']
    
    valid_methods = [m for m in results['methods'].keys() if 'error' not in results['methods'][m]]
    num_methods = len(valid_methods)
    
    if num_methods == 0:
        print("No valid results for confusion matrices")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Confusion Matrices - {superclass} Specialist", fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, method_name in enumerate(valid_methods):
        if idx >= 4:
            break
        
        metrics = results['methods'][method_name]
        conf_matrix = np.array(metrics['confusion_matrix'])
        
        # Normalize confusion matrix for heatmap
        conf_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(conf_normalized, annot=conf_matrix, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[idx], cbar=True, square=True)
        axes[idx].set_title(f"{method_name}", fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    # Hide unused subplots
    for idx in range(num_methods, 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    fig.savefig(out_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: confusion_matrices.png")
    plt.close(fig)


def plot_metrics_radar(results: Dict, out_dir: Path):
    """Create radar chart comparing all metrics across methods."""
    from math import pi
    
    methods = []
    method_data = []
    
    for method_name, metrics in results['methods'].items():
        if 'error' not in metrics:
            methods.append(method_name)
            # Use only the 4 scalar metrics (not confusion matrix)
            data = [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score']
            ]
            method_data.append(data)
    
    if not methods:
        return
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    num_vars = len(categories)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    colors = ['steelblue', 'forestgreen', 'coral', 'mediumpurple']
    
    for idx, (method, data) in enumerate(zip(methods, method_data)):
        values = data + data[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    fig.suptitle(f"Multi-Metric Radar Comparison - {results['superclass'].upper()}", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(out_dir / 'metrics_radar.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: metrics_radar.png")
    plt.close(fig)


def create_summary_table(results: Dict, out_dir: Path):
    """Create and save a summary table of all metrics."""
    methods = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    
    for method_name, metrics in results['methods'].items():
        if 'error' not in metrics:
            methods.append(method_name)
            accuracies.append(f"{metrics['accuracy']:.4f}")
            precisions.append(f"{metrics['precision']:.4f}")
            recalls.append(f"{metrics['recall']:.4f}")
            f1_scores.append(f"{metrics['f1_score']:.4f}")
            roc = metrics.get('roc_auc')
            roc_aucs.append(f"{roc:.4f}" if roc is not None else "N/A")
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
    for i in range(len(methods)):
        table_data.append([
            methods[i],
            accuracies[i],
            precisions[i],
            recalls[i],
            f1_scores[i],
            roc_aucs[i]
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    fig.suptitle(f"Metrics Summary Table - {results['superclass'].upper()} Specialist", 
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(out_dir / 'metrics_summary_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: metrics_summary_table.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("--results", required=True, help="Results JSON file")
    parser.add_argument("--out", default="results/visualizations", help="Output directory for visualizations")
    args = parser.parse_args()

    results_file = Path(args.results)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_file}...")
    results = load_results(results_file)

    print("Generating visualizations...")
    plot_metrics_comparison(results, out_dir)
    plot_confusion_matrices(results, out_dir)
    plot_metrics_radar(results, out_dir)
    create_summary_table(results, out_dir)

    print(f"\n✓ All visualizations saved to {out_dir}")


if __name__ == '__main__':
    main()
