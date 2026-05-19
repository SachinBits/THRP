#!/usr/bin/env python3
"""
Generate a single combined THRP PDF report that aggregates results
from all superclasses (fighter, cargo, helicopter, bomber).

Usage:
    python3 scripts/generate_combined_report.py --inputs results/sample_metrics_fighter.json results/sample_metrics_cargo.json results/sample_metrics_helicopter.json results/sample_metrics_bomber.json --out outputs/THRP_FINAL_REPORT_ALL.pdf

If `--inputs` is omitted the script will look for the four sample files in `results/`.
"""
import json
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def create_title_page(pdf, title):
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.text(0.5, 0.65, 'Two-Stage Hierarchical Recognition Pipeline (THRP)\nCombined Report', ha='center', fontsize=12)
    plt.text(0.5, 0.55, 'Includes Fighter, Cargo, Helicopter, Bomber analyses', ha='center', fontsize=10)
    plt.text(0.5, 0.45, 'Generated automatically', ha='center', fontsize=9)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_overall_comparison(pdf, all_results):
    # Create a bar chart comparing accuracy across methods and superclasses
    fig, ax = plt.subplots(figsize=(8.5, 6))
    superclasses = [r['superclass'].title() for r in all_results]
    methods = list(all_results[0]['methods'].keys())
    # Build data matrix: rows methods, cols superclasses
    data = np.zeros((len(methods), len(superclasses)))
    for j, r in enumerate(all_results):
        for i, m in enumerate(methods):
            data[i, j] = r['methods'][m]['accuracy']

    x = np.arange(len(superclasses))
    width = 0.18
    for i, m in enumerate(methods):
        ax.bar(x + i*width, data[i], width, label=m)

    ax.set_xticks(x + width*(len(methods)-1)/2)
    ax.set_xticklabels(superclasses)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title('Accuracy Comparison Across Superclasses (per method)')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_summary_table(pdf, all_results):
    # Build a summary table with methods as rows and superclasses as columns showing accuracy
    methods = list(all_results[0]['methods'].keys())
    superclasses = [r['superclass'].title() for r in all_results]
    table_data = []
    for m in methods:
        row = [m]
        for r in all_results:
            row.append(f"{r['methods'][m]['accuracy']*100:.1f}%")
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=['Method'] + superclasses, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax.set_title('Summary Table: Accuracy (%) by Method and Superclass')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_per_superclass_sections(pdf, all_results):
    # For each superclass, add a page with metrics and confusion matrices
    for r in all_results:
        sc = r['superclass'].title()
        # Metrics bar chart for this superclass
        methods = list(r['methods'].keys())
        accuracy = [r['methods'][m]['accuracy'] for m in methods]
        precision = [r['methods'][m]['precision'] for m in methods]
        recall = [r['methods'][m]['recall'] for m in methods]
        f1 = [r['methods'][m]['f1_score'] for m in methods]

        fig, axs = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle(f'{sc} — Metrics Comparison', fontsize=14, fontweight='bold')
        axs = axs.ravel()
        axs[0].bar(methods, accuracy, color='C0')
        axs[0].set_title('Accuracy')
        axs[0].set_ylim(0, 1)

        axs[1].bar(methods, precision, color='C1')
        axs[1].set_title('Precision')
        axs[1].set_ylim(0, 1)

        axs[2].bar(methods, recall, color='C2')
        axs[2].set_title('Recall')
        axs[2].set_ylim(0, 1)

        axs[3].bar(methods, f1, color='C3')
        axs[3].set_title('F1-Score')
        axs[3].set_ylim(0, 1)

        for ax in axs:
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_fontsize(8)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Confusion matrices page
        class_names = r['dataset_info']['class_names']
        valid_methods = list(r['methods'].keys())[:4]
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(f'{sc} — Confusion Matrices', fontsize=14, fontweight='bold')
        for idx, m in enumerate(valid_methods):
            ax = plt.subplot(2, 2, idx+1)
            cm = np.array(r['methods'][m]['confusion_matrix'])
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        ax=ax, cbar=True, square=True, cbar_kws={'shrink':0.8})
            ax.set_title(m, fontsize=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def add_overall_discussion(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle('Discussion & Conclusions', fontsize=14, fontweight='bold')
    plt.axis('off')
    text = (
        'This combined report compares classical baselines across four superclasses.\\n\\n'
        'Key observations:\\n'
        '- SVM performs consistently well across superclasses.\\n'
        '- Smaller class sets (bomber) yield higher accuracy.\\n'
        '- Confusion matrices indicate specific misclassification patterns per superclass.'
    )
    plt.text(0.05, 0.9, text, fontsize=10)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def generate_combined(inputs, out_path):
    all_results = [load_json(p) for p in inputs]
    with PdfPages(out_path) as pdf:
        create_title_page(pdf, 'THRP Combined Final Report')
        add_overall_comparison(pdf, all_results)
        add_summary_table(pdf, all_results)
        add_per_superclass_sections(pdf, all_results)
        add_overall_discussion(pdf)


def default_inputs(base_dir):
    return [
        base_dir / 'results' / 'sample_metrics_fighter.json',
        base_dir / 'results' / 'sample_metrics_cargo.json',
        base_dir / 'results' / 'sample_metrics_helicopter.json',
        base_dir / 'results' / 'sample_metrics_bomber.json',
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', help='Input JSON metrics files', default=None)
    parser.add_argument('--out', help='Output PDF path', default='outputs/THRP_FINAL_REPORT_ALL.pdf')
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    if args.inputs:
        inputs = [Path(p) for p in args.inputs]
    else:
        inputs = default_inputs(base)

    # Validate
    missing = [str(p) for p in inputs if not Path(p).exists()]
    if missing:
        print('Missing input files:')
        for m in missing:
            print(' -', m)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generate_combined(inputs, out_path)
    print('Combined PDF generated:', out_path)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
