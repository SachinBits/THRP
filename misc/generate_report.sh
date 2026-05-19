#!/bin/bash
# Quick script to generate complete THRP PDF report with all graphs and metrics

echo "════════════════════════════════════════════════════════════════"
echo "THRP FINAL REPORT GENERATION TOOLKIT"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Set working directory
THRP_DIR="/Users/sachin/Documents/Bits Pilani/DM_Project_v2/THRP"
cd "$THRP_DIR"

# Activate environment
echo "Step 1: Activating Python environment..."
source .venv/bin/activate

echo "Step 2: Generating PDF report..."
python3 scripts/generate_pdf_report.py \
    --metrics results/sample_metrics_fighter.json \
    --out outputs/THRP_FINAL_REPORT.pdf

echo ""
echo "✓ PDF Report generated: outputs/THRP_FINAL_REPORT.pdf"
echo ""
echo "Report includes:"
echo "  ✓ Title page with project overview"
echo "  ✓ Abstract with key findings"
echo "  ✓ Introduction & Literature Review"
echo "  ✓ Dataset & Methodology (with formulas)"
echo "  ✓ Evaluation metrics definitions (all 5 metrics + confusion matrix)"
echo "  ✓ Implementation details with code snippets"
echo "  ✓ Results: Accuracy, Precision, Recall, F1-Score bar charts"
echo "  ✓ Confusion matrices for all 4 classical methods"
echo "  ✓ Radar chart & summary table"
echo "  ✓ Discussion & conclusions"
echo "  ✓ References"
echo ""
echo "📄 Open the PDF: open outputs/THRP_FINAL_REPORT.pdf"
