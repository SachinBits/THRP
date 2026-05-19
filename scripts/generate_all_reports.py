#!/usr/bin/env python3
"""
Generate THRP Final Report PDFs for all superclasses (Fighter, Cargo, Helicopter, Bomber).
This creates separate professional 10-page PDFs with proper confusion matrix labels.

Usage:
    python3 scripts/generate_all_reports.py

Output:
    - outputs/THRP_FINAL_REPORT_Fighter.pdf
    - outputs/THRP_FINAL_REPORT_Cargo.pdf
    - outputs/THRP_FINAL_REPORT_Helicopter.pdf
    - outputs/THRP_FINAL_REPORT_Bomber.pdf
"""

import sys
from pathlib import Path
from generate_pdf_report import THRPReportGenerator

def main():
    """Generate reports for all 4 superclasses."""
    
    superclasses = ['fighter', 'cargo', 'helicopter', 'bomber']
    results_dir = Path(__file__).parent.parent / 'results'
    outputs_dir = Path(__file__).parent.parent / 'outputs'
    
    print("=" * 70)
    print("THRP FINAL REPORT GENERATION - ALL SUPERCLASSES")
    print("=" * 70)
    print()
    
    for superclass in superclasses:
        metrics_file = results_dir / f'sample_metrics_{superclass}.json'
        output_file = outputs_dir / f'THRP_FINAL_REPORT_{superclass.title()}.pdf'
        
        if not metrics_file.exists():
            print(f"⚠️  Missing metrics file: {metrics_file.name}")
            print(f"    Skipping {superclass.title()}...")
            print()
            continue
        
        print(f"📄 Generating PDF for {superclass.title()} aircraft...")
        print(f"   Input:  {metrics_file.name}")
        print(f"   Output: {output_file.name}")
        
        try:
            generator = THRPReportGenerator(metrics_file, output_file)
            generator.generate_report()
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"   ✅ Success! ({file_size_mb:.1f} MB)")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        print()
    
    print("=" * 70)
    print("✅ ALL REPORTS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print("Generated PDFs:")
    for superclass in superclasses:
        output_file = outputs_dir / f'THRP_FINAL_REPORT_{superclass.title()}.pdf'
        if output_file.exists():
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ {output_file.name} ({file_size_mb:.1f} MB)")
    
    print()
    print("To view reports:")
    print("  open outputs/THRP_FINAL_REPORT_Fighter.pdf")
    print("  open outputs/THRP_FINAL_REPORT_Cargo.pdf")
    print("  open outputs/THRP_FINAL_REPORT_Helicopter.pdf")
    print("  open outputs/THRP_FINAL_REPORT_Bomber.pdf")

if __name__ == '__main__':
    main()
