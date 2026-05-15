"""Orchestrate full two-stage pipeline: convert CSV, train generalist, train specialists.

Usage:
  python3 scripts/run_pipeline.py --dataset ../datasets/data --gen-epochs 10 --spec-epochs 8
"""
import argparse
import subprocess
from pathlib import Path
import sys


def run(cmd):
    print('>',' '.join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='../datasets/data')
    p.add_argument('--gen-epochs', type=int, default=50)
    p.add_argument('--spec-epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='auto')
    p.add_argument('--skip-convert', action='store_true')
    p.add_argument('--skip-train', action='store_true')
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / 'scripts'

    if not args.skip_convert:
        run([sys.executable, str(scripts_dir / 'convert_kaggle_csv_to_hierarchy.py'), '--dataset-root', args.dataset])

    if args.skip_train:
        print('Skipping training as requested')
        return

    # Train generalist
    run([sys.executable, str(scripts_dir / 'train_generalist_kaggle.py'), '--dataset', args.dataset,
         '--epochs', str(args.gen_epochs), '--batch-size', str(args.batch_size), '--device', args.device])

    # Train all specialists
    run([sys.executable, str(scripts_dir / 'train_specialist_kaggle.py'), '--dataset', args.dataset,
         '--all', '--epochs', str(args.spec_epochs), '--batch-size', str(args.batch_size), '--device', args.device])


if __name__ == '__main__':
    main()
