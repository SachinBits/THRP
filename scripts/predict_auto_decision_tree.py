#!/usr/bin/env python3
"""Shortcut wrapper for decision tree auto prediction."""
import sys
from pathlib import Path

from predict_auto_baseline import run_auto


def main():
    if len(sys.argv) < 2:
        print("Usage: predict_auto_decision_tree.py --image path/to.jpg [--generalist-model ...]")
        raise SystemExit(2)
    # forward args but enforce method
    args = ["--method", "decision_tree"] + sys.argv[1:]
    from predict_auto_baseline import main as _main  # reuse CLI
    sys.argv = [sys.argv[0]] + args
    _main()


if __name__ == "__main__":
    main()
