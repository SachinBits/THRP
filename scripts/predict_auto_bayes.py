#!/usr/bin/env python3
"""Shortcut wrapper for Gaussian Naive Bayes auto prediction."""
import sys

if len(sys.argv) < 2:
    print("Usage: predict_auto_bayes.py --image path/to.jpg [--generalist-model ...]")
    raise SystemExit(2)

args = ["--method", "bayes"] + sys.argv[1:]
from predict_auto_baseline import main as _main
sys.argv = [sys.argv[0]] + args
_main()
