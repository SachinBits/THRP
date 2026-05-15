# THRP Kaggle Dataset Integration Guide

## Overview

This guide explains how to use the **Kaggle Military Aircraft Detection Dataset** with the THRP (Two-Tier Hierarchical Recognition Pipeline) project.

## Dataset Statistics

```
Total Images: 638 real military aircraft photographs

SuperClasses:
  - Fighter:      203 images (7 aircraft types)
  - Bomber:       116 images (4 aircraft types)
  - Transport:    174 images (6 aircraft types)
  - Helicopter:   145 images (5 aircraft types)

Total Aircraft Types: 22
```

## Quick Start

### Step 1: Verify Dataset
```bash
cd "/Users/sachin/Documents/Bits Pilani/DM_Project_v2/THRP/scripts"
python3 kaggle_integration.py
```

### Step 2: Train Generalist Model
```bash
python3 train_generalist_kaggle.py \
    --epochs 50 \
    --batch-size 16 \
    --device cpu
```

### Step 3: Train Specialist Models
```bash
python3 train_specialist_kaggle.py \
    --all \
    --epochs 50 \
    --device cpu
```

### Step 4: Run Inference
```bash
python3 predict.py \
    --image /path/to/aircraft.jpg \
    --generalist ../models/generalist_model.pt \
    --specialists ../models/ \
    --visualize
```

## Training Parameters

### For CPU Training
```bash
python3 train_generalist_kaggle.py \
    --epochs 50 \
    --batch-size 16 \
    --device cpu
```

### For GPU Training (Faster)
```bash
python3 train_generalist_kaggle.py \
    --epochs 100 \
    --batch-size 32 \
    --device 0
```

## Dataset Structure

```
datasets/data/
├── Fighter/
│   ├── train/
│   │   ├── F-16/images/*.jpg, labels/*.txt
│   │   ├── F-18/images/*.jpg, labels/*.txt
│   │   └── ...
│   ├── val/
│   └── test/
├── Bomber/
├── Transport/
└── Helicopter/
```

## Training Time Estimates

- **CPU Training:** 2.5-5 hours total
- **GPU Training:** 30-60 minutes total

## New Scripts

### kaggle_integration.py
Verifies the Kaggle dataset structure and displays statistics.

**Usage:** `python3 kaggle_integration.py`

### train_generalist_kaggle.py
Trains the Stage 1 Generalist model for SuperClass detection.

**Usage:** `python3 train_generalist_kaggle.py --epochs 50 --device cpu`

### train_specialist_kaggle.py
Trains all 4 specialist models for aircraft identification.

**Usage:** `python3 train_specialist_kaggle.py --all --epochs 50 --device cpu`

## Troubleshooting

### Dataset not found
Run: `python3 kaggle_integration.py`

### Training too slow
Use GPU: `--device 0` (or any GPU index)

### Out of memory
Reduce batch size: `--batch-size 8`

### Installing Python dependencies (macOS notes)

If `pip install -r requirements.txt` fails with a `torch` resolution error on macOS, follow these steps:

1. Create and activate a virtual environment (recommended):
```bash
cd "/Users/sachin/Documents/Bits Pilani/DM_Project_v2/THRP"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install PyTorch via the official index (CPU build) then install remaining deps:
```bash
# CPU-only wheels (recommended for macOS without CUDA):
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Then install remaining packages from requirements
python -m pip install -r requirements.txt
```

3. If you have an Apple Silicon Mac and want metal-accelerated builds, follow the instructions at https://pytorch.org/get-started/locally/ and choose the macOS/Metal build appropriate for your system.

4. If pip still can't find compatible `torch` wheels, relax pins and install a compatible wheel manually. Example:
```bash
# Example: install latest torch wheel available on PyPI
python -m pip install "torch>=2.0.1,<3.0" "torchvision>=0.15.2,<0.18"
python -m pip install -r requirements.txt --no-deps
```

### Recommended: use Python 3.11 (important)

Some packages used in this project (PyTorch, torchvision, numpy, pandas, SciPy) provide prebuilt wheels for Python 3.11/macOS more reliably than for Python 3.13. If you see repeated build-from-source errors (compiling C extensions), create the virtualenv with Python 3.11.

Install Python 3.11 using `pyenv` (recommended) or `conda`:

Using `pyenv`:
```bash
# install pyenv (macOS Homebrew)
brew update && brew install pyenv
pyenv install 3.11.6
pyenv local 3.11.6
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r THRP/requirements.txt
```

Using `conda` (Miniforge/Miniconda):
```bash
conda create -n thrp python=3.11 -y
conda activate thrp
python -m pip install --upgrade pip
python -m pip install -r THRP/requirements.txt
```

If you'd like, I can add a short `setup.sh` script that detects/installs `pyenv` and creates the venv automatically — tell me if you want that.
I added `setup_env.sh` to the project root to automate creating a Python 3.11 venv and installing dependencies. Run:

```bash
cd "/Users/sachin/Documents/Bits Pilani/DM_Project_v2/THRP"
./setup_env.sh
```

The script will prefer an existing `python3.11`, fall back to `pyenv` (installing 3.11 if available), or instruct to use `conda`. It then creates `.venv`, activates it, and installs `requirements.txt` with a pre-install of PyTorch CPU wheels to reduce resolution issues.

## Resources

- Kaggle Dataset: https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset
- YOLOv8 Docs: https://docs.ultralytics.com/
- Original THRP README: See README.md

