# THRP GitHub Upload Guide

## Overview
This guide explains which files to upload to GitHub and which to exclude, along with the implementation schema for other developers.

---

## ✅ FILES TO UPLOAD

### 1. **Source Code** (`scripts/`)
```
scripts/
├── predict.py                        # ⭐ Two-stage inference (generalist → specialist)
├── train_specialist_kaggle.py        # ⭐ Specialist model trainer (Fighter, Bomber, etc.)
├── train_generalist_kaggle.py        # Generalist model trainer (superclass detection)
├── aircraft_superclass_map.py        # ⭐ Aircraft-to-superclass mapping
├── dataset_organizer.py              # Dataset prep
├── preprocessing.py                  # Image preprocessing utils
├── train_specialist.py               # Legacy specialist trainer
├── train_generalist.py               # Legacy generalist trainer
├── infer_two_stage.py                # Legacy inference
├── kaggle_integration.py             # Kaggle dataset integration
├── convert_kaggle_csv_to_hierarchy.py # Dataset converter
└── run_pipeline.py                   # Pipeline runner
```

### 2. **Configuration Files**
```
models/
└── generalist_config_kaggle.yaml     # Generalist YOLO training config

yolo_specialist_datasets/
├── fighter/
│   └── fighter_config.yaml           # Fighter specialist config
├── bomber/
│   └── bomber_config.yaml
├── cargo/
│   └── cargo_config.yaml
├── helicopter/
│   └── helicopter_config.yaml
├── attack aircraft/
│   └── attack aircraft_config.yaml
└── tiltrotor/
    └── tiltrotor_config.yaml
```

### 3. **Documentation** (`.md` files)
```
README.md                       # ⭐ Quick start guide (create this)
IMPLEMENTATION_GUIDE.md         # ⭐ Setup & implementation schema (create this)
DELIVERABLES.md                 # Project deliverables
KAGGLE_INTEGRATION_GUIDE.md      # Kaggle dataset setup
KAGGLE_UPDATE_SUMMARY.md         # Kaggle integration notes
RESULTS_SUMMARY.txt             # Results & benchmarks
```

### 4. **Setup & Execution Scripts**
```
setup_env.sh                     # Environment setup script
run_training_loop.sh             # Training execution script
requirements.txt                 # Python dependencies
```

### 5. **Metadata**
```
.gitignore                       # Git ignore rules
aircraft_names.txt               # List of aircraft class names
```

---

## ❌ FILES TO EXCLUDE

### **Large Files (>.gitignore)**
- `.venv/` — Virtual environment (~500MB+)
- `models/*.pt` — Trained model weights (~6MB each, download separately)
- `yolov8n.pt` — Pre-trained YOLOv8 base model (~6.5MB)

### **Training Data**
- `dataset/` — Raw dataset copies
- `datasets/` — Dataset variants
- `yolo_generalist_dataset/images/` & `labels/` — Training data
- `yolo_specialist_datasets/*/images/` & `labels/` — Specialist training data
- `test/` — Test images

### **Generated Output & Runtime**
- `outputs/` — Inference outputs
- `runs/` — Training run artifacts
- `logs/` — Runtime logs
- `.venv/` — Virtual environment

### **Cache & IDE**
- `__pycache__/` — Python cache
- `.vscode/`, `.idea/` — IDE settings
- `.DS_Store`, `Thumbs.db` — OS junk

---

## 📋 UPLOAD CHECKLIST

```bash
# 1. Clean up local files
rm -rf __pycache__ outputs runs logs .venv

# 2. Initialize git (if not already done)
git init
git add .

# 3. Verify .gitignore is working
git status  # Should NOT show .venv, models/*.pt, or datasets

# 4. Commit & push
git commit -m "Initial THRP two-stage aircraft detection pipeline"
git push origin main
```

**Files to verify in git status:**
```
scripts/
├── *.py (all Python files)
yolo_specialist_datasets/
├── */
│   └── *_config.yaml
models/
├── generalist_config_kaggle.yaml
.gitignore
requirements.txt
setup_env.sh
run_training_loop.sh
*.md
aircraft_names.txt
```

---

## 🚀 IMPLEMENTATION SCHEMA FOR OTHER DEVELOPERS

### **Repository Structure (Expected After Clone)**
```
THRP/
├── scripts/
│   ├── predict.py
│   ├── train_specialist_kaggle.py
│   ├── train_generalist_kaggle.py
│   └── aircraft_superclass_map.py
├── models/
│   ├── generalist_config_kaggle.yaml
│   ├── generalist_model.pt              # ⬇ Download from releases
│   ├── fighter_specialist.pt            # ⬇ Download from releases
│   ├── bomber_specialist.pt             # ⬇ Download from releases
│   ├── cargo_specialist.pt              # ⬇ Download from releases
│   ├── helicopter_specialist.pt         # ⬇ Download from releases
│   ├── attack aircraft_specialist.pt    # ⬇ Download from releases
│   └── tiltrotor_specialist.pt          # ⬇ Download from releases
├── yolo_specialist_datasets/
│   ├── fighter/fighter_config.yaml
│   ├── bomber/bomber_config.yaml
│   ├── cargo/cargo_config.yaml
│   ├── helicopter/helicopter_config.yaml
│   ├── attack aircraft/attack aircraft_config.yaml
│   └── tiltrotor/tiltrotor_config.yaml
├── requirements.txt
├── setup_env.sh
├── run_training_loop.sh
└── README.md
```

### **Setup Instructions (for new developers)**

#### **Step 1: Clone & Environment Setup**
```bash
git clone https://github.com/<username>/THRP.git
cd THRP

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### **Step 2: Download Pre-trained Models**
Models are hosted in GitHub Releases (too large for repo). Download:
```bash
cd models/

# Download from releases page or run this script
wget https://github.com/<username>/THRP/releases/download/v1.0/models.tar.gz
tar -xzf models.tar.gz
rm models.tar.gz

cd ..
```

#### **Step 3: Prepare Dataset** (if retraining)
```bash
# Option A: Use existing YOLO dataset directory structure
# (Images & labels must be in yolo_specialist_datasets/*/images/ & labels/)

# Option B: Organize from raw images
python3 scripts/dataset_organizer.py --source <your_dataset> --dest ./dataset
```

### **Usage**

#### **Inference (Two-Stage Pipeline)**
```bash
# Single image inference
python3 scripts/predict.py --image path/to/image.jpg --visualize

# Show all detections (debug mode)
python3 scripts/predict.py --image path/to/image.jpg --all-detections
```

**Output:**
```
SuperClass: Fighter (87.3%)
Aircraft: F-16 (75.2%)
Combined: 65.6%
BBox: (100, 50, 400, 300)
```

#### **Training**

**Train Fighter Specialist (example):**
```bash
cd scripts
python3 train_specialist_kaggle.py \
  --superclass Fighter \
  --epochs 50 \
  --batch-size 32 \
  --imgsz 768 \
  --augment \
  --device mps
```

**Train Generalist:**
```bash
python3 train_generalist_kaggle.py \
  --epochs 50 \
  --batch-size 32 \
  --device mps
```

### **Configuration Schema**

#### **Aircraft-to-Superclass Mapping** (`scripts/aircraft_superclass_map.py`)
```python
AIRCRAFT_TO_SUPERCLASS: Dict[str, str] = {
    # Fighter
    "F16": "Fighter",
    "F18": "Fighter",
    "F35": "Fighter",
    # ... add more
    
    # Cargo
    "C130": "Cargo",
    "A400M": "Cargo",
    # ... add more
}
```

**To add a new aircraft:**
1. Add entry to `AIRCRAFT_TO_SUPERCLASS`
2. Ensure training data is in `yolo_specialist_datasets/<superclass>/images/` with labels
3. Run specialist trainer for that superclass

#### **Specialist YAML Config** (`yolo_specialist_datasets/fighter/fighter_config.yaml`)
```yaml
path: yolo_specialist_datasets/fighter
train: images/train
val: images/val
nc: 22  # Number of aircraft classes in this superclass
names:
  0: EF2000
  1: F14
  2: F15
  3: F16
  4: F18
  # ... rest of classes
```

**Key requirements:**
- `path`: Relative path to dataset root
- `train`: Path to training images (relative to `path`)
- `val`: Path to validation images (relative to `path`)
- `nc`: Number of aircraft classes
- `names`: Dictionary mapping class indices to aircraft names (order matters!)

#### **Generalist Config** (`models/generalist_config_kaggle.yaml`)
```yaml
path: yolo_generalist_dataset
train: images/train
val: images/val
nc: 6  # Number of superclasses
names:
  0: Fighter
  1: Bomber
  2: Cargo
  3: Helicopter
  4: Attack Aircraft
  5: Tiltrotor
```

### **File Format & Labeling Requirements**

#### **Training Data Structure**
```
yolo_specialist_datasets/fighter/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── val/
│       ├── img_101.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img_001.txt  # YOLO format: class_id x_center y_center w h (normalized)
    │   └── ...
    └── val/
        ├── img_101.txt
        └── ...
```

#### **Label File Format** (YOLO format)
```
# img_001.txt
15 0.523 0.412 0.256 0.478  # class_id center_x center_y width height (all 0-1 normalized)
```

### **Inference Pipeline Flow**

```
Input Image
    ↓
[Stage 1: Generalist Model]
    ↓ (detect superclass + bounding box)
[ROI Extraction & 512x512 Padding]
    ↓
[Stage 2: Specialist Model (Aircraft-Specific)]
    ↓ (detect specific aircraft type)
[Ranking by Combined Confidence]
    ↓
Output: SuperClass (conf%), Aircraft (conf%), Combined Score
```

---

## 📊 Performance Targets

From validation on Fighter specialist (after 50-epoch retrain with aug + imgsz=768):

| Aircraft | mAP@0.5 | Notes |
|----------|---------|-------|
| F16 | 0.480 | Main variant |
| F18 | 0.544 | Good performance |
| F35 | 0.573 | Best performer |
| J20 | 0.666 | Top performer |
| Mig29 | 0.195 | Low - needs more training |
| F2 | 0.093 | Low - rare class |

**Key insights:**
- Augmentation + larger imgsz improves robustness on blurred images
- Rare classes (F2, Tejas, FCK1) need oversampling or class weighting
- Generalist mAP should be >0.5 for reliable superclass detection

---

## 🔗 GitHub Releases

**Pre-trained Model Weights** should be uploaded to releases:
```
v1.0:
├── generalist_model.pt (6.0MB)
├── fighter_specialist.pt (6.0MB)
├── bomber_specialist.pt (6.0MB)
├── cargo_specialist.pt (6.0MB)
├── helicopter_specialist.pt (6.0MB)
├── attack_aircraft_specialist.pt (6.0MB)
└── tiltrotor_specialist.pt (6.0MB)
```

Download URL: `https://github.com/<username>/THRP/releases/download/v1.0/generalist_model.pt`

---

## ❓ FAQ for New Developers

**Q: How do I add a new superclass?**
A: 
1. Add aircraft to `AIRCRAFT_TO_SUPERCLASS` in `aircraft_superclass_map.py`
2. Create `yolo_specialist_datasets/<superclass>/` directory
3. Organize training data into `images/train` & `images/val` with labels in `labels/`
4. Create `<superclass>_config.yaml` following the fighter example
5. Train: `python3 train_specialist_kaggle.py --superclass <name>`

**Q: Model accuracy is low for certain aircraft. What should I do?**
A:
1. Check class distribution in training set (use validation mAP by class)
2. If rare class, oversample or use class weights
3. Increase epochs, add augmentation, increase imgsz
4. For blur robustness, add motion blur / Gaussian blur to augmentation

**Q: How do I test on blurred images?**
A:
```bash
python3 scripts/predict.py --image blurred_f16.jpg --visualize --all-detections
# Shows combined confidence scores for ranking
```

**Q: Can I use different device (GPU/CPU)?**
A: Yes, pass `--device cuda` or `--device cpu` to training scripts

---

## 📝 License
Add appropriate license (MIT, Apache 2.0, etc.)

## 👥 Contributing
Guidelines for external contributors (if accepting PRs)

---

*Last Updated: May 15, 2026*
