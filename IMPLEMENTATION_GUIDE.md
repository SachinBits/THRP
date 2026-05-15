# THRP Implementation Guide for New Developers

This guide explains how to set up, understand, and extend the THRP pipeline for new aircraft classes, datasets, or use cases.

---

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Code Architecture](#code-architecture)
3. [Implementation Workflow](#implementation-workflow)
4. [Adding New Aircraft](#adding-new-aircraft)
5. [Training Specialist Models](#training-specialist-models)
6. [Debugging & Validation](#debugging--validation)

---

## Initial Setup

### **Step 1: Environment Configuration**

```bash
# Clone repository
git clone https://github.com/<username>/THRP.git
cd THRP

# Create Python virtual environment (Python 3.10+)
python3.10 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .\.venv\Scripts\activate  (Windows)

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python3 -c "from ultralytics import YOLO; print('✓ YOLO installed')"
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python3 -c "import cv2; print('✓ OpenCV installed')"
```

### **Step 2: Download Pre-trained Models**

Models are stored in GitHub Releases (too large for repo):

```bash
# Option A: Manual download
# Go to: https://github.com/<username>/THRP/releases/download/v1.0/
# Download each .pt file to models/ directory

# Option B: Automated download (create this script)
cd models
wget https://github.com/<username>/THRP/releases/download/v1.0/generalist_model.pt
wget https://github.com/<username>/THRP/releases/download/v1.0/fighter_specialist.pt
wget https://github.com/<username>/THRP/releases/download/v1.0/bomber_specialist.pt
wget https://github.com/<username>/THRP/releases/download/v1.0/cargo_specialist.pt
wget https://github.com/<username>/THRP/releases/download/v1.0/helicopter_specialist.pt
wget https://github.com/<username>/THRP/releases/download/v1.0/attack_aircraft_specialist.pt
wget https://github.com/<username>/THRP/releases/download/v1.0/tiltrotor_specialist.pt
cd ..

# Verify models exist
ls -lh models/*.pt  # Should show 7 files
```

### **Step 3: Test Installation**

```bash
# Quick inference test
python3 scripts/predict.py --image <path_to_test_image> --visualize

# Expected output:
# INFO:__main__:Initializing THRP Pipeline...
# INFO:__main__:✓ Generalist model loaded
# INFO:__main__:✓ 7 specialist models loaded
# Processing: <image>
# SuperClass: Fighter (XX.XX%)
# Aircraft: F-XX (XX.XX%)
# Combined: XX.XX%
```

---

## Code Architecture

### **Module Breakdown**

#### **1. `aircraft_superclass_map.py`** — Core Mapping & Utilities
**Purpose**: Define aircraft-to-superclass relationships and helper functions

**Key Components**:
```python
AIRCRAFT_TO_SUPERCLASS  # Dict[str, str] - aircraft name → superclass
SUPERCLASS_ORDER        # List[str] - ordered superclass names
build_superclass_lists() # → Dict[str, List[str]] - grouped by superclass
resolve_device()        # GPU/MPS/CPU selection
```

**Usage Example**:
```python
from aircraft_superclass_map import AIRCRAFT_TO_SUPERCLASS, build_superclass_lists

# Get all superclasses
all_superclasses = build_superclass_lists()  # {"Fighter": [F16, F18, ...], ...}

# Check an aircraft's superclass
print(AIRCRAFT_TO_SUPERCLASS["F16"])  # "Fighter"
```

**Extension**: Add new aircraft by adding to `AIRCRAFT_TO_SUPERCLASS`:
```python
AIRCRAFT_TO_SUPERCLASS["YOUR_NEW_AIRCRAFT"] = "Fighter"
```

---

#### **2. `predict.py`** — Inference Pipeline
**Purpose**: Two-stage detection and aircraft identification

**Key Classes**:
```python
class THRPInferencePipeline:
    def __init__(generalist_model_path, specialist_models_dir)
    def detect_superclass(image) → Dict[superclass, confidence, bbox]
    def identify_aircraft(image, bbox, superclass) → Dict[aircraft, confidence]
    def process_image(image_path) → Dict[results list, ranked by confidence]
```

**Workflow**:
```
Input Image
    ↓
generalist.predict(image) → [Bomber(80%), Fighter(10%), ...]
    ↓ (filter by conf=0.1)
extract_roi(image, bbox)  → crop of aircraft
    ↓
pad_to_square(roi, 512x512)  → normalized size
    ↓
specialist[superclass].predict(roi_padded) → [Tu22M(15%), ...]
    ↓
rank by (generalist_conf * specialist_conf)
    ↓
Output: Top-1 or all with --all-detections
```

**Code Flow** (simplified):
```python
# Stage 1: Generalist
results = generalist_model.predict(image, conf=0.1)
for box in results[0].boxes:
    superclass = SUPERCLASSES[int(box.cls[0])]
    bbox = box.xyxy[0]  # Top-left, bottom-right coords
    
# Stage 2: Specialist
roi = image[y1:y2, x1:x2]
roi_padded = pad_to_square(roi, 512)  # Preserve aspect, pad sides
specialist = specialist_models[superclass]
spec_results = specialist.predict(roi_padded, conf=0.01)

# Stage 3: Rank
combined_conf = gen_conf * spec_conf
```

---

#### **3. `train_specialist_kaggle.py`** — Specialist Model Trainer
**Purpose**: Train superclass-specific aircraft classifiers

**Key Functions**:
```python
def create_specialist_config(superclass, dataset_root)
    # Create YAML config for YOLO training
    
def _flatten_specialist_dataset(superclass, dataset_root)
    # Organize raw images/labels into train/val split
    
def train_specialist(superclass, epochs, batch_size, imgsz, augment, device)
    # Initialize YOLO, train, save model
```

**Example Training Flow**:
```bash
# User input
python3 train_specialist_kaggle.py \
  --superclass Fighter \
  --epochs 50 \
  --batch-size 32 \
  --imgsz 768 \
  --augment

# What happens internally:
# 1. create_specialist_config("Fighter", "./yolo_specialist_datasets")
#    → Generates fighter_config.yaml with correct class indices
# 2. _flatten_specialist_dataset("Fighter", ...)
#    → Organizes raw data into images/train, images/val, labels/train, labels/val
# 3. model = YOLO("yolov8n.pt")
# 4. model.train(data=config_path, epochs=50, imgsz=768, augment=True, ...)
# 5. model.save("models/fighter_specialist.pt")
```

**Key Implementation Details**:
- Config YAML paths are **relative** to repo root (not absolute)
- Class indices in YAML must match label file indices
- Augmentation includes motion blur, Gaussian blur, JPEG artifacts for robustness
- Model saved to `models/<superclass>_specialist.pt`

---

#### **4. `train_generalist_kaggle.py`** — Generalist Model Trainer
**Purpose**: Train 6-superclass detector

**Process**:
```
Input: Mixed dataset with 6 superclass labels
    ↓
Train YOLO on generalist_config.yaml
    ↓
Output: generalist_model.pt (6 classes: Fighter, Bomber, Cargo, ...)
```

**Config** (`models/generalist_config_kaggle.yaml`):
```yaml
path: yolo_generalist_dataset
train: images/train
val: images/val
nc: 6
names:
  0: Fighter
  1: Bomber
  2: Cargo
  3: Helicopter
  4: Attack Aircraft
  5: Tiltrotor
```

---

### **Data Flow Diagram**

```
┌─────────────────────────────────────────────────────────┐
│                   TRAINING PHASE                         │
└─────────────────────────────────────────────────────────┘

Raw Images + Annotations (arbitrary format)
    ↓
dataset_organizer.py  ← Converts to YOLO format
    ↓
yolo_specialist_datasets/fighter/
├── images/train/
├── images/val/
├── labels/train/   ← YOLO format: class_id x_center y_center w h
└── labels/val/

    ↓
train_specialist_kaggle.py --superclass Fighter
    ↓
models/fighter_specialist.pt  ← Trained model

┌─────────────────────────────────────────────────────────┐
│                  INFERENCE PHASE                         │
└─────────────────────────────────────────────────────────┘

Input Image
    ↓
generalist_model.pt  ← Detects superclass
    ↓
Extract ROI + Pad to 512×512
    ↓
fighter_specialist.pt  ← Detects aircraft subtype
    ↓
Combined Score Ranking
    ↓
Output: Aircraft Type + Confidence
```

---

## Implementation Workflow

### **Scenario 1: Using Pre-Trained Models (No Retraining)**

```bash
# 1. Setup (one-time)
git clone ...
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download models
cd models && wget <models_url> && cd ..

# 3. Run inference
python3 scripts/predict.py --image my_aircraft.jpg --visualize
```

**Output**: Annotated image in `outputs/`, console shows predictions

---

### **Scenario 2: Retrain Fighter Specialist (Improve Accuracy)**

```bash
# 1. Prepare training data
# Ensure data is in: yolo_specialist_datasets/fighter/images/train, etc.
# With labels in: yolo_specialist_datasets/fighter/labels/train, etc.

# 2. Retrain (50 epochs, with augmentation)
python3 scripts/train_specialist_kaggle.py \
  --superclass Fighter \
  --epochs 50 \
  --batch-size 32 \
  --imgsz 768 \
  --augment \
  --device mps  # or 'cuda' for NVIDIA, '0' for GPU 0

# 3. Validation
python3 - <<'PY'
from ultralytics import YOLO
m = YOLO('models/fighter_specialist.pt')
results = m.val(data='yolo_specialist_datasets/fighter/fighter_config.yaml')
print(f"mAP50: {results.box.map50:.4f}")
PY

# 4. Test inference (should see improvements)
python3 scripts/predict.py --image <fighter_image> --all-detections
```

---

### **Scenario 3: Add New Aircraft Type**

**Goal**: Support F-100 (currently not in training data)

```bash
# Step 1: Update mapping
# File: scripts/aircraft_superclass_map.py
AIRCRAFT_TO_SUPERCLASS = {
    ...
    "F100": "Fighter",  # ADD THIS
}

# Step 2: Collect training images
# Create folder: datasets/raw/f100/
# Place 100+ images in this folder

# Step 3: Label images (YOLO format)
# Use tool like LabelImg, Roboflow, or custom script
# Output: images/ and labels/ directories

# Step 4: Organize into dataset structure
python3 scripts/dataset_organizer.py \
  --source datasets/raw/f100 \
  --dest yolo_specialist_datasets/fighter

# Step 5: Update fighter_config.yaml
# File: yolo_specialist_datasets/fighter/fighter_config.yaml
names:
  0: EF2000
  ...
  22: F100  # ADD THIS (increment nc: 22 → 23)
nc: 23

# Step 6: Retrain Fighter specialist
python3 scripts/train_specialist_kaggle.py \
  --superclass Fighter \
  --epochs 50 \
  --batch-size 32 \
  --augment

# Step 7: Test
python3 scripts/predict.py --image <f100_image> --all-detections
# Should now detect F-100 with reasonable confidence
```

---

### **Scenario 4: Add Completely New Superclass**

**Goal**: Support "Drone" superclass with 10 drone types

```bash
# Step 1: Create mapping
# scripts/aircraft_superclass_map.py
AIRCRAFT_TO_SUPERCLASS = {
    ...
    "DJI_Phantom": "Drone",
    "Quadcopter_A": "Drone",
    ...
}

# Step 2: Create dataset directories
mkdir -p yolo_specialist_datasets/drone/{images/{train,val},labels/{train,val}}

# Step 3: Populate with training data
# Organize ~1000 drone images (70% train, 30% val)

# Step 4: Create Drone config
# File: yolo_specialist_datasets/drone/drone_config.yaml
cat > yolo_specialist_datasets/drone/drone_config.yaml <<'YAML'
path: yolo_specialist_datasets/drone
train: images/train
val: images/val
nc: 10
names:
  0: DJI_Phantom
  1: Quadcopter_A
  ... (all 10 classes)
YAML

# Step 5: Update generalist for 7 superclasses
# File: models/generalist_config_kaggle.yaml
# nc: 6 → 7
# Add "Drone" to names

# Step 6: Retrain generalist
python3 scripts/train_generalist_kaggle.py \
  --epochs 50 \
  --batch-size 32 \
  --augment

# Step 7: Train drone specialist
python3 scripts/train_specialist_kaggle.py \
  --superclass Drone \
  --epochs 50 \
  --batch-size 32 \
  --augment

# Step 8: Update predict.py (auto-discovers specialist)
# No changes needed! Pipeline auto-loads Drone specialist.pt

# Step 9: Test full pipeline
python3 scripts/predict.py --image <drone_image> --visualize
```

---

## Training Specialist Models

### **Understanding the Training Process**

```python
# Simplified train_specialist_kaggle.py logic
from ultralytics import YOLO

def train_specialist(superclass, epochs, batch_size, imgsz, augment, device):
    # 1. Create config YAML
    config_path = create_specialist_config(superclass, "./yolo_specialist_datasets")
    
    # 2. Flatten dataset if needed
    _flatten_specialist_dataset(superclass, "./yolo_specialist_datasets")
    
    # 3. Initialize YOLO with pre-trained weights
    model = YOLO("yolov8n.pt")  # Nano model (3M parameters)
    
    # 4. Train
    results = model.train(
        data=config_path,           # e.g., "yolo_specialist_datasets/fighter/fighter_config.yaml"
        epochs=epochs,              # e.g., 50
        imgsz=imgsz,               # e.g., 768
        batch=batch_size,          # e.g., 32
        augment=augment,           # Motion blur, Gaussian blur, etc.
        device=device,             # "mps", "cuda", "cpu", or "0"
        patience=10,               # Early stopping
        save=True,                 # Save model
        project="runs/detect/models",  # Output directory
        name=f"specialist_{superclass.lower()}"
    )
    
    # 5. Save to models/
    model.save(f"models/{superclass.lower()}_specialist.pt")
    return model
```

### **Key Training Parameters**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `imgsz` | 768 | Larger = better quality but slower |
| `batch-size` | 32 | Reduce if OOM |
| `epochs` | 50-100 | More epochs = better but slower |
| `augment` | True | Motion blur, JPEG noise, etc. for robustness |
| `device` | "mps" / "cuda" | GPU acceleration |

### **Augmentation Strategy**

For robustness on blurred/degraded images:

```yaml
# In YOLO augmentation config
augmentations:
  - motion_blur: 0.5
  - gaussian_blur: 0.3
  - jpeg_artifacts: 0.2
  - hsv_augment: 0.015
  - rotate: 10
  - shear: 5
```

**Effect**: Model learns to recognize aircraft despite:
- Motion blur (fast-moving targets)
- Compression artifacts (low-quality video)
- Lighting variations
- Slight rotations

---

## Debugging & Validation

### **Validation Metrics**

```bash
# Validate fighter specialist
python3 - <<'PY'
from ultralytics import YOLO
m = YOLO('models/fighter_specialist.pt')
results = m.val(data='yolo_specialist_datasets/fighter/fighter_config.yaml')

# Print per-class metrics
for i, class_name in results.names.items():
    map50 = results.box.map50[i]
    print(f"{class_name}: {map50:.4f}")
PY
```

**Interpretation**:
- `mAP50 > 0.5`: Good performance
- `mAP50 0.3-0.5`: Acceptable, consider retraining
- `mAP50 < 0.3`: Poor, check class distribution or increase training time

### **Debug Inference**

```bash
# Show all detections with confidence scores
python3 scripts/predict.py --image <img> --all-detections

# Expected output:
# SuperClass: Fighter (70%)
# Aircraft: F-16 (85%)
# Combined: 59%
# BBox: (100, 50, 400, 300)
#
# SuperClass: Bomber (20%)  ← Alternative (lower confidence)
# Aircraft: B-52 (5%)
# Combined: 1%
# BBox: (50, 25, 380, 280)
```

### **Troubleshooting Low Accuracy**

**Problem**: Fighter specialist returns "Unknown" with 0% confidence

**Diagnosis**:
```python
# Check 1: Is model loaded?
from ultralytics import YOLO
m = YOLO('models/fighter_specialist.pt')
print(m.names)  # Should print 22 fighter classes

# Check 2: Can model detect on val set?
results = m.predict('yolo_specialist_datasets/fighter/images/val/sample.jpg', conf=0.01)
print(len(results[0].boxes))  # Should be > 0

# Check 3: What confidence score?
for box in results[0].boxes:
    print(f"conf={float(box.conf[0]):.4f}, cls={int(box.cls[0])}")
```

**Solutions**:
1. Increase training epochs: `--epochs 100`
2. Lower confidence threshold in `predict.py`: `conf=0.01` → `conf=0.001`
3. Increase ROI size: `512x512` → `768x768`
4. Add more training data, especially for rare classes

---

## File Format Reference

### **YOLO Label Format**

Each image has a corresponding `.txt` label file with one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

**Example** (`fighter_001.txt`):
```
15 0.523 0.412 0.256 0.478
```

**Explanation**:
- `15`: Class ID (0-indexed, e.g., class 15 = KAAN)
- `0.523`: x-center normalized (0 = left, 1 = right)
- `0.412`: y-center normalized (0 = top, 1 = bottom)
- `0.256`: width normalized
- `0.478`: height normalized

### **Config YAML Format**

```yaml
path: /absolute/or/relative/path/to/dataset
train: images/train                # Path to training images
val: images/val                    # Path to validation images
test: images/test                  # Optional test set
nc: 22                            # Number of classes
names:
  0: EF2000
  1: F14
  ...
  21: Tejas
```

**Key Points**:
- `path`, `train`, `val` are relative to repo root
- Must have `images/` and `labels/` subdirs with matching names
- Class count (`nc`) must match number of entries in `names`

---

## Quick Reference

### **Training Commands**

```bash
# Fighter specialist (standard)
python3 scripts/train_specialist_kaggle.py --superclass Fighter --epochs 50 --batch-size 32 --imgsz 768 --augment --device mps

# Generalist (6 superclasses)
python3 scripts/train_generalist_kaggle.py --epochs 50 --batch-size 32 --imgsz 640 --device mps

# High-quality training (slower, better results)
python3 scripts/train_specialist_kaggle.py --superclass Fighter --epochs 200 --batch-size 16 --imgsz 1024 --augment --device mps
```

### **Inference Commands**

```bash
# Standard prediction
python3 scripts/predict.py --image path/to/image.jpg --visualize

# Debug mode (all candidates)
python3 scripts/predict.py --image path/to/image.jpg --all-detections

# No visualization (just console output)
python3 scripts/predict.py --image path/to/image.jpg
```

### **Validation Commands**

```bash
# Fighter specialist
python3 - <<'PY'
from ultralytics import YOLO
YOLO('models/fighter_specialist.pt').val(data='yolo_specialist_datasets/fighter/fighter_config.yaml')
PY

# Generalist
python3 - <<'PY'
from ultralytics import YOLO
YOLO('models/generalist_model.pt').val(data='yolo_generalist_dataset/generalist_config.yaml')
PY
```

---

## Next Steps

1. **Clone & Setup**: Follow Initial Setup section
2. **Run Inference**: Test with provided models
3. **Understand Architecture**: Review Code Architecture section
4. **Extend System**: Use Scenario 1-4 to add new aircraft/superclasses
5. **Optimize**: Retrain with augmentation for your use case

---

*Last Updated: May 15, 2026*
