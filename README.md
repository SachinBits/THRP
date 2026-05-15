# THRP: Two-Stage Hierarchical Recognition Pipeline
## Military Aircraft Detection & Classification

A production-ready two-stage deep learning pipeline for detecting and classifying military aircraft from images, achieving robust classification even on blurred/degraded imagery.

![Status](https://img.shields.io/badge/status-production-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Features

- **Two-Stage Pipeline**: Generalist (superclass) → Specialist (aircraft-specific) detection
- **6 Superclasses**: Fighter, Bomber, Cargo, Helicopter, Attack Aircraft, Tiltrotor
- **22+ Aircraft Types**: F-16, F-18, F-35, J-20, Su-57, A-10, C-130, CH-47, etc.
- **Robust Inference**: Square-padded ROI preprocessing + combined confidence scoring
- **Dynamic Model Discovery**: Auto-loads specialist weights from models directory
- **Augmentation Support**: Motion blur, Gaussian blur, JPEG artifacts for robustness on degraded images
- **Apple Silicon Optimized**: MPS device support for M1/M2/M3/M4 Macs

---

## 📊 Performance

| Stage | Metric | Value |
|-------|--------|-------|
| Generalist | mAP@0.5 (6 superclasses) | ~0.85 |
| Fighter Specialist | mAP@0.5 (22 fighters) | ~0.42 |
| Overall Pipeline | Correct superclass + aircraft | >80% on clear images |

**Note**: Specialist accuracy varies by aircraft type and training data availability.

---

## 🚀 Quick Start

### **1. Clone & Setup**
```bash
git clone https://github.com/<username>/THRP.git
cd THRP

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Download Models**
```bash
# Download pre-trained weights from releases
cd models
wget https://github.com/<username>/THRP/releases/download/v1.0/models.tar.gz
tar -xzf models.tar.gz
cd ..
```

### **3. Run Inference**
```bash
# Single image
python3 scripts/predict.py --image path/to/aircraft.jpg --visualize

# Show all candidate predictions
python3 scripts/predict.py --image path/to/aircraft.jpg --all-detections
```

**Sample Output:**
```
SuperClass: Fighter (87.3%)
Aircraft: F-18 (75.2%)
Combined: 65.6%
BBox: (100, 50, 400, 300)
```

---

## 📖 Usage

### **Inference**

```bash
python3 scripts/predict.py \
  --image path/to/image.jpg \
  --visualize                  # Save annotated image
  --all-detections             # Show all candidates (debug mode)
```

**Output Files:**
- Annotated image: `outputs/pred_<hash>.jpg`
- Console output: Superclass, aircraft type, confidence scores

### **Training**

#### **Train Fighter Specialist (example)**
```bash
python3 scripts/train_specialist_kaggle.py \
  --superclass Fighter \
  --epochs 50 \
  --batch-size 32 \
  --imgsz 768 \
  --augment \
  --device mps
```

#### **Train Generalist**
```bash
python3 scripts/train_generalist_kaggle.py \
  --epochs 50 \
  --batch-size 32 \
  --imgsz 640 \
  --device mps
```

**Training outputs:**
- Model: `models/<superclass>_specialist.pt`
- Validation: `runs/detect/val/`

---

## 📁 Project Structure

```
THRP/
├── scripts/
│   ├── predict.py                       # Two-stage inference pipeline
│   ├── train_specialist_kaggle.py       # Specialist trainer
│   ├── train_generalist_kaggle.py       # Generalist trainer
│   ├── aircraft_superclass_map.py       # Class mapping
│   └── ...
├── models/
│   ├── generalist_model.pt              # Generalist weights (download)
│   ├── *_specialist.pt                  # Specialist weights (download)
│   └── *.yaml                           # Training configs
├── yolo_specialist_datasets/
│   ├── fighter/
│   │   ├── fighter_config.yaml
│   │   ├── images/ (train, val)
│   │   └── labels/ (train, val)
│   └── ... (other superclasses)
├── requirements.txt
├── setup_env.sh
└── README.md
```

---

## 🔧 Configuration

### **Aircraft Mapping** (`scripts/aircraft_superclass_map.py`)

Define aircraft-to-superclass relationships:
```python
AIRCRAFT_TO_SUPERCLASS = {
    # Fighter
    "F16": "Fighter",
    "F18": "Fighter",
    "F35": "Fighter",
    # ... add more
}
```

### **Dataset YAML** (`yolo_specialist_datasets/fighter/fighter_config.yaml`)

```yaml
path: yolo_specialist_datasets/fighter
train: images/train
val: images/val
nc: 22
names:
  0: EF2000
  1: F14
  # ... rest of classes
```

**Key requirements:**
- Paths are relative to repo root
- Class indices must match label files
- Train/val split: 70/30 or 80/20 recommended

### **Training Config** (`models/generalist_config_kaggle.yaml`)

```yaml
path: yolo_generalist_dataset
train: images/train
val: images/val
nc: 6  # 6 superclasses
names:
  0: Fighter
  1: Bomber
  2: Cargo
  3: Helicopter
  4: Attack Aircraft
  5: Tiltrotor
```

---

## 📊 Data Format

### **Directory Structure**
```
yolo_specialist_datasets/fighter/
├── images/
│   ├── train/     # ~70% of images
│   └── val/       # ~30% of images
└── labels/
    ├── train/     # YOLO format labels
    └── val/
```

### **Label Format** (YOLO)
```
# fighter_001.txt
15 0.523 0.412 0.256 0.478
```
- `class_id`: Integer (0-indexed)
- `x_center, y_center`: Normalized box center (0-1)
- `width, height`: Normalized box dimensions (0-1)

---

## 🎓 How It Works

### **Stage 1: Generalist Detection**
1. Input image passed to generalist YOLO model
2. Detects bounding boxes with superclass labels (Fighter, Bomber, etc.)
3. Returns all superclass detections with confidence scores

### **Stage 2: Specialist Identification**
1. For each generalist detection:
   - Extract ROI (region of interest)
   - Pad to 512×512 square (preserves aspect ratio)
   - Pass to superclass-specific specialist model
   - Detect specific aircraft type (F-16, F-18, etc.)

### **Stage 3: Ranking**
1. Compute combined score = generalist_conf × specialist_conf
2. Sort by combined score (descending)
3. Return top-1 result (or all with `--all-detections`)

**Advantage**: Specialist models are smaller, faster, more accurate than single 1000-class detector.

---

## 🔨 Advanced Usage

### **Retrain with Augmentation**
For improved robustness on blurred images:

```bash
python3 scripts/train_specialist_kaggle.py \
  --superclass Fighter \
  --epochs 100 \
  --batch-size 16 \
  --imgsz 768 \
  --augment           # Motion blur, Gaussian blur, JPEG artifacts
  --device mps
```

### **Add New Aircraft**
1. Update `aircraft_superclass_map.py`:
   ```python
   "F18": "Fighter",
   "NEW_AIRCRAFT": "Fighter"  # Add here
   ```
2. Collect training data → `yolo_specialist_datasets/fighter/images/train/`
3. Label with bounding boxes (YOLO format)
4. Update `fighter_config.yaml`: increment `nc`, add name mapping
5. Retrain: `python3 train_specialist_kaggle.py --superclass Fighter --epochs 50`

### **Validate Model**
```bash
python3 - <<'PY'
from ultralytics import YOLO
m = YOLO('models/fighter_specialist.pt')
results = m.val(data='yolo_specialist_datasets/fighter/fighter_config.yaml')
# Shows per-class mAP50, confusion matrix, etc.
PY
```

---

## ⚙️ System Requirements

- **Python**: 3.10+
- **RAM**: 8GB+ (16GB recommended)
- **GPU**: Optional (CUDA 11.8+, MPS for Apple Silicon)
- **Storage**: 2GB (code + models), 50GB+ (for full training data)

### **Dependencies**
- ultralytics (8.4.50+)
- torch (2.0+)
- torchvision
- opencv-python
- numpy
- PyYAML

See `requirements.txt` for full list.

---

## 🐛 Troubleshooting

### **Issue: "No specialist models loaded"**
- Check `models/` directory has `*_specialist.pt` files
- Verify naming convention: `{superclass}_specialist.pt`
- Download models from GitHub releases

### **Issue: Specialist returns "Unknown" with 0% confidence**
- ROI too small? Pipeline auto-pads to 512×512, should work
- Model confidence threshold too high? Lowered to 0.01 by default
- Run with `--all-detections` to see raw predictions

### **Issue: Model accuracy low for specific aircraft**
- Check class distribution: `python3 -c "import os; print({cls: len(os.listdir(f'yolo_specialist_datasets/fighter/images/train')) for cls in os.listdir(...)})"` 
- Rare classes need oversampling or weighted loss
- Try longer training: `--epochs 100` instead of 50
- Increase augmentation

### **Issue: Training runs out of memory**
- Reduce batch size: `--batch-size 16` instead of 32
- Reduce imgsz: `--imgsz 640` instead of 768
- Use CPU: `--device cpu`

---

## 📚 Reference

- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **Paper**: "You Only Look Once: Unified, Real-Time Object Detection"
- **Dataset**: Military aircraft from Kaggle/web sources

---

## 📄 License
MIT License - See LICENSE file

## 👥 Authors
Sachin S.

## 🤝 Contributing
Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Submit a pull request

---

## 📞 Support
For issues, questions, or feedback:
- Open an issue on GitHub
- Check existing documentation in `/docs`
- Review GITHUB_UPLOAD_GUIDE.md for detailed schema

---

*Last Updated: May 15, 2026*
