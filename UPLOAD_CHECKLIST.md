# GitHub Upload Summary & Checklist

## 📦 Quick Summary

**To Upload** (source code + config):
- ✅ `scripts/` — All Python files (prediction, training, utilities)
- ✅ `models/` — Only `*.yaml` config files (NOT `*.pt` weights)
- ✅ `yolo_specialist_datasets/` — Only `*_config.yaml` files (NOT `images/` or `labels/`)
- ✅ All `*.md` documentation files
- ✅ `requirements.txt`, `setup_env.sh`, `run_training_loop.sh`
- ✅ `.gitignore`, `aircraft_names.txt`

**NOT to Upload** (excluded by `.gitignore`):
- ❌ `.venv/` — Virtual environment
- ❌ `models/*.pt` — Large model weights (>6MB each)
- ❌ `dataset/`, `datasets/` — Raw training data
- ❌ `yolo_*.dataset/images/`, `yolo_*.dataset/labels/` — Training images/labels
- ❌ `test/` — Test images
- ❌ `outputs/`, `runs/`, `logs/` — Generated files
- ❌ `__pycache__/`, `*.pyc` — Python cache
- ❌ `.DS_Store`, `.vscode/`, `.idea/` — OS/IDE files

---

## 📋 Pre-Upload Checklist

### **Local Cleanup**
```bash
cd /Users/sachin/Documents/Bits\ Pilani/DM_Project_v2/THRP

# Remove cache
rm -rf __pycache__ .venv

# Verify .gitignore exists and is correct
cat .gitignore

# Check what will be uploaded
git add .
git status
```

### **Files to Verify in `git status`**

Should see (green):
```
scripts/*.py
models/*.yaml
yolo_specialist_datasets/*/*.yaml
*.md
requirements.txt
setup_env.sh
run_training_loop.sh
.gitignore
aircraft_names.txt
```

Should NOT see:
```
models/*.pt
.venv/
__pycache__/
outputs/
runs/
dataset/
yolo_*/images/
yolo_*/labels/
test/
.DS_Store
```

---

## 🚀 GitHub Upload Steps

### **Step 1: Initialize Git (if not already done)**
```bash
cd /Users/sachin/Documents/Bits\ Pilani/DM_Project_v2/THRP

# Check if git is initialized
git status 2>/dev/null && echo "✓ Git initialized" || echo "✗ Not initialized"

# If not, initialize
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### **Step 2: Add All Files (respecting .gitignore)**
```bash
git add .
git status  # Verify correct files show as staged
```

### **Step 3: Create Initial Commit**
```bash
git commit -m "Initial THRP two-stage aircraft detection pipeline

- Two-stage inference: generalist (superclass) → specialist (aircraft-specific)
- Supports 6 superclasses (Fighter, Bomber, Cargo, Helicopter, Attack Aircraft, Tiltrotor)
- 22+ aircraft types with dynamic model discovery
- Robust preprocessing: square-padded ROI + combined confidence scoring
- Augmentation support for degraded images
- Ready for retraining with custom datasets"
```

### **Step 4: Create GitHub Repository**
1. Go to https://github.com/new
2. Create repo: `THRP` (or your chosen name)
3. Choose: **Public** (if allowing others to use) or **Private**
4. Do NOT initialize with README (you already have one)

### **Step 5: Push to GitHub**
```bash
# Add remote (replace USERNAME)
git remote add origin https://github.com/USERNAME/THRP.git

# Rename branch if needed (GitHub uses 'main' by default)
git branch -M main

# Push
git push -u origin main
```

### **Step 6: Create GitHub Release (for Models)**
```bash
# Create release with model weights
# Go to: https://github.com/USERNAME/THRP/releases/new

# Tag: v1.0
# Title: "Pre-trained Models v1.0"
# Description: "Pre-trained generalist and 6 specialist models"

# Upload files:
# - models/generalist_model.pt
# - models/fighter_specialist.pt
# - models/bomber_specialist.pt
# - models/cargo_specialist.pt
# - models/helicopter_specialist.pt
# - models/attack_aircraft_specialist.pt
# - models/tiltrotor_specialist.pt
```

---

## ✅ Verification Checklist

### **After Pushing Code**

```bash
# 1. Verify main branch updated
git log --oneline -5

# 2. Check GitHub (should show files)
# Visit: https://github.com/USERNAME/THRP

# 3. Verify .gitignore is working
# Should NOT see in repo:
# - .venv/
# - models/*.pt
# - outputs/
# - test/
# - datasets/
```

### **After Uploading Models to Releases**

```bash
# Download link format
https://github.com/USERNAME/THRP/releases/download/v1.0/generalist_model.pt

# Test download
cd tmp
wget https://github.com/USERNAME/THRP/releases/download/v1.0/generalist_model.pt
ls -lh *.pt
```

---

## 📄 File Manifest

### **Code Files** (in `scripts/`)
| File | Purpose | Size |
|------|---------|------|
| `predict.py` | Two-stage inference pipeline | ~11KB |
| `train_specialist_kaggle.py` | Specialist model trainer | ~8KB |
| `train_generalist_kaggle.py` | Generalist model trainer | ~9KB |
| `aircraft_superclass_map.py` | Class mapping & utilities | ~4KB |
| `dataset_organizer.py` | Dataset preparation | ~7KB |
| `preprocessing.py` | Image preprocessing | ~7KB |
| Other scripts | Utilities & legacy | ~20KB |

**Total code**: ~65KB

### **Config Files** (in `models/` and `yolo_specialist_datasets/`)
| File | Size |
|------|------|
| `models/generalist_config_kaggle.yaml` | ~500B |
| `yolo_specialist_datasets/*/config.yaml` | 6 files, ~3KB total |

**Total configs**: ~4KB

### **Documentation** (root directory)
| File | Size |
|------|------|
| `README.md` | ~15KB |
| `IMPLEMENTATION_GUIDE.md` | ~30KB |
| `GITHUB_UPLOAD_GUIDE.md` | ~25KB |
| `DELIVERABLES.md` | ~8KB |
| Others | ~20KB |

**Total docs**: ~98KB

### **Repository Total Size (without models/data)**
```
Code + Config + Docs: ~170KB
With test data: ~500MB (excluded from repo)
With models: +40MB (uploaded to releases, not repo)
With training data: +50GB (excluded from repo)
```

---

## 🔗 Documentation Files to Upload

### **Required**
- ✅ `README.md` — Quick start & feature overview
- ✅ `IMPLEMENTATION_GUIDE.md` — Setup & implementation schema for devs
- ✅ `GITHUB_UPLOAD_GUIDE.md` — This file's parent (for reference)

### **Optional but Helpful**
- ⭐ `DELIVERABLES.md` — Project deliverables
- ⭐ `KAGGLE_INTEGRATION_GUIDE.md` — If using Kaggle datasets
- ⭐ `RESULTS_SUMMARY.txt` — Benchmarks & results

---

## 💾 Alternative: GitHub Packages for Models

If models are too large for Releases (>2GB total):

### **Option 1: HuggingFace Hub** (Recommended)
```python
# Install
pip install huggingface_hub

# Upload
from huggingface_hub import upload_folder
upload_folder(
    folder_path="models/",
    repo_id="USERNAME/THRP-models",
    repo_type="model"
)

# Download in IMPLEMENTATION_GUIDE
from huggingface_hub import snapshot_download
snapshot_download(repo_id="USERNAME/THRP-models", local_dir="models/")
```

### **Option 2: Google Drive**
```python
# Upload models to Google Drive
# Update download link in README/IMPLEMENTATION_GUIDE
# Users download with: wget <gdrive_link>
```

### **Option 3: AWS S3**
```bash
# Upload to S3
aws s3 cp models/*.pt s3://my-bucket/THRP/

# Users download with: aws s3 cp s3://my-bucket/THRP/generalist_model.pt models/
```

---

## 📢 Update README Section (Add This)

```markdown
## 📥 Download Pre-trained Models

Models are hosted in GitHub Releases (not included in repo due to size).

### **Automatic Download**
```bash
./download_models.sh  # Create this script
```

### **Manual Download**
Visit: https://github.com/USERNAME/THRP/releases/download/v1.0/

Download all `.pt` files to `models/` directory.

### **Using HuggingFace Hub**
```bash
pip install huggingface_hub
huggingface-cli download USERNAME/THRP-models --local-dir models/
```
```

---

## 🎯 Final Checklist Before Push

- [ ] `.gitignore` created and correct
- [ ] `README.md` updated with quick start
- [ ] `IMPLEMENTATION_GUIDE.md` explains setup for new devs
- [ ] All `*.py` files in `scripts/`
- [ ] All `*.yaml` config files in `models/` and `yolo_specialist_datasets/`
- [ ] `requirements.txt` has all dependencies
- [ ] No `.venv/`, `__pycache__/`, or `.DS_Store` files
- [ ] No model `*.pt` files in repo (will add to releases)
- [ ] No raw training data in repo
- [ ] `git status` shows only intended files
- [ ] Commit message is descriptive
- [ ] GitHub repository created
- [ ] All files pushed to `main` branch
- [ ] Models uploaded to GitHub Releases

---

## 📞 Post-Upload

### **Share with Team**
```
Repo: https://github.com/USERNAME/THRP
Quick Start: See README.md
Setup: Follow IMPLEMENTATION_GUIDE.md
Models: Download from Releases
Questions?: See GITHUB_UPLOAD_GUIDE.md for schema
```

### **Monitor Issues**
- Check GitHub Issues for questions
- Update README with common problems
- Maintain IMPLEMENTATION_GUIDE

---

*Last Updated: May 15, 2026*
*Status: Ready for GitHub Upload*
