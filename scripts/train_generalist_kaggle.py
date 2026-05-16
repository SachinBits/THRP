"""Generalist Model Training for Kaggle Dataset"""
import os, json, logging, time
from pathlib import Path
from typing import Dict
import yaml, shutil
import subprocess
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Ultralytics not installed. Run: pip install ultralytics")
    exit(1)

from aircraft_superclass_map import SUPERCLASS_ORDER, resolve_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KaggleGeneralistTrainer:
    """Trains Stage 1 Generalist Model for SuperClass detection"""
    
    SUPERCLASSES = SUPERCLASS_ORDER
    
    def __init__(self, dataset_root: str, output_dir: str = "models"):
        candidate_roots = [Path(dataset_root)]
        script_root = Path(__file__).resolve().parent.parent
        candidate_roots.append(script_root / dataset_root)
        candidate_roots.append(script_root / "datasets" / "data")
        self.dataset_root = next((path for path in candidate_roots if path.exists()), Path(dataset_root))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.output_dir / "generalist_config_kaggle.yaml"
        self.yolo_dataset_dir = self.output_dir.parent / "yolo_generalist_dataset"
    
    def verify_kaggle_structure(self) -> bool:
        logger.info("Verifying Kaggle dataset structure...")
        if not self.dataset_root.exists():
            logger.error(f"Dataset root not found: {self.dataset_root}")
            return False
        
        for sc in self.SUPERCLASSES:
            if not (self.dataset_root / sc).exists():
                logger.warning(f"SuperClass not found: {sc}")
                return False
            logger.info(f"✓ Found {sc}")
        
        logger.info("✓ Dataset structure verified")
        return True
    
    def create_yolo_dataset_config(self) -> str:
        """Create YOLO config from Kaggle structure"""
        logger.info("Creating YOLO dataset configuration...")
        
        train_dir = self.yolo_dataset_dir / "images" / "train"
        val_dir = self.yolo_dataset_dir / "images" / "val"
        train_labels_dir = self.yolo_dataset_dir / "labels" / "train"
        val_labels_dir = self.yolo_dataset_dir / "labels" / "val"

        if self.yolo_dataset_dir.exists():
            shutil.rmtree(self.yolo_dataset_dir)
        
        for d in [train_dir, val_dir, train_labels_dir, val_labels_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        for split, img_dir, label_dir in [
            ('train', train_dir, train_labels_dir),
            ('val', val_dir, val_labels_dir)
        ]:
            self._flatten_kaggle_structure(img_dir, label_dir, split)

        # Ensure there's at least a small validation set. If val has no labels,
        # copy 5% of train (deterministic) into val so Ultralytics won't fail.
        try:
            train_lbls = sorted(train_labels_dir.glob('*.txt'))
            val_lbls = sorted(val_labels_dir.glob('*.txt'))
            if len(val_lbls) == 0 and len(train_lbls) > 0:
                import random
                random.seed(0)
                k = max(1, int(len(train_lbls) * 0.05))
                sampled = random.sample(train_lbls, k)
                logger.info(f"No val labels found - creating val split with {k} samples")
                for src in sampled:
                    # copy label
                    dst_lbl = val_labels_dir / src.name
                    shutil.copy2(src, dst_lbl)

                    # copy corresponding image if present in images/train
                    src_img = train_dir / src.with_suffix('.jpg').name
                    if not src_img.exists():
                        # try other extensions
                        for ext in ('.jpeg', '.png'):
                            alt = train_dir / src.with_suffix(ext).name
                            if alt.exists():
                                src_img = alt
                                break
                    if src_img.exists():
                        shutil.copy2(src_img, val_dir / src_img.name)
        except Exception:
            logger.exception('Failed to create fallback val split')
        
        # Use relative train/val paths (relative to `path`) to satisfy Ultralytics
        rel_train = os.path.relpath(train_dir, self.yolo_dataset_dir)
        rel_val = os.path.relpath(val_dir, self.yolo_dataset_dir)
        config = {
            "path": str(self.yolo_dataset_dir),
            "train": rel_train,
            "val": rel_val,
            "nc": len(self.SUPERCLASSES),
            # names as a list
            "names": [sc for sc in self.SUPERCLASSES]
        }
        
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"✓ YOLO config created: {self.config_path}")
        return str(self.config_path)
    
    def _flatten_kaggle_structure(self, img_dir: Path, label_dir: Path, split: str):
        """Flatten Kaggle structure for YOLO"""
        total = 0
        
        for sc_idx, sc in enumerate(self.SUPERCLASSES):
            sc_path = self.dataset_root / sc / split
            if not sc_path.exists():
                continue
            
            for aircraft_dir in sorted(sc_path.iterdir()):
                if not aircraft_dir.is_dir():
                    continue
                
                src_imgs = aircraft_dir / "images"
                if src_imgs.exists():
                    for img in sorted(src_imgs.glob("*.jpg")):
                        try:
                            shutil.copy2(img, img_dir / img.name)
                        except Exception as e:
                            logger.debug(f"Skip: {img}")
                
                src_labels = aircraft_dir / "labels"
                if src_labels.exists():
                    for lbl in sorted(src_labels.glob("*.txt")):
                        try:
                            with open(lbl, "r") as f:
                                lines = f.readlines()
                            converted = []
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    parts[0] = str(sc_idx)
                                    converted.append(" ".join(parts) + "\n")
                            
                            with open(label_dir / lbl.name, "w") as f:
                                f.writelines(converted)
                            total += 1
                        except:
                            pass
        
        logger.info(f"  {split}: {total} labels processed")
    
    def train(self, epochs=50, batch_size=16, device="cpu"):
        """Train generalist model"""
        logger.info("="*80)
        logger.info("THRP Stage 1: Training Generalist Model (Kaggle Dataset)")
        logger.info("="*80)
        
        if not self.verify_kaggle_structure():
            return {}
        
        config_path = self.create_yolo_dataset_config()
        
        logger.info("Loading YOLOv8n...")
        model = YOLO("yolov8n.pt")
        
        device = resolve_device(device)
        logger.info(f"Training on {device}... (epochs: {epochs}, batch: {batch_size})")
        results = model.train(
            data=config_path, epochs=epochs, imgsz=640, batch=batch_size,
            patience=20, workers=8, device=device, project=str(self.output_dir),
            name="generalist_kaggle", save=True, verbose=True, close_mosaic=10
        )
        
        model_path = self.output_dir / "generalist_model.pt"
        model.save(str(model_path))
        logger.info(f"✓ Model saved: {model_path}")
        
        info = {
            "model_type": "Generalist (Stage 1) - Kaggle Dataset",
            "base_model": "YOLOv8n",
            "superclasses": self.SUPERCLASSES,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "model_path": str(model_path)
        }
        
        with open(self.output_dir / "generalist_training_info_kaggle.json", "w") as f:
            json.dump(info, f, indent=2)
        
        logger.info("✓ Training Complete!")
        logger.info("="*80)
        # Attempt to save validation metrics (non-fatal)
        try:
            subprocess.run([sys.executable, "scripts/save_metrics.py", "--model", str(model_path), "--data", str(self.config_path)], check=False)
        except Exception:
            logger.exception("Failed to run metrics saver for generalist")

        return info

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/data")
    parser.add_argument("--output", default="./models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    
    trainer = KaggleGeneralistTrainer(args.dataset, args.output)
    trainer.train(args.epochs, args.batch_size, args.device)

if __name__ == "__main__":
    main()
