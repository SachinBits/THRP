"""Specialist Models Training for Kaggle Dataset"""
import os, json, logging, time
from pathlib import Path
from typing import Dict, List
import yaml, shutil

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Ultralytics not installed. Run: pip install ultralytics")
    exit(1)

from aircraft_superclass_map import SUPERCLASS_ORDER, build_superclass_lists, resolve_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KaggleSpecialistTrainer:
    """Trains specialized models for each SuperClass"""
    AIRCRAFT_BY_SUPERCLASS = build_superclass_lists()
    
    def __init__(self, dataset_root: str, output_dir: str = "models"):
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_specialist_base = self.output_dir.parent / "yolo_specialist_datasets"
    
    def create_specialist_config(self, sc: str) -> str:
        """Create YOLO config for specialist"""
        aircraft_list = self.AIRCRAFT_BY_SUPERCLASS.get(sc, [])
        if not aircraft_list:
            raise ValueError(f"Unknown superclass: {sc}")
        
        specialist_dir = self.yolo_specialist_base / sc.lower()
        train_dir = specialist_dir / "images" / "train"
        val_dir = specialist_dir / "images" / "val"
        train_labels_dir = specialist_dir / "labels" / "train"
        val_labels_dir = specialist_dir / "labels" / "val"

        if specialist_dir.exists():
            shutil.rmtree(specialist_dir)
        
        for d in [train_dir, val_dir, train_labels_dir, val_labels_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self._flatten_specialist(sc, train_dir, train_labels_dir, aircraft_list, 'train')
        self._flatten_specialist(sc, val_dir, val_labels_dir, aircraft_list, 'val')
        
        config = {
            "path": str(specialist_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": len(aircraft_list),
            "names": {i: a for i, a in enumerate(aircraft_list)}
        }
        
        config_path = specialist_dir / f"{sc.lower()}_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"  ✓ {sc} config: {len(aircraft_list)} classes")
        return str(config_path)
    
    def _flatten_specialist(self, sc: str, img_dir: Path, label_dir: Path,
                           aircraft_list: List[str], split: str):
        """Flatten specialist dataset"""
        total = 0
        sc_path = self.dataset_root / sc / split

        if not sc_path.exists():
            # Try common alternative split names (e.g., 'validation' instead of 'val')
            alternatives = ["validation", "valid", "test", "training"]
            found = False
            for alt in alternatives:
                alt_path = self.dataset_root / sc / alt
                if alt_path.exists():
                    logger.info(f"Using alternative split '{alt}' for {sc}: {alt_path}")
                    sc_path = alt_path
                    found = True
                    break

            if not found:
                logger.warning(f"Split not found: {sc_path}")
                return
        
        for a_idx, aircraft in enumerate(aircraft_list):
            a_dir = sc_path / aircraft
            if not a_dir.exists():
                continue
            
            src_imgs = a_dir / "images"
            if src_imgs.exists():
                for img in sorted(src_imgs.glob("*.jpg")):
                    try:
                        shutil.copy2(img, img_dir / img.name)
                    except:
                        pass
            
            src_labels = a_dir / "labels"
            if src_labels.exists():
                for lbl in sorted(src_labels.glob("*.txt")):
                    try:
                        with open(lbl, "r") as f:
                            lines = f.readlines()
                        
                        converted = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                parts[0] = str(a_idx)
                                converted.append(" ".join(parts) + "\n")
                        
                        with open(label_dir / lbl.name, "w") as f:
                            f.writelines(converted)
                        total += 1
                    except:
                        pass
        
        logger.info(f"    {split}: {total} labels")
    
    def train_specialist(self, sc: str, epochs=50, batch_size=16, device="cpu", imgsz=640, augment=False):
        """Train specialist for superclass"""
        logger.info("="*80)
        logger.info(f"THRP Stage 2: Training {sc} Specialist Model")
        logger.info("="*80)
        
        config_path = self.create_specialist_config(sc)
        
        logger.info(f"Loading YOLOv8n for {sc}...")
        model = YOLO("yolov8n.pt")
        
        device = resolve_device(device)
        logger.info(f"Training {sc}... (epochs: {epochs})")
        results = model.train(
            data=config_path, epochs=epochs, imgsz=imgsz, batch=batch_size,
            patience=20, workers=8, device=device, project=str(self.output_dir),
            name=f"specialist_{sc.lower()}", save=True, verbose=True, close_mosaic=10,
            augment=augment
        )
        
        model_path = self.output_dir / f"{sc.lower()}_specialist.pt"
        model.save(str(model_path))
        logger.info(f"✓ {sc} specialist saved: {model_path}")
        
        info = {
            "superclass": sc,
            "aircraft_models": self.AIRCRAFT_BY_SUPERCLASS[sc],
            "num_classes": len(self.AIRCRAFT_BY_SUPERCLASS[sc]),
            "model_path": str(model_path)
        }
        
        with open(self.output_dir / f"{sc.lower()}_specialist_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        logger.info("="*80 + "\n")
        return info
    
    def train_all(self, epochs=50, batch_size=16, device="cpu", imgsz=640, augment=False):
        """Train all specialists"""
        logger.info("THRP Stage 2: Training All Specialists\n")
        
        superclasses = SUPERCLASS_ORDER
        all_results = {}
        
        for i, sc in enumerate(superclasses, 1):
            logger.info(f"[{i}/4] {sc} specialist...")
            results = self.train_specialist(sc, epochs, batch_size, device, imgsz, augment)
            all_results[sc] = results
            
            if i < len(superclasses):
                logger.info("Pausing 2 seconds...")
                time.sleep(2)
        
        logger.info("✓ All Specialists Training Complete!")
        
        with open(self.output_dir / "specialist_training_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        return all_results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../datasets/data")
    parser.add_argument("--output", default="./models")
    parser.add_argument("--superclass", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    
    trainer = KaggleSpecialistTrainer(args.dataset, args.output)
    
    if args.all or args.superclass is None:
        trainer.train_all(args.epochs, args.batch_size, args.device, args.imgsz, args.augment)
    elif args.superclass:
        trainer.train_specialist(args.superclass, args.epochs, args.batch_size, args.device, args.imgsz, args.augment)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
