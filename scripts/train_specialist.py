"""THRP Specialist Models Training (Stage 2)"""
import json
from pathlib import Path
import logging

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Install ultralytics: pip install ultralytics")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpecialistModelTrainer:
    SUPERCLASSES = ["Fighter", "Bomber", "Transport", "Helicopter"]
    AIRCRAFT_BY_SUPERCLASS = {
        "Fighter": ["F-16", "F-18", "F-22", "F-35", "MiG-29", "Gripen", "Rafale"],
        "Bomber": ["B-52", "B-2", "Tu-160", "B-1"],
        "Transport": ["C-130", "C-17", "A400M", "C-5", "Airbus", "Boeing"],
        "Helicopter": ["UH-60", "CH-47", "AH-64", "Mi-24", "Black Hawk"],
    }
    
    def __init__(self, dataset_root: str, output_dir: str = "models"):
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.specialist_dir = self.output_dir / "specialist_models"
        self.specialist_dir.mkdir(parents=True, exist_ok=True)
        
    def create_specialist_config(self, superclass: str) -> str:
        """Create config for specialist model"""
        import yaml
        aircraft_types = self.AIRCRAFT_BY_SUPERCLASS[superclass]
        base_dir = self.output_dir.parent / f"yolo_specialist_{superclass.lower()}"
        for d in [base_dir / "images" / "train", base_dir / "images" / "val"]:
            d.mkdir(parents=True, exist_ok=True)
        
        config = {
            "path": str(base_dir),
            "train": str(base_dir / "images" / "train"),
            "val": str(base_dir / "images" / "val"),
            "nc": len(aircraft_types),
            "names": aircraft_types
        }
        config_path = self.specialist_dir / f"{superclass.lower()}_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return str(config_path)
    
    def train_specialist(self, superclass: str, epochs: int = 50, device: str = "cpu"):
        logger.info(f"\nTraining {superclass} Specialist Model...")
        config_path = self.create_specialist_config(superclass)
        
        model = YOLO("yolov8n.pt")
        try:
            results = model.train(
                data=config_path, epochs=epochs, imgsz=640,
                batch=16, device=device, patience=15,
                project=str(self.specialist_dir),
                name=f"specialist_{superclass.lower()}", verbose=False
            )
        except Exception as e:
            logger.error(f"Training error: {e}")
            return None
        
        model_path = self.specialist_dir / f"{superclass.lower()}_specialist.pt"
        model.save(str(model_path))
        logger.info(f"✓ {superclass} specialist saved")
        return str(model_path)
    
    def train_all(self, epochs: int = 50, device: str = "cpu"):
        logger.info("\n" + "="*70)
        logger.info("THRP Stage 2: Training Specialist Models")
        logger.info("="*70)
        
        results = {}
        for superclass in self.SUPERCLASSES:
            result = self.train_specialist(superclass, epochs, device)
            results[superclass] = result
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/Users/sachin/Documents/Bits Pilani/DM_Project_v2/THRP/datasets")
    parser.add_argument("--output", default="/Users/sachin/Documents/Bits Pilani/DM_Project_v2/THRP/models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    trainer = SpecialistModelTrainer(args.dataset, args.output)
    results = trainer.train_all(epochs=args.epochs, device=args.device)
