"""THRP Generalist Model Training (Stage 1)"""
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

class GeneralistModelTrainer:
    SUPERCLASSES = ["Fighter", "Bomber", "Transport", "Helicopter"]
    
    def __init__(self, dataset_root: str, output_dir: str = "models"):
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_yolo_config(self) -> str:
        """Create YOLO dataset config"""
        import yaml
        base_dir = self.output_dir.parent / "yolo_dataset"
        train_dir = base_dir / "images" / "train"
        val_dir = base_dir / "images" / "val"
        for d in [train_dir, val_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        config = {
            "path": str(base_dir),
            "train": str(train_dir),
            "val": str(val_dir),
            "nc": len(self.SUPERCLASSES),
            "names": self.SUPERCLASSES
        }
        config_path = self.output_dir / "generalist_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return str(config_path)
    
    def train(self, epochs: int = 50, batch_size: int = 16, device: str = "cpu"):
        logger.info("\n" + "="*70)
        logger.info("THRP Stage 1: Training Generalist Model")
        logger.info("="*70)
        
        config_path = self.create_yolo_config()
        logger.info(f"Loading YOLOv8n model...")
        model = YOLO("yolov8n.pt")
        
        logger.info(f"Training on {device}...")
        try:
            results = model.train(
                data=config_path, epochs=epochs, imgsz=640,
                batch=batch_size, workers=8, device=device, patience=20,
                project=str(self.output_dir),
                name="generalist_stage1", verbose=False
            )
        except Exception as e:
            logger.error(f"Training error: {e}")
            logger.info("Note: Ensure YOLOv8 can access training data")
            return None
        
        model_path = self.output_dir / "generalist_model.pt"
        model.save(str(model_path))
        logger.info(f"✓ Model saved to {model_path}")
        
        return {"model_path": str(model_path), "epochs": epochs}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/Users/sachin/Documents/Bits Pilani/DM_Project_v2/THRP/datasets")
    parser.add_argument("--output", default="/Users/sachin/Documents/Bits Pilani/DM_Project_v2/THRP/models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    trainer = GeneralistModelTrainer(args.dataset, args.output)
    result = trainer.train(epochs=args.epochs, device=args.device)
