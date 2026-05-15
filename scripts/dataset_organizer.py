"""
Dataset Organizer for THRP Pipeline
Downloads and organizes aircraft data into hierarchical structure
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetOrganizer:
    """Organizes aircraft dataset into hierarchical SuperClass/SubClass structure."""
    
    AIRCRAFT_SUPERCLASS_MAPPING = {
        "F-16": "Fighter", "F-18": "Fighter", "F-22": "Fighter", "F-35": "Fighter",
        "MiG-29": "Fighter", "Gripen": "Fighter", "Rafale": "Fighter",
        "B-52": "Bomber", "B-2": "Bomber", "Tu-160": "Bomber", "B-1": "Bomber",
        "C-130": "Transport", "C-17": "Transport", "A400M": "Transport",
        "C-5": "Transport", "Airbus": "Transport", "Boeing": "Transport",
        "UH-60": "Helicopter", "CH-47": "Helicopter", "AH-64": "Helicopter",
        "Mi-24": "Helicopter", "Black Hawk": "Helicopter",
    }
    
    def __init__(self, root_dir: str = "/tmp/aircraft_dataset"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.root_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
    def create_hierarchical_structure(self) -> None:
        """Create empty hierarchical directory structure."""
        superclasses = set(self.AIRCRAFT_SUPERCLASS_MAPPING.values())
        
        for superclass in superclasses:
            superclass_dir = self.data_dir / superclass
            superclass_dir.mkdir(exist_ok=True)
            
            splits = ["train", "val", "test"]
            for split in splits:
                split_dir = superclass_dir / split
                split_dir.mkdir(exist_ok=True)
                
                for aircraft, sc in self.AIRCRAFT_SUPERCLASS_MAPPING.items():
                    if sc == superclass:
                        aircraft_dir = split_dir / aircraft
                        aircraft_dir.mkdir(exist_ok=True)
                        (aircraft_dir / "images").mkdir(exist_ok=True)
                        (aircraft_dir / "labels").mkdir(exist_ok=True)
        
        logger.info(f"Hierarchical structure created at {self.data_dir}")
        
    def create_synthetic_dataset(self, num_per_class: int = 50) -> None:
        """Create synthetic dataset for testing."""
        self.create_hierarchical_structure()
        logger.info(f"Creating synthetic dataset with {num_per_class} images per class...")
        
        splits = {"train": 0.7, "val": 0.15, "test": 0.15}
        image_count = 0
        
        for superclass, subclasses in self._get_superclass_subclasses().items():
            for split, ratio in splits.items():
                split_num = int(num_per_class * ratio)
                
                for aircraft in subclasses:
                    aircraft_dir = self.data_dir / superclass / split / aircraft
                    images_dir = aircraft_dir / "images"
                    labels_dir = aircraft_dir / "labels"
                    
                    for i in range(split_num):
                        img = self._create_synthetic_aircraft_image(
                            superclass, aircraft, image_size=(640, 480)
                        )
                        img_path = images_dir / f"{aircraft}_{split}_{i:04d}.jpg"
                        cv2.imwrite(str(img_path), img)
                        
                        label_path = labels_dir / f"{aircraft}_{split}_{i:04d}.txt"
                        with open(label_path, "w") as f:
                            f.write("0 0.5 0.5 0.4 0.6\n")
                        
                        image_count += 1
        
        logger.info(f"Synthetic dataset created: {image_count} images total")
        
    def _get_superclass_subclasses(self) -> Dict[str, List[str]]:
        """Get grouping of aircraft by superclass."""
        result = {}
        for aircraft, superclass in self.AIRCRAFT_SUPERCLASS_MAPPING.items():
            if superclass not in result:
                result[superclass] = []
            result[superclass].append(aircraft)
        return result
    
    def _create_synthetic_aircraft_image(self, superclass: str, aircraft: str,
                                         image_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """Create a synthetic aircraft-like image."""
        h, w = image_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(h):
            intensity = int(255 * (i / h))
            img[i, :] = [100 + intensity // 2, 150 + intensity // 3, 200 - intensity // 4]
        
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        center = (w // 2, h // 2)
        color = self._get_color_by_superclass(superclass)
        
        if superclass == "Fighter":
            pts = np.array([[w//2, h//3], [w//2-80, 2*h//3], [w//2-40, 2*h//3],
                           [w//2+40, 2*h//3], [w//2+80, 2*h//3]])
            cv2.polylines(img, [pts], True, color, 3)
            cv2.circle(img, center, 60, color, -1)
        elif superclass == "Bomber":
            cv2.rectangle(img, (w//4, h//3), (3*w//4, 2*h//3), color, 3)
            cv2.circle(img, center, 70, color, -1)
        elif superclass == "Transport":
            cv2.ellipse(img, center, (100, 60), 0, 0, 360, color, 3)
            cv2.rectangle(img, (w//2-120, h//2-20), (w//2+120, h//2+20), color, 2)
        elif superclass == "Helicopter":
            cv2.circle(img, center, 50, color, 3)
            cv2.line(img, (center[0]-100, center[1]), (center[0]+100, center[1]), color, 2)
            cv2.line(img, (center[0], center[1]-100), (center[0], center[1]+100), color, 2)
        
        cv2.putText(img, aircraft, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img
    
    def _get_color_by_superclass(self, superclass: str) -> Tuple[int, int, int]:
        """Get BGR color for superclass."""
        colors = {
            "Fighter": (0, 255, 0), "Bomber": (0, 0, 255),
            "Transport": (255, 0, 0), "Helicopter": (0, 255, 255),
        }
        return colors.get(superclass, (255, 255, 255))
    
    def get_dataset_info(self) -> Dict:
        """Get information about organized dataset."""
        info = {
            "root_dir": str(self.data_dir),
            "superclasses": list(set(self.AIRCRAFT_SUPERCLASS_MAPPING.values())),
            "aircraft_count": len(self.AIRCRAFT_SUPERCLASS_MAPPING),
            "structure": {}
        }
        
        for superclass in info["superclasses"]:
            sc_path = self.data_dir / superclass
            if sc_path.exists():
                aircraft = [d.name for d in (sc_path / "train").iterdir() if d.is_dir()]
                info["structure"][superclass] = {
                    "aircraft_types": aircraft,
                    "count": len(aircraft)
                }
        
        return info


if __name__ == "__main__":
    organizer = DatasetOrganizer(root_dir="/Users/sachin/Documents/Bits Pilani/DM_Project_v2/THRP/datasets")
    organizer.create_synthetic_dataset(num_per_class=30)
    info = organizer.get_dataset_info()
    print("\n" + "="*60)
    print("THRP Dataset Created Successfully!")
    print("="*60)
    print(json.dumps(info, indent=2))
