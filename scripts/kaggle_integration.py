"""
Kaggle Dataset Integration Script
Organizes and verifies the Kaggle military aircraft detection dataset
into the THRP hierarchical structure.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict

from aircraft_superclass_map import SUPERCLASS_ORDER

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KaggleDatasetIntegrator:
    """Handles integration and organization of Kaggle military aircraft dataset."""

    AIRCRAFT_SUPERCLASS_MAPPING = {
        'F-16': 'Fighter', 'F-18': 'Fighter', 'F-22': 'Fighter', 'Gripen': 'Fighter',
        'MiG-29': 'Fighter', 'Rafale': 'Fighter', 'Su-27': 'Fighter',
        'B-1': 'Bomber', 'B-2': 'Bomber', 'B-52': 'Bomber', 'Tu-160': 'Bomber',
        'A400M': 'Transport', 'Airbus': 'Transport', 'Boeing': 'Transport',
        'C-5': 'Transport', 'C-17': 'Transport', 'C-130': 'Transport',
        'AH-64': 'Helicopter', 'Black Hawk': 'Helicopter', 'CH-47': 'Helicopter',
        'Mi-24': 'Helicopter', 'UH-60': 'Helicopter',
    }

    def __init__(self, dataset_root: str, output_root: str):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.stats = {'total_images': 0, 'total_labels': 0, 'superclass_distribution': {}}

    def verify_dataset_structure(self) -> bool:
        logger.info("Verifying Kaggle dataset structure...")
        if not self.dataset_root.exists():
            logger.error(f"Dataset root not found: {self.dataset_root}")
            return False
        
        for sc in SUPERCLASS_ORDER:
            if not (self.dataset_root / sc).exists():
                logger.warning(f"SuperClass not found: {sc}")
                return False
        
        logger.info("✓ Dataset structure verified successfully")
        return True

    def collect_dataset_info(self) -> Dict:
        logger.info("Collecting dataset information...")
        info = {'superclasses': {}, 'total_images': 0, 'total_labels': 0}
        
        for superclass in SUPERCLASS_ORDER:
            sc_path = self.dataset_root / superclass
            if sc_path.exists():
                info['superclasses'][superclass] = {'aircraft': {}, 'splits': {}}
                
                for split in ['train', 'val', 'test']:
                    split_path = sc_path / split
                    if split_path.exists():
                        split_count = 0
                        for aircraft_dir in split_path.iterdir():
                            if aircraft_dir.is_dir():
                                images_dir = aircraft_dir / 'images'
                                if images_dir.exists():
                                    img_count = len(list(images_dir.glob('*.jpg')))
                                    split_count += img_count
                                    info['superclasses'][superclass]['aircraft'][aircraft_dir.name] = img_count
                                    info['total_images'] += img_count
                        
                        info['superclasses'][superclass]['splits'][split] = split_count
        
        self.stats = info
        return info

    def print_dataset_summary(self):
        logger.info("\n" + "="*80)
        logger.info("KAGGLE DATASET SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Images: {self.stats['total_images']}")
        
        logger.info("\nDistribution by SuperClass:")
        for sc, data in self.stats['superclasses'].items():
            total = sum(data['splits'].values())
            logger.info(f"  {sc}: {total} images")
        
        logger.info("="*80 + "\n")

    def get_dataset_info(self) -> Dict:
        return self.collect_dataset_info()


def main():
    logger.info("Starting Kaggle Dataset Integration...")
    
    dataset_path = "../datasets/data"
    output_path = ".."
    
    integrator = KaggleDatasetIntegrator(dataset_root=dataset_path, output_root=output_path)
    
    if not integrator.verify_dataset_structure():
        logger.error("Dataset verification failed!")
        return False
    
    info = integrator.collect_dataset_info()
    integrator.print_dataset_summary()
    logger.info("✓ Kaggle dataset integration verified!")
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
