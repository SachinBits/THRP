"""THRP Inference Pipeline (predict.py)"""
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from datetime import datetime
import logging

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Install ultralytics: pip install ultralytics")
    exit(1)

from aircraft_superclass_map import AIRCRAFT_TO_SUPERCLASS, build_superclass_lists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class THRPInferencePipeline:
    AIRCRAFT_BY_SUPERCLASS = build_superclass_lists()
    SUPERCLASSES = list(AIRCRAFT_BY_SUPERCLASS.keys())
    
    def __init__(self, generalist_model_path: str, specialist_models_dir: str):
        logger.info("Initializing THRP Pipeline...")
        self.gen_model_path = Path(generalist_model_path)
        self.spec_models_dir = Path(specialist_models_dir)
        self.specialist_model_paths = self._discover_specialist_models()
        
        if self.gen_model_path.exists():
            self.generalist_model = YOLO(str(self.gen_model_path))
            logger.info("✓ Generalist model loaded")
        else:
            logger.warning(f"Generalist model not found: {self.gen_model_path}")
            self.generalist_model = None
        
        self.specialist_models = {}
        for superclass, spec_path in self.specialist_model_paths.items():
            self.specialist_models[superclass] = YOLO(str(spec_path))
        logger.info(f"✓ {len(self.specialist_models)} specialist models loaded")

    def _discover_specialist_models(self) -> Dict[str, Path]:
        """Find trained specialist weights in the expected directory."""
        models: Dict[str, Path] = {}

        if not self.spec_models_dir.exists():
            logger.warning(f"Specialist models directory not found: {self.spec_models_dir}")
            return models

        for spec_path in sorted(self.spec_models_dir.glob("*_specialist.pt")):
            superclass = spec_path.stem.replace("_specialist", "").strip().lower()
            normalized = superclass.replace(" ", "")

            for known_superclass in self.AIRCRAFT_BY_SUPERCLASS:
                if known_superclass.lower().replace(" ", "") == normalized:
                    models[known_superclass] = spec_path
                    break

        return models
    
    def detect_superclass(self, image: np.ndarray) -> Dict[str, Any]:
        """Stage 1: Detect SuperClass"""
        if self.generalist_model is None:
            return {"success": False, "detections": [], "error": "Model not loaded"}
        
        results = self.generalist_model.predict(image, conf=0.1, verbose=False)
        detections = []
        
        for result in results:
            if result.boxes and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    
                    if cls_id < len(self.SUPERCLASSES):
                        detections.append({
                            "superclass": self.SUPERCLASSES[cls_id],
                            "confidence": conf,
                            "bbox": tuple(bbox)
                        })
        
        return {"success": len(detections) > 0, "detections": detections}
    
    def identify_aircraft(self, image: np.ndarray, bbox: Tuple, superclass: str) -> Dict:
        """Stage 2: Identify specific aircraft"""
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return {"aircraft_model": "Unknown", "confidence": 0.0}

        original_h, original_w = roi.shape[:2]
        logger.debug(f"  Original ROI: {original_w}x{original_h}")

        # Make ROI square with padding to preserve aspect ratio and help specialist detection
        # Specialists were trained on full aircraft images at 768x768
        side = max(original_h, original_w)
        if side < 512:
            side = 512  # Minimum size to upscale to
        
        # Create square canvas and paste ROI in center
        square_roi = np.zeros((side, side, 3), dtype=roi.dtype)
        y_off = (side - original_h) // 2
        x_off = (side - original_w) // 2
        square_roi[y_off:y_off + original_h, x_off:x_off + original_w] = roi
        
        logger.debug(f"  Padded to square: {side}x{side}")

        # Use square ROI for specialist prediction
        if superclass not in self.specialist_models:
            return {"aircraft_model": f"{superclass} (Unknown)", "confidence": 0.0}

        model = self.specialist_models[superclass]
        results = model.predict(square_roi, conf=0.01, verbose=False)
        
        # Use class names directly from the model to avoid index mismatch
        class_names = list(model.names.values())
        
        best_aircraft = None
        best_conf = 0.0
        
        for result in results:
            if result.boxes and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    logger.debug(f"    {superclass} specialist: cls_id={cls_id}, conf={conf:.4f}, aircraft={class_names[cls_id] if cls_id < len(class_names) else 'OOB'}")
                    if cls_id < len(class_names) and conf > best_conf:
                        best_aircraft = class_names[cls_id]
                        best_conf = conf
            else:
                logger.debug(f"    {superclass} specialist: No boxes detected")
        
        return {
            "aircraft_model": self._format_aircraft_name(best_aircraft) if best_aircraft else f"{superclass} (Unknown)",
            "confidence": best_conf if best_aircraft else 0.0
        }
    
    def process_image(self, image_path: str, keep_all_detections: bool = False) -> Dict[str, Any]:
        """Process image through pipeline"""
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": "Failed to read image"}
        
        logger.info(f"Processing: {image_path}")
        
        # Stage 1
        stage1 = self.detect_superclass(image)
        if not stage1["success"]:
            return {"success": False, "error": "No detections"}
        
        # Stage 2
        results = []
        for det in stage1["detections"]:
            stage2 = self.identify_aircraft(image, det["bbox"], det["superclass"])
            combined_score = det["confidence"] * stage2["confidence"]
            results.append({
                "superclass": det["superclass"],
                "superclass_confidence": det["confidence"],
                "aircraft_model": stage2["aircraft_model"],
                "aircraft_confidence": stage2["confidence"],
                "combined_confidence": combined_score,
                "bbox": det["bbox"]
            })

        # Re-rank by specialist-supported confidence. This suppresses superclass false positives
        # when the specialist cannot identify an aircraft for that branch.
        results.sort(key=lambda r: r["combined_confidence"], reverse=True)

        if not keep_all_detections and results:
            results = [results[0]]
        
        return {"success": True, "results": results, "image": image}
    
    def visualize(self, pipeline_results: Dict, output_path: Optional[str] = None) -> np.ndarray:
        """Visualize results"""
        image = pipeline_results["image"].copy()
        
        for res in pipeline_results["results"]:
            x1, y1, x2, y2 = res["bbox"]
            color = self._get_color(res["superclass"])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            label = f"{res['aircraft_model']} ({res['aircraft_confidence']:.1%})"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
            logger.info(f"✓ Visualization saved: {output_path}")
        
        return image
    
    def _get_color(self, superclass: str) -> Tuple:
        colors = {
            "Fighter": (0, 255, 0),
            "Bomber": (0, 0, 255),
            "Cargo": (255, 0, 0),
            "Attack Aircraft": (0, 255, 255),
            "Helicopter": (255, 255, 0),
            "Tiltrotor": (255, 0, 255),
            "Recon / Patrol": (128, 255, 0),
            "Reconnaissance": (128, 255, 128),
            "Interceptor": (0, 128, 255),
            "Multirole Combat": (255, 128, 0),
            "AEW&C": (128, 128, 255),
            "Tanker": (128, 0, 255),
            "Amphibious Aircraft": (0, 128, 128),
            "Drone": (200, 200, 0),
            "Prototype Aircraft": (200, 0, 200),
        }
        return colors.get(superclass, (255, 255, 255))

    def _format_aircraft_name(self, aircraft: Optional[str]) -> str:
        if not aircraft:
            return "Unknown"

        match = re.fullmatch(r"([A-Za-z]+)(\d+)", aircraft)
        if match:
            prefix, number = match.groups()
            if prefix.upper() == "F":
                return f"F-{number}"
            if prefix.upper() == "B":
                return f"B-{number}"
            if prefix.upper() == "C":
                return f"C-{number}"
            if prefix.upper() == "J":
                return f"J-{number}"
            if prefix.upper() == "T" and aircraft.upper() != "TB2":
                return f"T-{number}"

        return aircraft.replace("MiG", "MiG-")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--generalist", default="/Users/sachin/Documents/Bits Pilani/DM_Project_v2/THRP/models/generalist_model.pt")
    parser.add_argument("--specialists", default="/Users/sachin/Documents/Bits Pilani/DM_Project_v2/THRP/models")
    parser.add_argument("--output", default="/Users/sachin/Documents/Bits Pilani/DM_Project_v2/THRP/outputs")
    parser.add_argument("--visualize", action="store_true", help="Save visualization of the results")
    parser.add_argument("--all-detections", action="store_true", help="Show all detections instead of the best-ranked one")
    args = parser.parse_args()
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    pipeline = THRPInferencePipeline(args.generalist, args.specialists)
    results = pipeline.process_image(args.image, keep_all_detections=args.all_detections)
    
    print("\n" + "="*70)
    print("INFERENCE RESULTS")
    print("="*70)
    if results["success"]:
        for r in results["results"]:
            print(f"\nSuperClass: {r['superclass']} ({r['superclass_confidence']:.2%})")
            print(f"Aircraft: {r['aircraft_model']} ({r['aircraft_confidence']:.2%})")
            print(f"Combined: {r['combined_confidence']:.2%}")
            print(f"BBox: {r['bbox']}")
            
        if args.visualize:
            image_name = Path(args.image).name
            output_file = Path(args.output) / f"pred_{image_name}"
            pipeline.visualize(results, str(output_file))
    else:
        print(f"Error: {results.get('error')}")
    print("="*70)
