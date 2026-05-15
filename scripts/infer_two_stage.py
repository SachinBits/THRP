"""Two-stage inference: generalist -> specialist per detection.

Example:
  python3 scripts/infer_two_stage.py --image examples/img1.jpg --generalist models/generalist_model.pt --specialists models/ --out out.json
"""
import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
import json

from aircraft_superclass_map import SUPERCLASS_ORDER, resolve_device


SUPERCLASSES = SUPERCLASS_ORDER


def load_specialist_names(specialists_base: Path, sc: str):
    cfg = specialists_base.parent / 'yolo_specialist_datasets' / sc.lower() / f"{sc.lower()}_config.yaml"
    if cfg.exists():
        try:
            with open(cfg) as f:
                data = yaml.safe_load(f)
            names = data.get('names')
            if isinstance(names, dict):
                return [names[str(i)] for i in range(len(names))]
        except Exception:
            pass

    # fallback: try model.names
    model_path = specialists_base / f"{sc.lower()}_specialist.pt"
    if model_path.exists():
        try:
            m = YOLO(str(model_path))
            if hasattr(m, 'model') and hasattr(m.model, 'names'):
                return [m.model.names[i] for i in sorted(m.model.names.keys())]
        except Exception:
            pass

    return []


def crop_image(img, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = img.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
    return img[y1:y2, x1:x2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--generalist', default='models/generalist_model.pt')
    p.add_argument('--specialists', default='models')
    p.add_argument('--conf', type=float, default=0.3)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--device', default='auto')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit('Image not found: ' + str(img_path))

    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit('Failed to read image')

    device = resolve_device(args.device)
    gen_model = YOLO(str(args.generalist))
    preds = gen_model.predict(str(img_path), conf=args.conf, iou=args.iou, device=device)

    results = []
    specialists_base = Path(args.specialists)

    for p0 in preds:
        boxes = p0.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy.numpy()
            cls = int(box.cls[0].cpu().numpy()) if hasattr(box.cls, 'cpu') else int(box.cls.numpy())
            score = float(box.conf[0].cpu().numpy()) if hasattr(box.conf, 'cpu') else float(box.conf.numpy())

            sc = SUPERCLASSES[cls] if cls < len(SUPERCLASSES) else 'Unknown'

            crop = crop_image(img, xyxy)
            if crop.size == 0:
                continue

            spec_model_path = specialists_base / f"{sc.lower()}_specialist.pt"
            specialist_names = load_specialist_names(specialists_base, sc)

            specialist_pred = None
            if spec_model_path.exists():
                try:
                    sm = YOLO(str(spec_model_path))
                    spreds = sm.predict(crop, conf=0.2, device=device)
                    # take top prediction if exists
                    for s in spreds:
                        if hasattr(s, 'boxes') and len(s.boxes) > 0:
                            b = s.boxes[0]
                            scls = int(b.cls[0].cpu().numpy()) if hasattr(b.cls, 'cpu') else int(b.cls.numpy())
                            sconf = float(b.conf[0].cpu().numpy()) if hasattr(b.conf, 'cpu') else float(b.conf.numpy())
                            name = specialist_names[scls] if scls < len(specialist_names) else str(scls)
                            specialist_pred = {'name': name, 'class_id': scls, 'score': sconf}
                            break
                except Exception:
                    specialist_pred = None

            results.append({
                'bbox': [float(x) for x in xyxy.tolist()],
                'superclass': sc,
                'generalist_score': score,
                'specialist': specialist_pred
            })

    out = {'image': str(img_path), 'detections': results}
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print('Wrote', args.out)
    else:
        print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
