#!/usr/bin/env python3
"""Automatic pipeline: generalist detector -> resolve superclass -> specialist baseline predictor.

Supports methods: decision_tree, svm, knn, bayes
"""
from pathlib import Path
import argparse
import sys

from aircraft_superclass_map import AIRCRAFT_TO_SUPERCLASS


def resolve_superclass_from_name(aircraft_name: str) -> str | None:
    if not aircraft_name:
        return None
    # If the generalist already returned a superclass label (e.g. 'Bomber'), accept it
    values = set(AIRCRAFT_TO_SUPERCLASS.values())
    for v in values:
        if aircraft_name.strip().lower() == str(v).strip().lower():
            return v

    # Otherwise, try to map aircraft model name -> superclass
    if aircraft_name in AIRCRAFT_TO_SUPERCLASS:
        return AIRCRAFT_TO_SUPERCLASS[aircraft_name]
    key = aircraft_name.strip()
    if key.upper() in AIRCRAFT_TO_SUPERCLASS:
        return AIRCRAFT_TO_SUPERCLASS[key.upper()]
    for k in AIRCRAFT_TO_SUPERCLASS:
        if k.lower() == key.lower():
            return AIRCRAFT_TO_SUPERCLASS[k]
    return None


def run_auto(image_path: Path, method: str, generalist_model: Path, dataset_root: Path, out: Path, conf: float = 0.25):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(f"Failed to import ultralytics YOLO: {exc}")

    model = YOLO(str(generalist_model))
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    if not results:
        raise RuntimeError("No detections from generalist model")
    res = results[0]
    try:
        cls_array = res.boxes.cls.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
    except Exception:
        cls_array = []
        confs = []
        for b in res.boxes:
            cls_array.append(int(b.cls))
            confs.append(float(b.conf))

    if len(cls_array) == 0:
        raise RuntimeError("No boxes predicted")

    best_idx = int(confs.argmax()) if hasattr(confs, "argmax") else max(range(len(confs)), key=lambda i: confs[i])
    best_cls = int(cls_array[best_idx])
    names = getattr(model, "names", None)
    predicted_name = names[best_cls] if names is not None and best_cls in names else str(best_cls)

    # Try to crop to the detected bounding box (with small padding) before extracting features
    try:
        import cv2
        boxes_xyxy = None
        try:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        except Exception:
            try:
                boxes_xyxy = res.boxes.data.cpu().numpy()[:, :4]
            except Exception:
                boxes_xyxy = None

        crop_image = None
        if boxes_xyxy is not None and len(boxes_xyxy) > best_idx:
            x1, y1, x2, y2 = boxes_xyxy[best_idx].astype(int).tolist()
            img = cv2.imread(str(image_path))
            if img is not None:
                h, w = img.shape[:2]
                # add 5% padding
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.05)
                x1p = max(0, x1 - pad_x)
                y1p = max(0, y1 - pad_y)
                x2p = min(w, x2 + pad_x)
                y2p = min(h, y2 + pad_y)
                if x2p > x1p and y2p > y1p:
                    crop_image = img[y1p:y2p, x1p:x2p]
        else:
            crop_image = None
    except Exception:
        crop_image = None

    superclass = resolve_superclass_from_name(predicted_name)
    if superclass is None:
        raise RuntimeError(f"Could not resolve superclass for predicted: {predicted_name}")
    superclass_folder = superclass.lower()

    method = method.lower()
    if method not in {"decision_tree", "svm", "knn", "bayes"}:
        raise ValueError("Unsupported method. Choose decision_tree, svm, knn or bayes")

    model_filename = f"{method}_{superclass_folder}.joblib"
    model_path = Path(out) / method if method != "decision_tree" else Path(out) / "decision_tree"
    model_path = model_path / model_filename
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")

    # load model and predict
    try:
        import joblib
        import cv2
        from extract_features import extract_combined
        from classical_baseline_common import load_class_names_from_config
    except Exception as exc:
        raise RuntimeError(f"Missing deps when preparing prediction: {exc}")

    model_obj = joblib.load(model_path)
    # use cropped image when available
    image = None
    try:
        import cv2 as _cv2
        if crop_image is not None:
            image = crop_image
        else:
            image = _cv2.imread(str(image_path))
    except Exception:
        image = None

    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    # Decide whether the loaded model expects ResNet embeddings (2048) or the HOG+color vectors.
    use_embedding = False
    try:
        n_in = int(getattr(model_obj, "n_features_in_", -1))
        if n_in == 2048:
            use_embedding = True
    except Exception:
        use_embedding = False

    if use_embedding:
        # compute ResNet50 embedding for the ROI/image
        try:
            import torch
            from PIL import Image
            from torchvision import transforms
            try:
                from torchvision.models import resnet50, ResNet50_Weights
                weights = ResNet50_Weights.DEFAULT
                res_model = resnet50(weights=weights)
            except Exception:
                from torchvision.models import resnet50
                res_model = resnet50(pretrained=True)

            import torch.nn as nn
            res_model.fc = nn.Identity()

            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")

            res_model.to(device)
            res_model.eval()

            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            pil = Image.fromarray(image[:, :, ::-1])
            inp = transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = res_model(inp).cpu().numpy()
            features = emb.reshape(1, -1)
        except Exception as exc:
            # fallback to original extractor if embedding computation fails
            try:
                features = extract_combined(image).reshape(1, -1)
            except Exception:
                raise RuntimeError(f"Failed to compute features: {exc}")
    else:
        features = extract_combined(image).reshape(1, -1)
    pred = int(model_obj.predict(features)[0])
    probs = None
    if hasattr(model_obj, "predict_proba"):
        try:
            probs = model_obj.predict_proba(features)[0]
        except Exception:
            probs = None

    class_names = load_class_names_from_config(Path(dataset_root) / superclass_folder)
    predicted_name = class_names[pred] if class_names and pred < len(class_names) else str(pred)
    confidence = float(probs[pred]) if probs is not None and pred < len(probs) else None

    return {
        "superclass": superclass_folder,
        "predicted_name": predicted_name,
        "predicted_index": int(pred),
        "confidence": confidence,
        "generalist_label": predicted_name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--method", required=True, choices=["decision_tree", "svm", "knn", "bayes"]) 
    parser.add_argument("--generalist-model", default="models/generalist_model.pt")
    parser.add_argument("--dataset-root", default="yolo_specialist_datasets")
    parser.add_argument("--out", default="results/baselines")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    out = run_auto(Path(args.image), args.method, Path(args.generalist_model), Path(args.dataset_root), Path(args.out), args.conf)
    print(out)


if __name__ == "__main__":
    main()
