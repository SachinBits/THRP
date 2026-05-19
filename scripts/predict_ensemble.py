#!/usr/bin/env python3
"""Ensemble predictor: queries THRP specialist (if available) and classical baselines, then fuses predictions.

Usage:
  python3 scripts/predict_ensemble.py --image test/F14.jpg
"""
from pathlib import Path
import argparse
from collections import defaultdict
import statistics


def normalize_name(n: str) -> str:
    return str(n).strip()


def weighted_vote(preds):
    # preds: list of (name, confidence)
    scores = defaultdict(float)
    counts = defaultdict(int)
    for name, conf in preds:
        name = normalize_name(name)
        w = float(conf) if conf is not None else 0.5
        scores[name] += w
        counts[name] += 1

    if not scores:
        return None, None
    best = max(scores.items(), key=lambda kv: (kv[1], counts[kv[0]]))
    # also compute mean confidence
    chosen = best[0]
    confs = [c for n, c in preds if normalize_name(n) == chosen and c is not None]
    mean_conf = float(statistics.mean(confs)) if confs else None
    return chosen, mean_conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--generalist-model", default="models/generalist_model.pt")
    parser.add_argument("--specialists-dir", default="models")
    parser.add_argument("--baselines-out", default="results/baselines")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="If THRP specialist confidence >= threshold, prefer it")
    args = parser.parse_args()

    img = Path(args.image)
    if not img.exists():
        raise SystemExit(f"Image not found: {img}")

    # We'll avoid calling ultralytics/generalist here (may not be installed). Instead
    # load each classical model saved under results/baselines/<method>/ and evaluate it
    # on the cropped/whole image. For each method we examine all saved specialist models
    # (one per superclass) and pick the most confident prediction.

    methods = ["decision_tree", "svm", "knn", "bayes"]
    baseline_preds = []
    thrp_pred = None

    # helper: compute features matching model input dim
    from extract_features import extract_combined
    try:
        import torch
        from PIL import Image
        from torchvision import transforms
        import torch.nn as nn
        from torchvision.models import resnet50, ResNet50_Weights
        have_torch = True
    except Exception:
        have_torch = False

    def compute_embedding(img_arr):
        if not have_torch:
            return None
        # Build a fresh ResNet50 (cpu) quickly
        try:
            weights = ResNet50_Weights.DEFAULT
            res_model = resnet50(weights=weights)
        except Exception:
            res_model = resnet50(pretrained=True)
        res_model.fc = nn.Identity()
        device = torch.device("cpu")
        res_model.to(device)
        res_model.eval()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pil = Image.fromarray(img_arr[:, :, ::-1])
        inp = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = res_model(inp).cpu().numpy()
        return emb.reshape(1, -1)

    def run_cnn_specialist(model_path, img_arr):
        """Load a finetuned ResNet specialist and run it on img_arr.
        Returns (pred_name, confidence) or (None, 0.0) on error."""
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            import torch.nn as nn
            device = torch.device("cpu")
            # determine superclass from filename
            stem = Path(model_path).stem
            sc = stem.split("_")[0]
            # load class names to reconstruct final layer size
            cfg_names = load_class_names_from_config(Path("yolo_specialist_datasets") / sc)
            if not cfg_names:
                return None, 0.0
            num_classes = len(cfg_names)

            # build model
            try:
                weights = ResNet50_Weights.DEFAULT
                model = resnet50(weights=weights)
            except Exception:
                model = resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            # load state dict
            sd = torch.load(model_path, map_location=device)
            model.load_state_dict(sd)
            model.to(device)
            model.eval()

            # preprocess
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            pil = Image.fromarray(img_arr[:, :, ::-1])
            inp = transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred = int(probs.argmax())
                conf = float(probs[pred])
            pred_name = cfg_names[pred] if pred < len(cfg_names) else f"{sc}:{pred}"
            return pred_name, conf
        except Exception:
            return None, 0.0

    import joblib
    import cv2
    from classical_baseline_common import load_class_names_from_config

    img_arr = cv2.imread(str(img))
    if img_arr is None:
        raise SystemExit(f"Failed to read image: {img}")

    # Run any finetuned CNN specialists present in the specialists dir
    cnn_preds = []  # list of (pred_name, conf, path)
    spec_dir = Path(args.specialists_dir)
    if spec_dir.exists():
        for p in sorted(spec_dir.glob("*_specialist_finetune.pt")):
            pred_name, conf = run_cnn_specialist(p, img_arr)
            if pred_name:
                cnn_preds.append((pred_name, float(conf), p))

    for m in methods:
        method_dir = Path(args.baselines_out) / m
        best_for_method = (None, 0.0)  # (name, confidence)
        if not method_dir.exists():
            baseline_preds.append((f"missing:{m}", 0.0))
            continue

        for model_file in sorted(method_dir.glob("*.joblib")):
            try:
                clf = joblib.load(model_file)
            except Exception:
                continue

            # determine expected input dim
            n_in = getattr(clf, "n_features_in_", None)
            if n_in == 2048:
                feats = compute_embedding(img_arr)
                if feats is None:
                    feats = extract_combined(img_arr).reshape(1, -1)
            else:
                feats = extract_combined(img_arr).reshape(1, -1)

            pred_name = None
            conf_val = 0.0
            try:
                pred = int(clf.predict(feats)[0])
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(feats)[0]
                    conf_val = float(probs[pred])
                else:
                    conf_val = 1.0
            except Exception:
                continue

            # map pred index to class name via dataset config or embeddings class_names
            sc = model_file.stem.split("_")[-1]
            cfg_names = load_class_names_from_config(Path("yolo_specialist_datasets") / sc)
            if cfg_names and pred < len(cfg_names):
                pred_name = cfg_names[pred]
            else:
                pred_name = f"{sc}:{pred}"

            if conf_val > best_for_method[1]:
                best_for_method = (pred_name, conf_val)

        baseline_preds.append(best_for_method)

    # Aggregation logic
    # Priority: THRP specialist (if confident) -> finetuned CNN specialist -> classical ensemble
    chosen = None
    chosen_conf = None
    chosen_source = None

    if thrp_pred and thrp_pred[1] >= args.conf_threshold:
        chosen = normalize_name(thrp_pred[0])
        chosen_conf = thrp_pred[1]
        chosen_source = "THRP_specialist"
    else:
        # prefer CNN specialist if any is present and confident
        if cnn_preds:
            best_cnn = max(cnn_preds, key=lambda x: x[1])
            if best_cnn[1] >= args.conf_threshold:
                chosen = normalize_name(best_cnn[0])
                chosen_conf = best_cnn[1]
                chosen_source = f"cnn_specialist:{best_cnn[2].name}"
        if chosen is None:
            chosen, chosen_conf = weighted_vote(baseline_preds)
            chosen_source = "classical_ensemble"

    result = {
        "image": str(img),
        "thrp": thrp_pred,
        "baselines": baseline_preds,
        "ensemble_choice": chosen,
        "ensemble_confidence": chosen_conf,
        "choice_source": chosen_source,
    }

    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
