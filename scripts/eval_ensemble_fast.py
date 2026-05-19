#!/usr/bin/env python3
"""Evaluate updated ensemble on specialist validation sets efficiently.

Saves results to results/baselines/cnn/ensemble_eval_<superclass>.json
"""
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

from classical_baseline_common import load_class_names_from_config
from extract_features import extract_combined

import joblib
import cv2

try:
    import torch
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights
    import torch.nn as nn
    from PIL import Image
    have_torch = True
except Exception:
    have_torch = False


def build_embedding_model():
    if not have_torch:
        return None
    try:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    except Exception:
        model = resnet50(pretrained=True)
    model.fc = nn.Identity()
    model.eval()
    device = torch.device('cpu')
    model.to(device)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform, device


def compute_embedding_mat(model, transform, device, img_arr):
    pil = Image.fromarray(img_arr[:, :, ::-1])
    inp = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(inp).cpu().numpy()
    return emb.reshape(1, -1)


def load_cnn_for_sc(model_path, sc):
    # build a model with correct fc size and load state_dict
    try:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    except Exception:
        model = resnet50(pretrained=True)
    cfg_names = load_class_names_from_config(Path('yolo_specialist_datasets') / sc)
    if not cfg_names:
        return None
    num_classes = len(cfg_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    sd = torch.load(model_path, map_location='cpu')
    model.load_state_dict(sd)
    model.eval()
    return model


def predict_with_cnn(model, transform, device, img_arr, cfg_names):
    pil = Image.fromarray(img_arr[:, :, ::-1])
    inp = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
        conf = float(probs[pred])
    pred_name = cfg_names[pred] if pred < len(cfg_names) else f"{cfg_names[0].split(':')[0]}:{pred}"
    return pred_name, conf


def main():
    root = Path('yolo_specialist_datasets')
    out_dir = Path('results/baselines/cnn')
    out_dir.mkdir(parents=True, exist_ok=True)
    superclasses = [p.name for p in root.iterdir() if p.is_dir()]

    # prepare embedding model once
    if have_torch:
        emb_model, emb_transform, emb_device = build_embedding_model()
    else:
        emb_model = emb_transform = emb_device = None

    for sc in ['fighter', 'cargo', 'helicopter', 'bomber']:
        sc_root = root / sc
        img_dir = sc_root / 'images' / 'val'
        lbl_dir = sc_root / 'labels' / 'val'
        if not img_dir.exists():
            print('Skipping', sc, 'no val images')
            continue

        # load CNN if exists
        cnn_path = Path('models') / f"{sc}_specialist_finetune.pt"
        cnn_model = None
        cfg_names = load_class_names_from_config(sc_root)
        if cnn_path.exists() and have_torch and cfg_names:
            cnn_model = load_cnn_for_sc(str(cnn_path), sc)

        # load classical classifiers for this sc
        methods = ['decision_tree', 'svm', 'knn', 'bayes']
        classifiers = []  # list of (clf, method)
        for m in methods:
            method_dir = Path('results/baselines') / m
            if not method_dir.exists():
                continue
            for jf in sorted(method_dir.glob(f"*_{sc}.joblib")):
                try:
                    clf = joblib.load(jf)
                    classifiers.append((clf, m))
                except Exception:
                    continue

        total = 0
        correct = 0
        for img_p in sorted(img_dir.glob('*.jpg')):
            total += 1
            img_arr = cv2.imread(str(img_p))
            if img_arr is None:
                continue
            # true label
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            true_idx = None
            if lbl_p.exists():
                try:
                    with open(lbl_p, 'r', encoding='utf-8') as h:
                        line = h.readline().strip()
                        true_idx = int(line.split()[0])
                except Exception:
                    true_idx = None

            chosen = None

            # prefer CNN specialist
            if cnn_model is not None and have_torch and cfg_names:
                try:
                    pred_name, conf = predict_with_cnn(cnn_model, emb_transform, emb_device, img_arr, cfg_names)
                    chosen = pred_name
                except Exception:
                    chosen = None

            if chosen is None:
                # fall back to classical weighted vote across classifiers
                preds = []
                for clf, m in classifiers:
                    try:
                        n_in = getattr(clf, 'n_features_in_', None)
                        if n_in == 2048 and have_torch:
                            feats = compute_embedding_mat(emb_model, emb_transform, emb_device, img_arr)
                        else:
                            feats = extract_combined(img_arr).reshape(1, -1)
                        pred = int(clf.predict(feats)[0])
                        if hasattr(clf, 'predict_proba'):
                            probs = clf.predict_proba(feats)[0]
                            conf = float(probs[pred])
                        else:
                            conf = 1.0
                        # map pred index
                        pred_name = cfg_names[pred] if cfg_names and pred < len(cfg_names) else f"{sc}:{pred}"
                        preds.append((pred_name, conf))
                    except Exception:
                        continue
                if preds:
                    # simple weighted vote
                    scores = defaultdict(float)
                    for n, c in preds:
                        scores[n] += float(c)
                    chosen = max(scores.items(), key=lambda kv: kv[1])[0]

            # compare
            if true_idx is not None and cfg_names:
                true_name = cfg_names[true_idx] if true_idx < len(cfg_names) else f"{sc}:{true_idx}"
                if chosen == true_name:
                    correct += 1

        acc = float(correct) / total if total else 0.0
        out = {
            'superclass': sc,
            'total': total,
            'correct': correct,
            'accuracy': acc,
        }
        out_path = out_dir / f'ensemble_eval_{sc}.json'
        with open(out_path, 'w', encoding='utf-8') as h:
            json.dump(out, h, indent=2)
        print('Saved', out_path, out)


if __name__ == '__main__':
    main()
