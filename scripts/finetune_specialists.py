#!/usr/bin/env python3
"""Fine-tune ResNet50 specialists per superclass.

Workflow:
- Freeze pretrained backbone, train new FC head for `--epochs-head` epochs.
- Optionally unfreeze last ResNet layer and fine-tune for `--epochs-finetune` epochs.
- Uses data augmentation, class-balanced sampling, weight decay, and early stopping.

Saves:
- models/<superclass>_specialist_finetune.pt (full model state_dict)
- results/baselines/cnn/metrics_<superclass>.json

Usage:
  python3 scripts/finetune_specialists.py --dataset yolo_specialist_datasets --superclasses fighter cargo helicopter bomber
"""
from pathlib import Path
import argparse
import json
import time
import copy
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class SpecialistDataset(Dataset):
    def __init__(self, root: Path, split: str, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.img_dir = root / "images" / split
        self.lbl_dir = root / "labels" / split
        self.items = []
        if self.img_dir.exists() and self.lbl_dir.exists():
            for p in sorted(self.img_dir.glob("*.jpg")):
                lbl = self.lbl_dir / f"{p.stem}.txt"
                if not lbl.exists():
                    continue
                try:
                    with open(lbl, 'r', encoding='utf-8') as h:
                        line = h.readline().strip()
                    if not line:
                        continue
                    cls = int(line.split()[0])
                except Exception:
                    continue
                self.items.append((p, cls))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, cls = self.items[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(cls)


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def choose_device(pref='auto'):
    pref = str(pref).lower()
    if pref == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_one(sc, dataset_root: Path, out_models: Path, out_results: Path, epochs_head=3, epochs_finetune=2, batch_size=32, lr_head=1e-3, lr_finetune=1e-4, weight_decay=1e-4, patience=3):
    specialist_root = dataset_root / sc
    train_ds = SpecialistDataset(specialist_root, 'train', transform=build_transforms(True))
    val_ds = SpecialistDataset(specialist_root, 'val', transform=build_transforms(False))

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Not enough data for {sc} (train {len(train_ds)} val {len(val_ds)})")

    # balanced sampler
    labels = [lbl for _, lbl in train_ds.items]
    class_sample_count = np.array([labels.count(i) for i in sorted(set(labels))])
    # compute per-sample weights
    weight_per_class = 1.0 / (class_sample_count + 1e-8)
    sample_weights = [weight_per_class[labels[i]] for i in range(len(labels))]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # build model
    from torchvision.models import resnet50, ResNet50_Weights
    try:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    except Exception:
        model = resnet50(pretrained=True)

    # determine number of classes for this specialist
    all_labels = sorted(set([lbl for _, lbl in train_ds.items] + [lbl for _, lbl in val_ds.items]))
    num_classes = max(all_labels) + 1

    # freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = choose_device('auto')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=lr_head, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val = math.inf
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def evaluate(loader):
        model.eval()
        ys, ys_pred = [], []
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                losses.append(float(loss.cpu().item()))
                preds = torch.argmax(out, dim=1).cpu().numpy().tolist()
                ys_pred.extend(preds)
                ys.extend(yb.cpu().numpy().tolist())
        return np.mean(losses), np.array(ys), np.array(ys_pred)

    # train head
    for epoch in range(epochs_head):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.cpu().item()))

        train_loss = float(np.mean(losses))
        val_loss, ys, ys_pred = evaluate(val_loader)
        val_acc = float(accuracy_score(ys, ys_pred))
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"{sc} head epoch {epoch+1}/{epochs_head}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if epochs_no_improve >= patience:
            print("Early stopping head")
            break

    # load best head
    model.load_state_dict(best_state)

    # optional finetune last block
    if epochs_finetune > 0:
        # unfreeze last layer4
        for name, p in model.named_parameters():
            if name.startswith('layer4') or name.startswith('fc'):
                p.requires_grad = True
            else:
                p.requires_grad = False

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr_finetune, weight_decay=weight_decay)
        best_val_ft = best_val
        epochs_no_improve = 0
        for epoch in range(epochs_finetune):
            model.train()
            losses = []
            for xb, yb in train_loader:
                xb = xb.to(device); yb = yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.cpu().item()))

            train_loss = float(np.mean(losses))
            val_loss, ys, ys_pred = evaluate(val_loader)
            val_acc = float(accuracy_score(ys, ys_pred))
            print(f"{sc} finetune epoch {epoch+1}/{epochs_finetune}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if val_loss < best_val_ft - 1e-4:
                best_val_ft = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping finetune")
                break

        model.load_state_dict(best_state)

    # final evaluation
    val_loss, ys, ys_pred = evaluate(val_loader)
    acc = float(accuracy_score(ys, ys_pred))
    prec = float(precision_score(ys, ys_pred, average='macro', zero_division=0))
    rec = float(recall_score(ys, ys_pred, average='macro', zero_division=0))
    f1 = float(f1_score(ys, ys_pred, average='macro', zero_division=0))
    cm = confusion_matrix(ys, ys_pred).tolist()

    out_models.mkdir(parents=True, exist_ok=True)
    model_path = out_models / f"{sc}_specialist_finetune.pt"
    torch.save(model.state_dict(), model_path)

    out_results.mkdir(parents=True, exist_ok=True)
    metrics = {
        'superclass': sc,
        'method': 'resnet50_finetune',
        'dataset_info': {
            'total_samples': int(len(train_ds)+len(val_ds)),
            'num_classes': int(num_classes),
            'train_size': int(len(train_ds)),
            'val_size': int(len(val_ds)),
        },
        'metrics': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm,
        }
    }
    with open(out_results / f"metrics_{sc}.json", 'w', encoding='utf-8') as h:
        json.dump(metrics, h, indent=2)

    print(f"Saved finetuned model for {sc}: {model_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--superclasses', nargs='*', default=['fighter','cargo','helicopter','bomber'])
    parser.add_argument('--out-models', default='models')
    parser.add_argument('--out-results', default='results/baselines/cnn')
    parser.add_argument('--epochs-head', type=int, default=3)
    parser.add_argument('--epochs-finetune', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr-head', type=float, default=1e-3)
    parser.add_argument('--lr-finetune', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3)
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    out_models = Path(args.out_models)
    out_results = Path(args.out_results)

    summary = {}
    for sc in args.superclasses:
        t0 = time.time()
        try:
            metrics = train_one(sc, dataset_root, out_models, out_results, epochs_head=args.epochs_head, epochs_finetune=args.epochs_finetune, batch_size=args.batch_size, lr_head=args.lr_head, lr_finetune=args.lr_finetune, weight_decay=args.weight_decay, patience=args.patience)
            summary[sc] = metrics
        except Exception as exc:
            summary[sc] = {'error': str(exc)}
        dt = time.time() - t0
        print(f"Completed {sc} in {dt:.1f}s")

    with open(out_results.parent / 'finetune_summary.json', 'w', encoding='utf-8') as h:
        json.dump(summary, h, indent=2)


if __name__ == '__main__':
    main()
