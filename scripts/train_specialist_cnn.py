"""Fine-tune ResNet50 specialists per superclass.

Saves weights to `models/<superclass>_specialist.pt` by default.

Usage:
  python3 scripts/train_specialist_cnn.py --dataset yolo_specialist_datasets --superclasses fighter cargo bomber helicopter --epochs 10 --batch-size 32
"""
from pathlib import Path
import argparse
import time
import copy
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np


class YoloLabelDataset(Dataset):
    def __init__(self, root: Path, split: str = "train", transform=None):
        self.root = root
        self.img_dir = root / "images" / split
        self.lbl_dir = root / "labels" / split
        self.transform = transform
        self.samples = []
        for p in sorted(self.img_dir.glob("*.jpg")):
            lbl = self.lbl_dir / f"{p.stem}.txt"
            if not lbl.exists():
                continue
            try:
                with open(lbl, "r", encoding="utf-8") as h:
                    line = h.readline().strip()
                if not line:
                    continue
                cls = int(line.split()[0])
            except Exception:
                continue
            self.samples.append((p, cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, cls = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, cls


def build_dataloaders(specialist_root: Path, batch_size: int, workers: int = 4):
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = YoloLabelDataset(specialist_root, split="train", transform=transform_train)
    val_ds = YoloLabelDataset(specialist_root, split="val", transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    num_classes = 0
    if len(train_ds) > 0:
        num_classes = int(max([s[1] for s in train_ds.samples]) + 1)
    elif len(val_ds) > 0:
        num_classes = int(max([s[1] for s in val_ds.samples]) + 1)

    return train_loader, val_loader, num_classes


def build_model(num_classes: int, device: torch.device):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)
    return model


def train_one_specialist(specialist_root: Path, out_dir: Path, epochs: int, batch_size: int, lr: float, device: torch.device):
    train_loader, val_loader, num_classes = build_dataloaders(specialist_root, batch_size)
    if num_classes <= 0:
        raise RuntimeError(f"No classes found for {specialist_root}")

    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = running_corrects / total if total > 0 else 0.0

        # validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += float(loss.item()) * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += inputs.size(0)

        val_loss = val_loss / val_total if val_total > 0 else 0.0
        val_acc = val_corrects / val_total if val_total > 0 else 0.0

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={time.time()-t0:.1f}s")

    # save best model
    model.load_state_dict(best_model_wts)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{specialist_root.name.lower().replace(' ', '_')}_specialist.pt"
    torch.save(model.state_dict(), out_path)
    # also save class names if available
    try:
        import yaml
        yaml_candidates = [
            specialist_root / f"{specialist_root.name}_config.yaml",
            specialist_root / f"{specialist_root.name.lower()}_config.yaml",
            specialist_root / f"{specialist_root.name.upper()}_config.yaml",
        ]
        class_names = []
        for y in yaml_candidates:
            if not y.exists():
                continue
            with open(y, "r", encoding="utf-8") as h:
                cfg = yaml.safe_load(h)
            names = cfg.get("names", {})
            if isinstance(names, dict):
                class_names = [str(names.get(index, f"Class_{index}")) for index in sorted(names.keys())]
            elif isinstance(names, list):
                class_names = [str(n) for n in names]
            break
        if class_names:
            with open(out_dir / f"{specialist_root.name.lower()}_class_names.json", "w", encoding="utf-8") as h:
                json.dump(class_names, h, indent=2)
    except Exception:
        pass

    print(f"Saved specialist model: {out_path} (best_val_acc={best_acc:.4f})")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Root of yolo_specialist_datasets")
    parser.add_argument("--superclasses", nargs="*", default=["fighter", "cargo", "helicopter", "bomber"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="models")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # select device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out)
    dataset_root = Path(args.dataset)

    for sc in args.superclasses:
        sc_root = dataset_root / sc
        if not sc_root.exists():
            print(f"Skipping {sc}: path not found: {sc_root}")
            continue
        print(f"Training specialist for {sc} -> {out_dir} on device {device}")
        try:
            train_one_specialist(sc_root, out_dir, args.epochs, args.batch_size, args.lr, device)
        except Exception as exc:
            print(f"Failed {sc}: {exc}")


if __name__ == "__main__":
    main()
