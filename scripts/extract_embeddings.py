"""Extract ResNet50 embeddings for specialist datasets and cache them.

Saves per-superclass:
- embeddings.npy (N x D)
- labels.npy
- image_paths.txt
- class_names.json

Usage:
  python3 scripts/extract_embeddings.py --dataset yolo_specialist_datasets --out results/embeddings --batch-size 32
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np

import torch
from torchvision import transforms
from PIL import Image

DEFAULT_SUPERCLASSES = ["fighter", "cargo", "helicopter", "bomber"]


def build_model(device: torch.device):
    try:
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    except Exception:
        from torchvision.models import resnet50
        model = resnet50(pretrained=True)

    # remove final fc
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def image_loader(path: Path, transform):
    img = Image.open(path).convert("RGB")
    return transform(img)


def process_specialist(specialist_root: Path, out_dir: Path, device: torch.device, batch_size: int = 32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = []
    labels = []

    for split in ("train", "val"):
        img_dir = specialist_root / "images" / split
        lbl_dir = specialist_root / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            try:
                with open(lbl_path, "r", encoding="utf-8") as h:
                    line = h.readline().strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
            except Exception:
                continue
            image_paths.append(img_path)
            labels.append(class_id)

    if not image_paths:
        raise RuntimeError(f"No images found under {specialist_root}")

    model = build_model(device)

    embeddings = []
    labels_arr = []
    paths_out = []

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_imgs = []
            for p in batch_paths:
                try:
                    t = image_loader(p, transform)
                    batch_imgs.append(t)
                except Exception:
                    batch_imgs.append(torch.zeros(3, 224, 224))

            batch_tensor = torch.stack(batch_imgs).to(device)
            feats = model(batch_tensor).cpu().numpy()
            embeddings.append(feats)
            for pidx, p in enumerate(batch_paths):
                paths_out.append(str(p))
                labels_arr.append(labels[i + pidx])

    X = np.concatenate(embeddings, axis=0)
    y = np.array(labels_arr, dtype=np.int32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", X)
    np.save(out_dir / "labels.npy", y)
    with open(out_dir / "image_paths.txt", "w", encoding="utf-8") as h:
        for p in paths_out:
            h.write(p + "\n")

    # try to read class names
    yaml_candidates = [
        specialist_root / f"{specialist_root.name}_config.yaml",
        specialist_root / f"{specialist_root.name.lower()}_config.yaml",
        specialist_root / f"{specialist_root.name.upper()}_config.yaml",
    ]
    class_names = []
    for yaml_path in yaml_candidates:
        if not yaml_path.exists():
            continue
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as h:
                cfg = yaml.safe_load(h)
            names = cfg.get("names", {})
            if isinstance(names, dict):
                class_names = [str(names.get(index, f"Class_{index}")) for index in sorted(names.keys())]
            elif isinstance(names, list):
                class_names = [str(n) for n in names]
            break
        except Exception:
            continue

    with open(out_dir / "class_names.json", "w", encoding="utf-8") as h:
        json.dump(class_names, h, indent=2)

    return X.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--superclasses", nargs="*", default=DEFAULT_SUPERCLASSES)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    out_root = Path(args.out)

    # device selection
    dev = args.device
    if dev == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(dev)

    for sc in args.superclasses:
        sc_root = dataset_root / sc
        if not sc_root.exists():
            print(f"Skipping {sc}: {sc_root} not found")
            continue
        out_dir = out_root / sc
        print(f"Processing embeddings for {sc} -> {out_dir} on device {device}")
        t0 = time.time()
        try:
            shape = process_specialist(sc_root, out_dir, device, args.batch_size)
        except Exception as exc:
            print(f"Failed {sc}: {exc}")
            continue
        dt = time.time() - t0
        print(f"Completed {sc} in {dt:.1f}s, saved embeddings shape {shape}")


if __name__ == "__main__":
    main()
