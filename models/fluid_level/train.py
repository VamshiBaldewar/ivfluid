"""M1 fluid-level CNN training.

Transfer-learning MobileNetV3-Small on the Mendeley SESOVERA fill-level set
(and any extra numeric subfolders dropped under data/raw/fluid_level/).

  python -m models.fluid_level.train --epochs 24 --batch-size 32 --img-size 224

Classes = integer-named subdirectories of the root (e.g. 0, 50, 80, 100).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from app.paths import DATA_RAW, PROJECT_ROOT

M1_WEIGHTS = Path(__file__).resolve().parent / "weights" / "fluid_level.pt"
M1_METRICS_DIR = Path(__file__).resolve().parent / "metrics"
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_class_dirs(root: Path, max_depth: int = 3) -> dict[int, list[Path]]:
    """Find integer-named folders containing images (e.g. public_mendeley/50)."""
    found: dict[int, list[Path]] = {}
    if not root.exists():
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        rel_parts = here.relative_to(root).parts
        if len(rel_parts) >= max_depth:
            dirnames[:] = []
            continue
        try:
            label = int(here.name)
        except ValueError:
            continue
        imgs = [f for f in filenames if Path(f).suffix.lower() in IMAGE_EXTS]
        if imgs:
            found.setdefault(label, []).append(here)
            dirnames[:] = []
    return found


def load_image_paths(class_dirs: dict[int, list[Path]]) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for label, folders in class_dirs.items():
        for folder in folders:
            for img in folder.iterdir():
                if img.suffix.lower() in IMAGE_EXTS:
                    samples.append((img, label))
    return samples


class ImageFolderFromDisk(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], classes: list[int], transform=None):
        self.samples = samples
        self.classes = classes
        self.idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        from PIL import Image

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.idx[label]


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.75, 1.333)),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2, hue=0.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10, fill=128),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ]
    )
    return train_tf, val_tf


class MobileNetV3Level(nn.Module):
    def __init__(self, n_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)


def eval_model(model, loader, device):
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[list[float]] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.tolist())
            all_probs.extend(probs.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_preds, all_labels, all_probs


def train(args) -> dict:
    t0 = time.time()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M1_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    class_dirs = discover_class_dirs(Path(args.root))
    classes = sorted(class_dirs)
    if len(classes) < 2:
        raise SystemExit(f"No numeric class folders found under {args.root}")
    samples = load_image_paths(class_dirs)
    print(f"[M1] classes={classes} total_images={len(samples)} device={device}")
    if args.limit > 0:
        rng = random.Random(seed)
        rng.shuffle(samples)
        samples = samples[: args.limit]
        print(f"[M1] limited to {len(samples)} images")

    train_tf, val_tf = build_transforms(args.img_size)

    val_n = max(1, int(len(samples) * args.val_split))
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_samples = shuffled[:val_n]
    train_samples = shuffled[val_n:]

    train_ds = ImageFolderFromDisk(train_samples, classes, train_tf)
    val_ds = ImageFolderFromDisk(val_samples, classes, val_tf)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = MobileNetV3Level(n_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict] = []
    best_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        scheduler.step()

        val_acc, _, _, _ = eval_model(model, val_ld, device)
        row = {"epoch": epoch, "train_loss": round(running / max(1, n_batches), 4), "val_acc": round(val_acc, 4)}
        history.append(row)
        print(f"[M1] epoch {epoch}/{args.epochs} loss={row['train_loss']:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": best_state,
                    "classes": classes,
                    "class_to_idx": {str(c): i for i, c in enumerate(classes)},
                    "img_size": args.img_size,
                    "mean": IMG_MEAN,
                    "std": IMG_STD,
                    "model": "mobilenet_v3_small",
                    "val_acc": float(best_acc),
                    "epochs": int(epoch),
                },
                M1_WEIGHTS,
            )

    # Final metrics on the best model.
    model.load_state_dict(torch.load(M1_WEIGHTS, map_location="cpu", weights_only=True)["state_dict"])
    val_acc, preds, labels, probs = eval_model(model, val_ld, device)
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes)))).tolist()
    class_names = [str(c) for c in classes]
    report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)

    metrics = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": class_names,
        "n_classes": len(classes),
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "val_accuracy": round(val_acc, 4),
        "per_class_accuracy": {
            name: round(report[name]["recall"], 4) for name in class_names
        },
        "confusion_matrix": cm,
        "history": history,
        "best_saved_val_acc": best_acc,
        "history_best": round(best_acc, 4),
        "seconds": round(time.time() - t0, 1),
        "weights": str(M1_WEIGHTS.relative_to(PROJECT_ROOT)),
    }

    M1_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (M1_METRICS_DIR / "m1_train_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Loss/accuracy curve plot.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train_loss")
        plt.plot([h["epoch"] for h in history], [h["val_acc"] for h in history], label="val_acc")
        plt.xlabel("epoch")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(M1_METRICS_DIR / "m1_learning_curve.png", dpi=120)
        plt.close()
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[M1] plot skipped: {exc}")

    print(f"[M1] done. val_acc={val_acc:.4f} -> {M1_WEIGHTS.relative_to(PROJECT_ROOT)}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DATA_RAW / "fluid_level"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0, help="0 = use all images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
