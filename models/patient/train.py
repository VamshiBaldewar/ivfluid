"""M3 patient-distress model training (FER2013 emotion CNN).

Trains a compact 7-class emotion CNN on the cached FER2013 arrays, then maps
negative emotions (angry, disgust, fear, sad) to a continuous distress score.

  python -m tools.prepare_fer2013
  python -m models.patient.train --epochs 45 --batch-size 256

Outputs: models/patient/weights/emotion_fer.pt, metrics/m3_metrics.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from app.paths import DATA_PROCESSED, PROJECT_ROOT

FER_DIR = DATA_PROCESSED / "fer2013"
WEIGHTS = Path(__file__).resolve().parent / "weights" / "emotion_fer.pt"
METRICS_DIR = Path(__file__).resolve().parent / "metrics"
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DISTRESS_IDX = [0, 1, 2, 5]  # angry, disgust, fear, sad


class EmotionCNN(nn.Module):
    def __init__(self, n_classes: int = 7):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
            )

        self.features = nn.Sequential(block(1, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class FERDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, train: bool):
        self.x = x
        self.y = y
        self.train = train

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        img = torch.from_numpy(self.x[i].astype(np.float32) / 255.0).unsqueeze(0)
        img = (img - 0.5) / 0.5
        if self.train:
            # pad + random crop
            pad = torch.from_numpy(np.pad(img.numpy(), ((0, 0), (4, 4), (4, 4)), mode="edge"))
            top = random.randint(0, 8)
            left = random.randint(0, 8)
            img = pad[:, top : top + 48, left : left + 48]
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2])
            if random.random() < 0.5:
                img = img * random.uniform(0.85, 1.15)
            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                img = _rotate(img, angle)
        return img.contiguous(), int(self.y[i])


def _rotate(t: torch.Tensor, angle: float) -> torch.Tensor:
    # t: [1,48,48]; rotate via affine_grid
    theta = torch.tensor(
        [[np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0.0],
         [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0.0]],
        dtype=torch.float32,
    ).unsqueeze(0)
    grid = F.affine_grid(theta, (1, 1, 48, 48), align_corners=False)
    return F.grid_sample(t.unsqueeze(0), grid, align_corners=False)[0]


def _softmax_probs(model, x, device, batch=512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.as_tensor(x[i : i + batch], dtype=torch.float32)
            xb = ((xb / 255.0) - 0.5) / 0.5
            xb = xb.unsqueeze(1).to(device)
            p = torch.softmax(model(xb), dim=1)
            p = p + torch.softmax(model(torch.flip(xb, dims=[3])), dim=1)  # hflip TTA
            out.append((p / 2).cpu().numpy())
    return np.concatenate(out, axis=0)


def _distress_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    distress_prob = probs[:, DISTRESS_IDX].sum(axis=1)
    truth = np.isin(y, DISTRESS_IDX).astype(int)
    pred = (distress_prob > 0.5).astype(int)
    acc = float((pred == truth).mean())
    try:
        auroc = float(roc_auc_score(truth, distress_prob))
    except ValueError:
        auroc = None
    return {"distress_accuracy": round(acc, 4), "distress_auroc": round(auroc, 4) if auroc else None,
            "positive_rate": round(float(truth.mean()), 4)}


def train(args) -> dict:
    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_tr = np.load(FER_DIR / "train_x.npy")
    y_tr = np.load(FER_DIR / "train_y.npy")
    x_te = np.load(FER_DIR / "test_x.npy")
    y_te = np.load(FER_DIR / "test_y.npy")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(y_tr))
    n_val = int(len(perm) * 0.1)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    train_ds = FERDataset(x_tr[tr_idx], y_tr[tr_idx], train=True)
    val_ds = FERDataset(x_tr[val_idx], y_tr[val_idx], train=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = EmotionCNN(len(CLASSES)).to(device)
    history = []
    best_val = -1.0
    if args.resume and WEIGHTS.exists():
        checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        baseline_probs = _softmax_probs(model, x_tr[val_idx], device)
        best_val = float((baseline_probs.argmax(1) == y_tr[val_idx]).mean())
        print(f"[M3] resumed checkpoint; baseline val_acc={best_val:.4f}")

    counts = np.bincount(y_tr[tr_idx], minlength=len(CLASSES)).astype(np.float32)
    weights = (counts.sum() / (counts + 1e-6))
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, nb = 0.0, 0
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
            nb += 1
        scheduler.step()
        val_probs = _softmax_probs(model, x_tr[val_idx], device)
        val_acc = float((val_probs.argmax(1) == y_tr[val_idx]).mean())
        history.append({"epoch": epoch, "train_loss": round(running / max(1, nb), 4), "val_acc": round(val_acc, 4)})
        print(f"[M3] epoch {epoch}/{args.epochs} loss={history[-1]['train_loss']:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "classes": CLASSES,
                    "distress_idx": DISTRESS_IDX,
                    "size": 48,
                    "val_acc": float(val_acc),
                },
                WEIGHTS,
            )

    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True)["state_dict"])
    test_probs = _softmax_probs(model, x_te, device)
    test_acc = float((test_probs.argmax(1) == y_te).mean())
    cm = confusion_matrix(y_te, test_probs.argmax(1), labels=list(range(len(CLASSES)))).tolist()
    per_class = {
        CLASSES[c]: round(float((test_probs[y_te == c].argmax(1) == c).mean()), 4)
        for c in range(len(CLASSES))
        if (y_te == c).any()
    }

    metrics = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "EmotionCNN (FER2013, hflip TTA)",
        "classes": CLASSES,
        "distress_classes": [CLASSES[i] for i in DISTRESS_IDX],
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(y_te)),
        "val_accuracy": round(float(best_val), 4),
        "test_accuracy_7class": round(test_acc, 4),
        "test_per_class_accuracy": per_class,
        "confusion_matrix": cm,
        **_distress_metrics(test_probs, y_te),
        "history": history,
        "seconds": round(time.time() - t0, 1),
        "weights": str(WEIGHTS.relative_to(PROJECT_ROOT)),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "m3_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[M3] done. 7-class test acc={test_acc:.4f} distress acc={metrics.get('distress_accuracy')} "
          f"auroc={metrics.get('distress_auroc')} -> {WEIGHTS.relative_to(PROJECT_ROOT)}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="fine-tune existing weights and keep them unless validation improves")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()