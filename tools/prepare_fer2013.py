"""Download FER2013 (HuggingFace, login-free) and cache as 48x48 grayscale arrays.

  python -m tools.prepare_fer2013

Writes data/processed/fer2013/{train,test}_{x,y}.npy
Classes: angry, disgust, fear, happy, neutral, sad, surprise (index order).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.paths import DATA_PROCESSED, PROJECT_ROOT

OUT = DATA_PROCESSED / "fer2013"
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _decode(split) -> tuple[np.ndarray, np.ndarray]:
    xs = np.zeros((len(split), 48, 48), dtype=np.uint8)
    ys = np.zeros((len(split),), dtype=np.int64)
    for i, row in enumerate(split):
        buf = np.frombuffer(row["img_bytes"], dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (48, 48):
            img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
        xs[i] = img
        ys[i] = int(row["labels"])
    return xs, ys


def main() -> None:
    from datasets import load_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    print("[M3] loading Jeneral/fer-2013 ...")
    ds = load_dataset("Jeneral/fer-2013")
    for split_name, out_name in (("train", "train"), ("test", "test")):
        split = ds[split_name]
        xs, ys = _decode(split)
        np.save(OUT / f"{out_name}_x.npy", xs)
        np.save(OUT / f"{out_name}_y.npy", ys)
        counts = {CLASSES[c]: int((ys == c).sum()) for c in range(len(CLASSES))}
        print(f"[M3] {split_name}: {xs.shape} -> {out_name} counts={counts}")
    (OUT / "classes.txt").write_text("\n".join(CLASSES), encoding="utf-8")
    print(f"[M3] cached -> {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()