"""Synthetic drip-chamber dataset generator for M2 (YOLOv8 drop detector).

The real IEEE/Roboflow drip sets are behind account walls and the team has no
rig footage yet, so this renders a physically-plausible drip chamber with falling
drops and emits YOLO-format bounding boxes. Output:

  data/processed/drip_yolo/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt
    data.yaml

  python -m tools.gen_drip_dataset --train 480 --val 120

Every box is class 0 ("drop"). Counts at inference come from tracking, not this.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from app.paths import DATA_PROCESSED, PROJECT_ROOT

OUT_ROOT = DATA_PROCESSED / "drip_yolo"
IMG_W, IMG_H = 416, 416


def _base_chamber(rng: random.Random) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return (image, chamber box as x0,y0,x1,y1)."""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    bg = rng.randint(20, 70)
    img[:] = (bg, bg, bg)
    # subtle vertical gradient background wall
    grad = np.linspace(bg - 10, bg + 25, IMG_H).astype(np.uint8)
    img[:] = grad[:, None, None]

    # chamber occupies a vertical band
    cw = rng.randint(int(IMG_W * 0.22), int(IMG_W * 0.34))
    x0 = rng.randint(int(IMG_W * 0.28), int(IMG_W * 0.55))
    y0 = rng.randint(int(IMG_H * 0.06), int(IMG_H * 0.16))
    y1 = IMG_H - rng.randint(int(IMG_H * 0.06), int(IMG_H * 0.18))
    x1 = min(IMG_W - 2, x0 + cw)

    # glassy chamber fill (light)
    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 205, 210), -1)
    # side shading
    cv2.line(img, (x0, y0), (x0, y1), (150, 158, 165), 2)
    cv2.line(img, (x1, y0), (x1, y1), (240, 242, 245), 2)
    cv2.line(img, (x0, y0), (x1, y0), (170, 176, 182), 2)

    # fluid column at the bottom of the chamber
    frac = rng.uniform(0.15, 0.5)
    fy = int(y1 - (y1 - y0) * frac)
    cv2.rectangle(img, (x0 + 2, fy), (x1 - 2, y1), (120, 150, 170), -1)

    # specular highlight stripe
    hx = x0 + int((x1 - x0) * rng.uniform(0.15, 0.35))
    cv2.line(img, (hx, y0 + 4), (hx, y1 - 4), (245, 248, 250), 1)
    return img, (x0, y0, x1, y1)


def _draw_drop(img: np.ndarray, cx: float, cy: float, r: int) -> tuple[int, int, int, int]:
    cv2.circle(img, (int(cx), int(cy)), r, (235, 240, 245), -1)
    cv2.circle(img, (int(cx), int(cy)), r, (255, 255, 255), 1)
    # tiny highlight
    cv2.circle(img, (int(cx - r * 0.3), int(cy - r * 0.3)), max(1, r // 3), (255, 255, 255), -1)
    return int(cx - r), int(cy - r), int(cx + r), int(cy + r)


def render_frame(rng: random.Random, n_drops: int | None = None):
    img, (x0, y0, x1, y1) = _base_chamber(rng)
    if n_drops is None:
        n_drops = rng.randint(1, 4)
    boxes = []
    for _ in range(n_drops):
        cx = rng.uniform(x0 + 6, x1 - 6)
        cy = rng.uniform(y0 + 8, y1 - 8)
        r = rng.randint(3, max(4, (x1 - x0) // 10))
        boxes.append(_draw_drop(img, cx, cy, r))

    # sensor noise + slight blur for realism
    noise = np.random.normal(0, rng.uniform(2, 8), img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img, boxes


def _write_split(split: str, count: int, rng: random.Random) -> int:
    img_dir = OUT_ROOT / "images" / split
    lbl_dir = OUT_ROOT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i in range(count):
        n = rng.randint(0, 4)  # allow empty frames (hard negatives)
        img, boxes = render_frame(rng, n_drops=n)
        name = f"{split}_{i:05d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
        lines = []
        for (bx0, by0, bx1, by1) in boxes:
            cx = (bx0 + bx1) / 2 / IMG_W
            cy = (by0 + by1) / 2 / IMG_H
            w = (bx1 - bx0) / IMG_W
            h = (by1 - by0) / IMG_H
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=480)
    parser.add_argument("--val", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n_tr = _write_split("train", args.train, rng)
    n_va = _write_split("val", args.val, rng)

    yaml = (
        f"path: {OUT_ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['drop']\n"
    )
    (OUT_ROOT / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"[M2] synthetic drip dataset -> {OUT_ROOT.relative_to(PROJECT_ROOT)} (train={n_tr}, val={n_va})")


if __name__ == "__main__":
    main()
