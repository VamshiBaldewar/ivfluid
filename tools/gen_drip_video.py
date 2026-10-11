"""Render a synthetic falling-drop video for M2 tracker validation / E2E demo.

  python -m tools.gen_drip_video --out data/raw/drip_rate/demo_videos/20dpm.mp4 --dpm 20 --seconds 30

The chamber is fixed; drops spawn at the top at the requested rate and fall under
gravity, which is what the tracker counts.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from app.paths import PROJECT_ROOT
from tools.gen_drip_dataset import IMG_H, IMG_W, _base_chamber, _draw_drop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpm", type=float, default=20.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    base, (x0, y0, x1, y1) = _base_chamber(rng)
    cx = (x0 + x1) / 2 + rng.uniform(-6, 6)
    r = 5

    dt = 1.0 / args.fps
    n_frames = int(args.seconds * args.fps)
    interval = 60.0 / max(1.0, args.dpm)

    drops: list[dict] = []
    next_spawn = 0.0
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (IMG_W, IMG_H))

    for i in range(n_frames):
        t = i * dt
        if t >= next_spawn:
            drops.append({"x": cx + rng.uniform(-3, 3), "y": y0 + 6.0, "vy": 55.0})
            next_spawn += interval

        frame = base.copy()
        for d in drops:
            d["vy"] += 430.0 * dt
            d["y"] += d["vy"] * dt
            if y0 + 4 <= d["y"] <= y1 - 4:
                _draw_drop(frame, d["x"], d["y"], r)
        drops = [d for d in drops if d["y"] <= y1 - 4]

        frame = np.clip(
            frame.astype(np.int16) + np.random.normal(0, 4, frame.shape).astype(np.int16), 0, 255
        ).astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"[M2] wrote {n_frames} frames @ {args.dpm} dpm -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
