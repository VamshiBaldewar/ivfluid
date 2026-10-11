"""M2 drip-rate tracker validation against synthetic clips with known dpm.

  python -m tools.gen_drip_video --out data/raw/drip_rate/demo_videos/20dpm.mp4 --dpm 20 --seconds 18
  python -m models.drip_rate.validate
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from app.paths import DATA_RAW, PROJECT_ROOT
from models.drip_rate.infer import infer_drip_rate, reset_tracker

METRICS = Path(__file__).resolve().parent / "metrics" / "m2_validation.json"
DEFAULT_DIR = DATA_RAW / "drip_rate" / "demo_videos"


def evaluate_video(path: Path, target_dpm: float, stride: int = 4) -> dict:
    reset_tracker()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    estimates: list[float] = []
    idx = 0
    ok, frame = cap.read()
    while ok:
        if idx % stride == 0:
            res = infer_drip_rate(frame, ts=idx / fps)
            if res.status == "ready" and res.drops_per_min is not None:
                estimates.append(res.drops_per_min)
        idx += 1
        ok, frame = cap.read()
    cap.release()
    # Ignore warmup (first 25% of ready estimates) for a steady-state estimate.
    warm = estimates[len(estimates) // 4 :] or estimates
    est = float(np.median(warm)) if warm else None
    return {
        "video": str(path.relative_to(PROJECT_ROOT)),
        "target_dpm": target_dpm,
        "estimated_dpm": round(est, 2) if est is not None else None,
        "abs_error": round(abs(est - target_dpm), 2) if est is not None else None,
        "n_ready": len(estimates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    vids = sorted(Path(args.dir).glob("*dpm.mp4"))
    if not vids:
        raise SystemExit(f"No *dpm.mp4 clips in {args.dir}. Run tools.gen_drip_video first.")

    rows = []
    for v in vids:
        m = re.search(r"(\d+)dpm", v.name)
        if not m:
            continue
        rows.append(evaluate_video(v, float(m.group(1)), args.stride))

    errors = [r["abs_error"] for r in rows if r["abs_error"] is not None]
    summary = {
        "clips": rows,
        "mae_dpm": round(float(np.mean(errors)), 3) if errors else None,
        "n_clips": len(rows),
        "note": "synthetic falling-drop clips; tracker validated on frame-timestamped replay",
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for r in rows:
        print(f"[M2] {Path(r['video']).name}: target={r['target_dpm']} est={r['estimated_dpm']} err={r['abs_error']}")
    print(f"[M2] MAE={summary['mae_dpm']} dpm over {summary['n_clips']} clips")


if __name__ == "__main__":
    main()