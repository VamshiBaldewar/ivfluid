"""M2 drip-rate YOLOv8 training on the synthetic drip dataset.

  python -m tools.gen_drip_dataset
  python -m models.drip_rate.train --epochs 60 --imgsz 416

Writes models/drip_rate/weights/drip_yolo.pt and metrics/m2_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from app.paths import DATA_PROCESSED, PROJECT_ROOT

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS = WEIGHTS_DIR / "drip_yolo.pt"
METRICS_DIR = Path(__file__).resolve().parent / "metrics"
DATA_YAML = DATA_PROCESSED / "drip_yolo" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "data" / "processed" / "drip_yolo_runs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit("Missing synthetic dataset. Run: python -m tools.gen_drip_dataset")

    from ultralytics import YOLO

    t0 = time.time()
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=0,
        seed=args.seed,
        project=str(RUNS_DIR),
        name="drip_yolov8n",
        exist_ok=True,
        verbose=False,
        plots=False,
    )

    best = RUNS_DIR / "drip_yolov8n" / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished but best.pt missing at {best}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, WEIGHTS)

    metrics = model.val(data=str(DATA_YAML), imgsz=args.imgsz, device="cpu", verbose=False)
    box = getattr(metrics, "box", None)
    summary = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "yolov8n",
        "dataset": str(DATA_YAML.relative_to(PROJECT_ROOT)),
        "data_source": "synthetic drip chamber (tools.gen_drip_dataset)",
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "map50": round(float(box.map50), 4) if box is not None else None,
        "map50_95": round(float(box.map), 4) if box is not None else None,
        "precision": round(float(box.mp), 4) if box is not None else None,
        "recall": round(float(box.mr), 4) if box is not None else None,
        "seconds": round(time.time() - t0, 1),
        "weights": str(WEIGHTS.relative_to(PROJECT_ROOT)),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "m2_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[M2] done. mAP50={summary['map50']} -> {WEIGHTS.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()