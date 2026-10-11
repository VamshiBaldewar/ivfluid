"""CLI dataset recorder (same folders as the dashboard).

Examples:
  python -m app.collect --module drip_rate --label 20dpm --seconds 60
  python -m app.collect --module fluid_level --label 50 --frames 40 --interval 1
"""

from __future__ import annotations

import argparse
import time

from app.capture import FrameSource, save_frame
from app.paths import DATA_RAW
from app.pipeline import _safe_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Record labeled IV dataset clips/frames.")
    parser.add_argument("--module", required=True, choices=["fluid_level", "drip_rate", "patient"])
    parser.add_argument("--label", required=True, help="e.g. 25, 50, 75 or 15dpm")
    parser.add_argument("--source", default="0", help="Webcam index or video path")
    parser.add_argument("--seconds", type=float, default=0, help="Record video for N seconds")
    parser.add_argument("--frames", type=int, default=0, help="Save N still frames")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between stills")
    args = parser.parse_args()

    source: int | str = int(args.source) if args.source.isdigit() else args.source
    cam = FrameSource(source)
    if not cam.open():
        raise SystemExit(f"Could not open camera/source: {args.source}")

    module = _safe_token(args.module)
    label = _safe_token(args.label)
    dest = DATA_RAW / module / label
    dest.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    try:
        if args.frames > 0:
            print(f"Saving {args.frames} frames → {dest}")
            for i in range(1, args.frames + 1):
                frame = cam.read()
                if frame is None:
                    raise SystemExit("Lost camera while capturing frames")
                save_frame(frame, dest / f"{stamp}_{i:04d}.jpg")
                print(f"  {i}/{args.frames}")
                time.sleep(max(0.05, args.interval))
        else:
            seconds = args.seconds if args.seconds > 0 else 30
            path = dest / f"{stamp}.mp4"
            if not cam.start_recording(path):
                raise SystemExit("Could not start video writer")
            print(f"Recording {seconds:.0f}s → {path}")
            end = time.time() + seconds
            while time.time() < end:
                if cam.read() is None:
                    raise SystemExit("Lost camera while recording")
                time.sleep(0.01)
            cam.stop_recording()
            print("Done.")
    finally:
        cam.close()


if __name__ == "__main__":
    main()
