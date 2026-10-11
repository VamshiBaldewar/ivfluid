"""Orchestrates capture → M1–M5 → shared state. Modules stay independently callable."""

from __future__ import annotations

import csv
import threading
import time
from app.capture import FrameSource, encode_jpeg, save_frame
from app.paths import DATA_RAW, DATA_TIMESERIES, PROJECT_ROOT
from app.schema import PipelineSnapshot
from app.state import STATE
from models.drip_rate.infer import infer_drip_rate
from models.eta.infer import predict_eta
from models.fluid_level.infer import infer_fluid_level
from models.fusion.risk import fuse
from models.patient.infer import infer_patient

TIMESERIES_CSV = DATA_TIMESERIES / "vision_log.csv"


class Pipeline:
    def __init__(self, source: int | str = 0) -> None:
        self.source = source
        self.camera = FrameSource(source)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._history: list[dict] = []
        self._record_meta: dict = {
            "active": False,
            "kind": None,  # "video" | "frames"
            "module": None,
            "label": None,
            "path": None,
            "frames_saved": 0,
        }
        self._frame_interval = 0.0
        self._next_frame_save = 0.0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="iv-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.camera.close()

    def start_dataset_video(self, module: str, label: str) -> str:
        module = _safe_token(module)
        label = _safe_token(label)
        dest_dir = DATA_RAW / module / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = dest_dir / f"{stamp}.mp4"
        if not self.camera.start_recording(path):
            raise RuntimeError("Could not start video writer (is the camera open?)")
        self._record_meta = {
            "active": True,
            "kind": "video",
            "module": module,
            "label": label,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "frames_saved": 0,
        }
        return str(path)

    def start_dataset_frames(self, module: str, label: str, interval_s: float = 1.0) -> str:
        module = _safe_token(module)
        label = _safe_token(label)
        dest_dir = DATA_RAW / module / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        self.camera.stop_recording()
        self._frame_interval = max(0.2, interval_s)
        self._next_frame_save = 0.0
        self._record_meta = {
            "active": True,
            "kind": "frames",
            "module": module,
            "label": label,
            "path": str(dest_dir.relative_to(PROJECT_ROOT)),
            "frames_saved": 0,
        }
        return str(dest_dir)

    def stop_dataset(self) -> dict:
        self.camera.stop_recording()
        meta = dict(self._record_meta)
        self._record_meta = {
            "active": False,
            "kind": None,
            "module": None,
            "label": None,
            "path": None,
            "frames_saved": meta.get("frames_saved", 0),
        }
        return meta

    def _loop(self) -> None:
        opened = self.camera.open()
        last = time.perf_counter()
        fps = 0.0
        while not self._stop.is_set():
            frame = self.camera.read() if opened else None
            now = time.time()
            t1 = time.perf_counter()
            dt = t1 - last
            last = t1
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            if frame is None:
                fluid = infer_fluid_level(None)
                drip = infer_drip_rate(None)
                patient = infer_patient(None)
                jpeg = None
            else:
                fluid = infer_fluid_level(frame)
                drip = infer_drip_rate(frame)
                patient = infer_patient(frame)
                jpeg = encode_jpeg(frame)
                self._maybe_save_frame(frame, now)

            risk = fuse(fluid, drip, patient)
            row = {
                "timestamp": now,
                "level_pct": fluid.level_pct,
                "drops_per_min": drip.drops_per_min,
                "distress_score": patient.distress_score,
                "risk_tier": risk.tier,
            }
            self._history.append(row)
            if len(self._history) > 2000:
                self._history = self._history[-1500:]
            _append_timeseries(row)
            eta = predict_eta(self._history)

            STATE.update(
                PipelineSnapshot(
                    timestamp=now,
                    frame_jpeg=jpeg,
                    camera_ok=frame is not None,
                    fluid=fluid,
                    drip=drip,
                    patient=patient,
                    risk=risk,
                    eta=eta,
                    fps=fps,
                    recording=dict(self._record_meta),
                )
            )
            time.sleep(0.03)

        self.camera.close()

    def _maybe_save_frame(self, frame, now: float) -> None:
        meta = self._record_meta
        if not meta.get("active") or meta.get("kind") != "frames":
            return
        if now < self._next_frame_save:
            return
        module = meta["module"]
        label = meta["label"]
        dest = DATA_RAW / module / label
        stamp = time.strftime("%Y%m%d_%H%M%S")
        n = int(meta.get("frames_saved", 0)) + 1
        save_frame(frame, dest / f"{stamp}_{n:04d}.jpg")
        meta["frames_saved"] = n
        self._next_frame_save = now + self._frame_interval


def _safe_token(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    return cleaned or "unlabeled"


def _append_timeseries(row: dict) -> None:
    TIMESERIES_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not TIMESERIES_CSV.exists()
    with TIMESERIES_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


PIPELINE = Pipeline()
