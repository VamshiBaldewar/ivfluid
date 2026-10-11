"""M2 drip-rate inference: YOLOv8 drop detector + lightweight centroid tracker.

The pipeline calls `infer_drip_rate(frame)` once per frame on a background thread.
Because drops/min is temporal, a module-level stateful tracker accumulates drop
events across calls. Each newly-appearing drop in the upper chamber region is
counted once; the rate is derived from inter-drop intervals (median, robust).

Weights: models/drip_rate/weights/drip_yolo.pt (trained by models.drip_rate.train).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from app.schema import DripRateResult

WEIGHTS = Path(__file__).resolve().parent / "weights" / "drip_yolo.pt"
CONF = 0.35
ENTRY_FRAC = 0.35  # a new drop must appear in the top 35% of the frame to count
GATE = 0.25  # normalized diagonal gate for prediction-based association
STALE_TRACK = 6  # frames a track survives without a detection
DPM_WINDOW = 60.0  # seconds of events used for the rate window
REFRACTORY = 0.2  # min seconds between counted drops (dedupes slow-fps echoes)

_LOCK = threading.Lock()
_MODEL = None
_LOAD_ERROR: str | None = None
_TRACKER: "_DropTracker | None" = None


class _DropTracker:
    def __init__(self) -> None:
        self.events: deque[float] = deque(maxlen=240)
        self.tracks: dict[int, dict] = {}
        self._next_id = 1
        self.started: float | None = None
        self.last_count: float | None = None

    def reset(self) -> None:
        self.events.clear()
        self.tracks.clear()
        self._next_id = 1
        self.started = None
        self.last_count = None

    def _predict(self, now: float) -> dict[int, tuple[float, float]]:
        eff: dict[int, tuple[float, float]] = {}
        for tid, tr in self.tracks.items():
            dt = max(0.0, now - tr["t"])
            eff[tid] = (tr["x"] + tr["vx"] * dt, tr["y"] + tr["vy"] * dt)
        return eff

    def update(self, dets: list[tuple[float, float, float]], now: float) -> None:
        if self.started is None:
            self.started = now

        eff = self._predict(now)
        pairs = []
        for di, (x, y, _c) in enumerate(dets):
            for tid, (px, py) in eff.items():
                dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                if dist <= GATE:
                    pairs.append((dist, di, tid))
        pairs.sort(key=lambda p: p[0])

        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()
        for _dist, di, tid in pairs:
            if di in matched_dets or tid in matched_tracks:
                continue
            x, y, c = dets[di]
            tr = self.tracks[tid]
            dt = max(1e-2, now - tr["t"])
            tr["vx"] = 0.5 * tr["vx"] + 0.5 * (x - tr["x"]) / dt
            tr["vy"] = 0.5 * tr["vy"] + 0.5 * (y - tr["y"]) / dt
            tr.update({"x": x, "y": y, "c": c, "t": now, "miss": 0})
            matched_dets.add(di)
            matched_tracks.add(tid)

        for di, (x, y, c) in enumerate(dets):
            if di in matched_dets:
                continue
            self.tracks[self._next_id] = {"x": x, "y": y, "vx": 0.0, "vy": 0.0, "c": c, "t": now, "miss": 0}
            if y <= ENTRY_FRAC and (self.last_count is None or now - self.last_count >= REFRACTORY):
                self.events.append(now)
                self.last_count = now
            self._next_id += 1

        for tid in list(self.tracks):
            if tid in matched_tracks:
                continue
            self.tracks[tid]["miss"] += 1
            if self.tracks[tid]["miss"] > STALE_TRACK:
                del self.tracks[tid]

    def rate(self, now: float) -> float | None:
        recent = [t for t in self.events if now - t <= DPM_WINDOW]
        if len(recent) >= 3:
            intervals = sorted(b - a for a, b in zip(recent, recent[1:]) if b - a > 1e-3)
            if intervals:
                med = intervals[len(intervals) // 2]
                return min(600.0, 60.0 / med)
        if len(recent) >= 1:
            return None
        if self.started is not None and now - self.started > 5.0:
            last = self.last_count if self.last_count is not None else self.started
            if now - last > 3.0:
                return 0.0
        return None


def _load():
    global _MODEL, _LOAD_ERROR
    if _MODEL is not None or _LOAD_ERROR is not None:
        return
    if not WEIGHTS.exists():
        _LOAD_ERROR = "YOLO weights not trained yet - run tools.gen_drip_dataset + models.drip_rate.train"
        return
    try:
        from ultralytics import YOLO

        _MODEL = YOLO(str(WEIGHTS))
        _MODEL.fuse()
    except Exception as exc:  # pragma: no cover - defensive
        _LOAD_ERROR = f"failed to load M2 weights: {exc}"


def reset_tracker() -> None:
    global _TRACKER
    with _LOCK:
        if _TRACKER is not None:
            _TRACKER.reset()


def _detect(frame: np.ndarray) -> list[tuple[float, float, float]]:
    res = _MODEL.predict(frame, conf=CONF, imgsz=416, verbose=False)[0]
    h, w = frame.shape[:2]
    dets: list[tuple[float, float, float]] = []
    boxes = getattr(res, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return dets
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    for (x0, y0, x1, y1), c in zip(xyxy, confs):
        cx = ((x0 + x1) / 2) / max(1, w)
        cy = ((y0 + y1) / 2) / max(1, h)
        dets.append((cx, cy, float(c)))
    return dets


def infer_drip_rate(frame: np.ndarray | None, ts: float | None = None) -> DripRateResult:
    global _TRACKER

    if frame is None or frame.size == 0:
        return DripRateResult(status="error", detail="empty frame")

    _load()
    if _MODEL is None:
        return DripRateResult(status="unavailable", detail=_LOAD_ERROR or "M2 unavailable")

    with _LOCK:
        if _TRACKER is None:
            _TRACKER = _DropTracker()
        now = ts if ts is not None else time.time()
        dets_frame_shape = (float(frame.shape[0]), float(frame.shape[1]))
        try:
            dets = _detect(frame)
        except Exception as exc:  # pragma: no cover - defensive
            return DripRateResult(status="error", detail=f"M2 inference failed: {exc}")
        _TRACKER.update(dets, now)
        dpm = _TRACKER.rate(now)
        recent = sum(1 for t in _TRACKER.events if now - t <= DPM_WINDOW)
        conf = float(np.mean([d[2] for d in dets])) if dets else None

    if dpm is None:
        return DripRateResult(
            status="unavailable",
            drops_in_window=recent,
            confidence=conf,
            detail="accumulating drop events for rate estimate",
        )
    return DripRateResult(
        status="ready",
        drops_per_min=round(dpm, 1),
        drops_in_window=recent,
        confidence=round(conf, 3) if conf is not None else None,
        detail=f"{dpm:.1f} drops/min from {len(_TRACKER.tracks)} tracked drops",
    )
