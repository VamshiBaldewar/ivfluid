"""Laptop webcam / video-file capture. Independently usable for dataset recording."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FrameSource:
    def __init__(self, source: int | str = 0, width: int = 1280, height: int = 720) -> None:
        self.source = source
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self.record_path: Path | None = None
        self._frame_timestamp: float | None = None

    def open(self) -> bool:
        self.close()
        cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            self._cap = None
            return False
        self._cap = cap
        return True

    @property
    def ok(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_timestamp(self) -> float | None:
        """Timestamp in a video file's timeline for the most recent frame.

        Live camera captures deliberately return ``None`` so their consumers
        continue to use wall-clock time. File timestamps make replayed videos
        independent of the machine's inference speed.
        """
        return self._frame_timestamp

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        self._frame_timestamp = None
        if isinstance(self.source, str):
            position_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            if position_ms >= 0:
                self._frame_timestamp = position_ms / 1000.0
        if self._writer is not None:
            self._writer.write(frame)
        return frame

    def start_recording(self, path: Path, fps: float = 20.0) -> bool:
        if self._cap is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width)
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if not writer.isOpened():
            return False
        self.stop_recording()
        self._writer = writer
        self.record_path = path
        return True

    def stop_recording(self) -> Path | None:
        path = self.record_path
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self.record_path = None
        return path

    def close(self) -> None:
        self.stop_recording()
        self._frame_timestamp = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return buf.tobytes()


def save_frame(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)
