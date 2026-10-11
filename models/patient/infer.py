"""M3 patient-distress inference wrapper.

Phase-I: pretrained FER-style model. Do not collect a custom pain dataset.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from app.schema import PatientResult
from models.patient.train import CLASSES, DISTRESS_IDX, EmotionCNN

WEIGHTS = Path(__file__).resolve().parent / "weights" / "emotion_fer.pt"
FACE_SIZE = 48

_LOCK = threading.Lock()
_MODEL: EmotionCNN | None = None
_LOAD_ERROR: str | None = None
_FACE_CASCADE = None


def _load() -> None:
    global _MODEL, _LOAD_ERROR
    if _MODEL is not None or _LOAD_ERROR is not None:
        return
    with _LOCK:
        if _MODEL is not None or _LOAD_ERROR is not None:
            return
        if not WEIGHTS.exists():
            _LOAD_ERROR = "FER weights not found - run python -m models.patient.train"
            return
        try:
            checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
            classes = checkpoint.get("classes", CLASSES)
            model = EmotionCNN(len(classes))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            _MODEL = model
        except Exception as exc:  # pragma: no cover - defensive
            _LOAD_ERROR = f"failed to load M3 weights: {exc}"


def _face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(str(cascade_path))
    return _FACE_CASCADE


def infer_patient(frame: np.ndarray | None) -> PatientResult:
    if frame is None or frame.size == 0:
        return PatientResult(status="error", detail="empty frame")

    _load()
    if _MODEL is None:
        return PatientResult(
            status="unavailable",
            face_detected=False,
            detail=_LOAD_ERROR or "M3 unavailable",
        )

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade().detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(32, 32),
        )
        if len(faces) == 0:
            return PatientResult(
                status="unavailable",
                face_detected=False,
                detail="M3 ready; no face detected",
            )

        x, y, width, height = max(faces, key=lambda rect: rect[2] * rect[3])
        face = cv2.resize(gray[y : y + height, x : x + width], (FACE_SIZE, FACE_SIZE))
        tensor = torch.from_numpy(face.astype(np.float32) / 255.0)
        tensor = (((tensor - 0.5) / 0.5).unsqueeze(0).unsqueeze(0))
        with torch.no_grad():
            probs = torch.softmax(_MODEL(tensor), dim=1)
            probs = (probs + torch.softmax(_MODEL(torch.flip(tensor, dims=[3])), dim=1)) / 2
        probabilities = probs[0]
        class_idx = int(probabilities.argmax().item())
        classes = list(CLASSES)
        expression = classes[class_idx] if class_idx < len(classes) else str(class_idx)
        distress_idx = [idx for idx in DISTRESS_IDX if idx < len(probabilities)]
        distress_score = float(probabilities[distress_idx].sum().item())
    except Exception as exc:  # pragma: no cover - defensive
        return PatientResult(status="error", detail=f"M3 inference failed: {exc}")

    return PatientResult(
        status="ready",
        distress_score=round(distress_score, 4),
        face_detected=True,
        expression=expression,
        detail=f"FER expression={expression} (distress {distress_score:.2f})",
    )
