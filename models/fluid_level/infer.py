"""M1 fluid-level inference wrapper (trained MobileNetV3-Small).

Loads weights produced by `python -m models.fluid_level.train`. Keeps the same
`infer_fluid_level(frame)` signature the pipeline expects.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from app.schema import FluidLevelResult

WEIGHTS = Path(__file__).resolve().parent / "weights" / "fluid_level.pt"

_LOCK = threading.Lock()
_MODEL = None
_META: dict | None = None
_LOAD_ERROR: str | None = None


def _build_model(n_classes: int) -> nn.Module:
    net = models.mobilenet_v3_small(weights=None)
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, n_classes)
    return net


def _load():
    global _MODEL, _META, _LOAD_ERROR
    if _MODEL is not None or _LOAD_ERROR is not None:
        return
    with _LOCK:
        if _MODEL is not None or _LOAD_ERROR is not None:
            return
        if not WEIGHTS.exists():
            _LOAD_ERROR = "CNN weights not trained yet - run python -m models.fluid_level.train"
            return
        try:
            ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
            classes = ckpt["classes"]
            model = _build_model(len(classes))
            state = {k.replace("backbone.", "", 1): v for k, v in ckpt["state_dict"].items()}
            model.load_state_dict(state)
            model.eval()
            _MODEL = model
            _META = ckpt
        except Exception as exc:  # pragma: no cover - defensive
            _LOAD_ERROR = f"failed to load M1 weights: {exc}"


def _preprocess(frame: np.ndarray, img_size: int):
    import cv2
    from PIL import Image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return tf(img).unsqueeze(0)


def infer_fluid_level(frame: np.ndarray | None) -> FluidLevelResult:
    if frame is None or frame.size == 0:
        return FluidLevelResult(status="error", detail="empty frame")

    _load()
    if _MODEL is None:
        return FluidLevelResult(status="unavailable", detail=_LOAD_ERROR or "M1 unavailable")

    meta = _META or {}
    classes: list[int] = list(meta.get("classes", []))
    img_size = int(meta.get("img_size", 224))
    try:
        x = _preprocess(frame, img_size)
        with torch.no_grad():
            probs = torch.softmax(_MODEL(x), dim=1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
        label = int(classes[idx]) if idx < len(classes) else idx
    except Exception as exc:  # pragma: no cover - defensive
        return FluidLevelResult(status="error", detail=f"M1 inference failed: {exc}")

    return FluidLevelResult(
        status="ready",
        level_pct=float(label),
        label=f"{label}%",
        confidence=round(conf, 4),
        detail=f"CNN level={label}% (conf {conf:.2f})",
    )
