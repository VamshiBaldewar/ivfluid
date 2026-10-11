import numpy as np
import pytest

import models.drip_rate.infer as drip_infer
import models.fluid_level.infer as fluid_infer
import models.patient.infer as patient_infer


def test_wrappers_unavailable_without_weights(monkeypatch, tmp_path):
    for module in (fluid_infer, drip_infer, patient_infer):
        monkeypatch.setattr(module, "WEIGHTS", tmp_path / f"{module.__name__}.pt")
        monkeypatch.setattr(module, "_MODEL", None)
        monkeypatch.setattr(module, "_LOAD_ERROR", None)
    monkeypatch.setattr(drip_infer, "_TRACKER", None)

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    assert fluid_infer.infer_fluid_level(frame).status == "unavailable"
    assert drip_infer.infer_drip_rate(frame).status == "unavailable"
    assert patient_infer.infer_patient(frame).status == "unavailable"


def test_patient_inference_uses_fer_checkpoint(monkeypatch):
    if not patient_infer.WEIGHTS.exists():
        pytest.skip("FER checkpoint is not present")

    class DetectedFace:
        def detectMultiScale(self, gray, **_kwargs):
            return np.array([[0, 0, gray.shape[1], gray.shape[0]]])

    monkeypatch.setattr(patient_infer, "_MODEL", None)
    monkeypatch.setattr(patient_infer, "_LOAD_ERROR", None)
    monkeypatch.setattr(patient_infer, "_FACE_CASCADE", DetectedFace())
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    result = patient_infer.infer_patient(frame)

    assert result.status == "ready"
    assert result.face_detected is True
    assert result.expression in patient_infer.CLASSES
    assert 0.0 <= result.distress_score <= 1.0


def test_wrappers_error_on_empty():
    assert fluid_infer.infer_fluid_level(None).status == "error"
