import numpy as np

import app.pipeline as pipeline_module
from app.schema import PipelineSnapshot, FluidLevelResult, DripRateResult, PatientResult, RiskResult, EtaResult
from app.state import SharedState
from app.pipeline import Pipeline


def test_alert_logged_on_tier_change():
    state = SharedState()
    snap = PipelineSnapshot(
        timestamp=1.0,
        frame_jpeg=None,
        camera_ok=False,
        fluid=FluidLevelResult(status="unavailable"),
        drip=DripRateResult(status="unavailable"),
        patient=PatientResult(status="unavailable"),
        risk=RiskResult(tier="HIGH", score=0.8, reasons=["test"]),
        eta=EtaResult(status="unavailable"),
        fps=1.0,
    )
    state.update(snap)
    assert state.get().alerts[0]["tier"] == "HIGH"


def test_pipeline_opens_source_before_starting_worker(monkeypatch):
    class CameraStub:
        ok = False

        def __init__(self, _source):
            self.open_calls = 0

        def open(self):
            self.open_calls += 1
            self.ok = True
            return True

        def close(self):
            self.ok = False

    monkeypatch.setattr("app.pipeline.FrameSource", CameraStub)
    pipe = Pipeline("example.mp4")
    monkeypatch.setattr(pipe, "_loop", lambda: None)

    assert pipe.start() is True
    assert pipe.camera.open_calls == 1
    pipe.stop()


def test_pipeline_does_not_start_when_source_cannot_open(monkeypatch):
    class CameraStub:
        ok = False

        def __init__(self, _source):
            pass

        def open(self):
            return False

        def close(self):
            pass

    monkeypatch.setattr("app.pipeline.FrameSource", CameraStub)
    pipe = Pipeline("missing.mp4")

    assert pipe.start() is False
    assert pipe.running is False


def test_pipeline_uses_video_timestamp_for_drip_rate(monkeypatch):
    class CameraStub:
        ok = True
        frame_timestamp = 12.5

        def read(self):
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def close(self):
            self.ok = False

    pipe = Pipeline("demo.mp4")
    pipe.camera = CameraStub()
    seen_timestamps = []

    monkeypatch.setattr(
        pipeline_module,
        "infer_fluid_level",
        lambda _frame: FluidLevelResult(status="ready", level_pct=80, confidence=0.9),
    )

    def infer_drip(_frame, ts=None):
        seen_timestamps.append(ts)
        return DripRateResult(status="ready", drops_per_min=20, confidence=0.9)

    monkeypatch.setattr(pipeline_module, "infer_drip_rate", infer_drip)
    monkeypatch.setattr(
        pipeline_module,
        "infer_patient",
        lambda _frame: PatientResult(status="ready", distress_score=0.1, face_detected=True),
    )
    monkeypatch.setattr(
        pipeline_module,
        "fuse",
        lambda *_results: RiskResult(tier="LOW", score=0.1, reasons=["test"]),
    )
    monkeypatch.setattr(
        pipeline_module,
        "predict_eta",
        lambda _history: EtaResult(status="unavailable"),
    )
    monkeypatch.setattr(pipeline_module, "encode_jpeg", lambda _frame: b"jpeg")
    monkeypatch.setattr(pipeline_module, "_append_timeseries", lambda _row: None)
    monkeypatch.setattr(pipeline_module.time, "sleep", lambda _seconds: pipe._stop.set())

    pipe._loop()

    assert seen_timestamps == [12.5]
