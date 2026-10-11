from app.schema import PipelineSnapshot, FluidLevelResult, DripRateResult, PatientResult, RiskResult, EtaResult
from app.state import SharedState


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
