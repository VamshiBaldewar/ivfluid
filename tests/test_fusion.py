from app.schema import DripRateResult, FluidLevelResult, PatientResult
from models.eta.infer import predict_eta
from models.fusion.risk import fuse


def _fluid(pct):
    return FluidLevelResult(status="ready", level_pct=pct, confidence=0.9)


def _drip(dpm):
    return DripRateResult(status="ready", drops_per_min=dpm, confidence=0.9)


def _patient(score):
    return PatientResult(status="ready", distress_score=score, face_detected=True)


def test_unknown_when_all_modules_unavailable():
    risk = fuse(
        FluidLevelResult(status="unavailable"),
        DripRateResult(status="unavailable"),
        PatientResult(status="unavailable"),
    )
    assert risk.tier == "UNKNOWN"
    assert "fluid_level" in risk.missing_inputs


def test_critical_low_bag():
    risk = fuse(_fluid(8), _drip(20), _patient(0.1))
    assert risk.tier == "CRITICAL"


def test_drip_stopped_with_low_bag_is_critical():
    risk = fuse(_fluid(15), _drip(0), _patient(0.1))
    assert risk.tier == "CRITICAL"


def test_high_distress_without_iv_signal():
    risk = fuse(
        FluidLevelResult(status="unavailable"),
        DripRateResult(status="unavailable"),
        _patient(0.85),
    )
    assert risk.tier == "CRITICAL"
    assert "fluid_level" in risk.missing_inputs


def test_moderate_level():
    risk = fuse(_fluid(35), _drip(18), _patient(0.1))
    assert risk.tier == "MODERATE"


def test_low_when_healthy():
    risk = fuse(_fluid(80), _drip(20), _patient(0.1))
    assert risk.tier == "LOW"


def test_incomplete_inputs_cannot_confirm_low_risk():
    risk = fuse(
        _fluid(100),
        _drip(20),
        PatientResult(status="unavailable"),
    )
    assert risk.tier == "UNKNOWN"
    assert "patient_distress" in risk.missing_inputs


def test_eta_unavailable_without_weights():
    eta = predict_eta([{"timestamp": 1, "level_pct": 50, "drops_per_min": 20}])
    assert eta.status == "unavailable"
    assert eta.minutes_remaining is None
