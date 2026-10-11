"""M4 fusion — one risk signal from fluid, drip, and distress.

Phase-I: rule-based. Not a clinical protocol. Re-validate when M1–M3 are live.
"""

from __future__ import annotations

from app.schema import DripRateResult, FluidLevelResult, PatientResult, RiskResult

# Prototype thresholds (document in PROJECT_MASTER §9).
CRITICAL_LEVEL = 10.0
HIGH_LEVEL = 20.0
MODERATE_LEVEL = 40.0
CRITICAL_DISTRESS = 0.80
HIGH_DISTRESS = 0.60
MODERATE_DISTRESS = 0.40
DRIP_STOPPED = 0.5  # drops/min treated as stopped


def fuse(
    fluid: FluidLevelResult,
    drip: DripRateResult,
    patient: PatientResult,
) -> RiskResult:
    reasons: list[str] = []
    missing: list[str] = []
    score = 0.0

    level = fluid.level_pct if fluid.status == "ready" else None
    dpm = drip.drops_per_min if drip.status == "ready" else None
    distress = patient.distress_score if patient.status == "ready" else None

    if level is None:
        missing.append("fluid_level")
    if dpm is None:
        missing.append("drip_rate")
    if distress is None:
        missing.append("patient_distress")

    if level is None and dpm is None and distress is None:
        return RiskResult(
            tier="UNKNOWN",
            score=0.0,
            reasons=["No vision modules are producing values yet."],
            missing_inputs=missing,
        )

    if level is not None:
        if level < CRITICAL_LEVEL:
            score = max(score, 1.00)
            reasons.append(f"Bag level critically low ({level:.0f}%).")
        elif level < HIGH_LEVEL:
            score = max(score, 0.75)
            reasons.append(f"Bag level low ({level:.0f}%).")
        elif level < MODERATE_LEVEL:
            score = max(score, 0.45)
            reasons.append(f"Bag level moderate ({level:.0f}%).")

    if distress is not None:
        if distress >= CRITICAL_DISTRESS:
            score = max(score, 1.00)
            reasons.append(f"Patient distress high ({distress:.2f}).")
        elif distress >= HIGH_DISTRESS:
            score = max(score, 0.75)
            reasons.append(f"Patient distress elevated ({distress:.2f}).")
        elif distress >= MODERATE_DISTRESS:
            score = max(score, 0.45)
            reasons.append(f"Patient distress mild ({distress:.2f}).")

    if dpm is not None and dpm <= DRIP_STOPPED:
        if level is not None and level < HIGH_LEVEL:
            score = max(score, 1.00)
            reasons.append("Drip stopped while bag is already low.")
        else:
            score = max(score, 0.75)
            reasons.append("Drip appears stopped.")

    if not reasons:
        reasons.append("Observed signals within prototype low-risk bands.")
        score = max(score, 0.10)

    if score >= 0.90:
        tier = "CRITICAL"
    elif score >= 0.70:
        tier = "HIGH"
    elif score >= 0.40:
        tier = "MODERATE"
    else:
        tier = "LOW"

    if missing:
        reasons.append(
            "Incomplete inputs: " + ", ".join(missing) + " (tier uses available signals only)."
        )
        if tier == "LOW":
            tier = "UNKNOWN"
            reasons.append("Cannot confirm low risk while required inputs are missing.")

    return RiskResult(
        tier=tier,
        score=round(score, 3),
        reasons=reasons,
        missing_inputs=missing,
    )
