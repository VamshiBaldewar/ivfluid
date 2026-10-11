"""Shared payload shapes between capture, modules, pipeline, and dashboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RiskTier = Literal["LOW", "MODERATE", "HIGH", "CRITICAL", "UNKNOWN"]
ModuleStatus = Literal["ready", "unavailable", "error"]


@dataclass
class FluidLevelResult:
    status: ModuleStatus
    level_pct: float | None = None
    label: str | None = None
    confidence: float | None = None
    detail: str = ""


@dataclass
class DripRateResult:
    status: ModuleStatus
    drops_per_min: float | None = None
    drops_in_window: int | None = None
    confidence: float | None = None
    detail: str = ""


@dataclass
class PatientResult:
    status: ModuleStatus
    distress_score: float | None = None
    face_detected: bool = False
    expression: str | None = None
    detail: str = ""


@dataclass
class RiskResult:
    tier: RiskTier
    score: float
    reasons: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)


@dataclass
class EtaResult:
    status: ModuleStatus
    minutes_remaining: float | None = None
    detail: str = ""


@dataclass
class PipelineSnapshot:
    timestamp: float
    frame_jpeg: bytes | None
    camera_ok: bool
    fluid: FluidLevelResult
    drip: DripRateResult
    patient: PatientResult
    risk: RiskResult
    eta: EtaResult
    fps: float
    recording: dict[str, Any] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        """JSON-safe view without the JPEG blob."""
        return {
            "timestamp": self.timestamp,
            "camera_ok": self.camera_ok,
            "fps": round(self.fps, 2),
            "fluid": asdict(self.fluid),
            "drip": asdict(self.drip),
            "patient": asdict(self.patient),
            "risk": asdict(self.risk),
            "eta": asdict(self.eta),
            "recording": self.recording,
            "alerts": self.alerts,
        }
