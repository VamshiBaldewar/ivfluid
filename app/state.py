"""Thread-safe pipeline snapshot + alert ring buffer."""

from __future__ import annotations

import threading
from collections import deque
from time import time

from app.schema import (
    DripRateResult,
    EtaResult,
    FluidLevelResult,
    PatientResult,
    PipelineSnapshot,
    RiskResult,
)


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = PipelineSnapshot(
            timestamp=time(),
            frame_jpeg=None,
            camera_ok=False,
            fluid=FluidLevelResult(status="unavailable", detail="pipeline not started"),
            drip=DripRateResult(status="unavailable", detail="pipeline not started"),
            patient=PatientResult(status="unavailable", detail="pipeline not started"),
            risk=RiskResult(tier="UNKNOWN", score=0.0, reasons=["Idle"]),
            eta=EtaResult(status="unavailable", detail="pipeline not started"),
            fps=0.0,
            recording={"active": False, "module": None, "label": None, "path": None},
            alerts=[],
        )
        self._alerts: deque[dict] = deque(maxlen=40)
        self._last_tier = "UNKNOWN"

    def update(self, snapshot: PipelineSnapshot) -> None:
        with self._lock:
            if snapshot.risk.tier != self._last_tier and snapshot.risk.tier != "UNKNOWN":
                self._alerts.appendleft(
                    {
                        "t": snapshot.timestamp,
                        "tier": snapshot.risk.tier,
                        "reasons": list(snapshot.risk.reasons),
                    }
                )
                self._last_tier = snapshot.risk.tier
            snapshot.alerts = list(self._alerts)
            self._snapshot = snapshot

    def get(self) -> PipelineSnapshot:
        with self._lock:
            return self._snapshot


STATE = SharedState()
