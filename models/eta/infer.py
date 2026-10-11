"""M5 ETA inference. Must remain a trained model — never volume÷rate."""

from __future__ import annotations

from pathlib import Path

from app.schema import EtaResult

WEIGHTS = Path(__file__).resolve().parent / "weights" / "eta_lstm.pt"


def predict_eta(history_rows: list[dict]) -> EtaResult:
    """history_rows: dicts with timestamp, level_pct, drops_per_min (may be None)."""
    if not WEIGHTS.exists():
        return EtaResult(
            status="unavailable",
            minutes_remaining=None,
            detail="ETA model not trained — logging time series to data/timeseries for later LSTM training",
        )
    if len(history_rows) < 8:
        return EtaResult(
            status="unavailable",
            detail="not enough logged samples for the sequence model",
        )
    return EtaResult(
        status="error",
        detail="weights file present but ETA loader not implemented",
    )
