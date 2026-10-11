# Architecture & Workflow — AI-Driven Vision-Based IV Fluid Monitoring System

> Companion to PROJECT_CONTEXT.md. That file has scope/constraints/status; this file has how the system is built and how data moves through it. Read both before writing code.

## 1. System diagram

```
                ┌─────────────────────┐
                │   Camera(s) Input    │
                │ (IV bag + patient)   │
                └──────────┬───────────┘
                           │ raw video frames
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌────────────┐ ┌────────────┐ ┌──────────────┐
     │ Fluid-Level│ │ Drip-Rate  │ │   Patient    │
     │  CNN Model │ │ YOLO+Track │ │  Distress    │
     │  (Module1) │ │ (Module 2) │ │  Model (M3)  │
     └──────┬─────┘ └──────┬─────┘ └──────┬───────┘
            │ level%        │ drops/min     │ distress score
            └───────┬───────┴───────┬───────┘
                     ▼               ▼
              ┌───────────────────────────┐
              │   Fusion & Risk Scoring    │
              │        (Module 4)         │
              │  combines M1+M2+M3 into   │
              │  one alert tier           │
              └─────────────┬─────────────┘
                             │
                     ┌───────┴────────┐
                     ▼                ▼
           ┌──────────────────┐ ┌───────────────┐
           │  Predictive ETA   │ │  Dashboard /  │
           │   Model (M5)      │ │  Alerting(M6) │
           └──────────────────┘ └───────────────┘
```

## 2. Component breakdown

| Module | Input | Output | Notes |
|---|---|---|---|
| M1 Fluid-Level CNN | video frame (bag ROI) | level class/% | retrain-able per bag type |
| M2 Drip-Rate YOLO+Tracker | video frames (drip chamber ROI) | drops/min | tracker persists drop IDs across frames |
| M3 Patient Distress | video frame (patient ROI) | distress score/flag | facial or motion-based, TBD in Section 6 of PROJECT_CONTEXT |
| M4 Fusion & Risk Scoring | M1+M2+M3 outputs | alert tier (low/med/high) | rules or a small learned classifier — decide once, don't re-litigate per session |
| M5 Predictive ETA | M1+M2 time series | est. minutes remaining | must be a trained model (see constraint in PROJECT_CONTEXT §5) |
| M6 Dashboard/Alerting | M4+M5 outputs | UI + push notification | no physical buzzer |

## 3. Data flow (per frame/cycle)

1. Camera captures frame(s).
2. ROI extraction splits frame into bag region + patient region (fixed crop or detector-based).
3. M1 and M2 run on bag ROI; M3 runs on patient ROI — independent, parallel.
4. Outputs pushed to a shared state object (in-memory or lightweight queue).
5. M4 reads latest state, computes risk tier.
6. M5 reads level+rate history (rolling window), predicts ETA.
7. M6 reads M4+M5, updates dashboard, fires alert if tier crosses threshold.
8. Loop repeats at fixed interval (e.g., every 1–2 sec, matching inference cost).

## 4. Repo structure

```
/data/            raw + processed video/frames, per module
/models/
  fluid_level/    training script, saved weights, inference wrapper
  drip_rate/      YOLO config, weights, tracker wrapper
  patient/        distress model, weights, inference wrapper
  fusion/         risk-scoring logic
  eta/            ETA model, training + inference
/app/
  capture.py      camera capture loop
  pipeline.py      orchestrates M1-M5 per cycle
  dashboard/       FastAPI/Streamlit app (M6)
/tests/           per-module unit tests
PROJECT_CONTEXT.md
ARCHITECTURE.md
```

## 5. Build workflow (order matters — don't jump ahead)

1. Build M1 and M2 independently first, validate each on held-out video before touching fusion.
2. Build M3 independently, same validation discipline.
3. Only after M1–M3 individually work: build M4 fusion logic against real outputs, not mocked ones.
4. Build M5 once M1+M2 are producing a stable time series to train against.
5. Build M6 last — it only consumes outputs, shouldn't need module internals.
6. After each module, update PROJECT_CONTEXT.md §7 (status) before moving to the next.

## 6. Runtime workflow (once built)

`capture.py` → `pipeline.py` (runs M1–M5 per cycle) → shared state → `dashboard/` polls or subscribes to state → renders + alerts.

Keep modules callable independently (own function/class, own inputs/outputs) so any one can be swapped or retrained without touching the others.
