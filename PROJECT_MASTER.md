# AI-Driven Vision-Based IV Fluid Monitoring System — Master Document

> Give this single file to any coding agent (Cursor, Antigravity, Codex) at the start of every session. Read it fully before writing code. Section 6 is hard constraints — never violate them. Section 10 is the living status — update it before ending any session so the next tool/session doesn't guess or hallucinate.

## 1. One-line pitch
Single-camera AI system that monitors IV fluid level, drip rate, and patient distress together, fusing them into one risk signal — replacing load-cell/microcontroller hardware with vision + ML.

## 2. Problem
Manual IV monitoring is error-prone and understaffed. Existing published/patented solutions handle fluid-level detection OR drip-rate counting OR patient monitoring in isolation — never fused — and don't generalize across bag types/lighting.

## 3. Scope
**IS:** single camera, CNN fluid-level model, YOLO drip counting, patient-distress model, fusion layer, predictive ETA, web/mobile dashboard.
**IS NOT:** load cell, HX711, ESP8266/ESP32-CAM, physical buzzer/LCD, Blynk/ThingSpeak cloud.

## 4. Modules (build in this order)
1. **Fluid-level estimation** — CNN on video frames of the IV bag → level class/%
2. **Drip-rate counting** — YOLO + tracker on drip chamber → drops/min
3. **Patient distress monitoring** — pretrained facial/pain model, minimal fine-tuning
4. **Fusion & risk scoring** — combines 1–3 into one alert tier (not independent alerts)
5. **Predictive ETA** — trained model estimating time-to-empty from noisy vision data (must be learned, not volume÷rate arithmetic)
6. **Dashboard/alerting** — web UI + notifications, no physical hardware

## 5. System diagram

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
     │  CNN (M1)  │ │ YOLO+Track │ │  Distress    │
     │            │ │    (M2)    │ │  Model (M3)  │
     └──────┬─────┘ └──────┬─────┘ └──────┬───────┘
            │ level%        │ drops/min     │ distress score
            └───────┬───────┴───────┬───────┘
                     ▼               ▼
              ┌───────────────────────────┐
              │   Fusion & Risk Scoring    │
              │           (M4)             │
              └────────────┬──────────────┘
                             │
                     ┌───────┴────────┐
                     ▼                ▼
           ┌──────────────────┐ ┌───────────────┐
           │ Predictive ETA(M5)│ │ Dashboard(M6) │
           └──────────────────┘ └───────────────┘
```

## 6. Non-negotiable constraints
- NO load cell, HX711, ESP8266/ESP32-CAM, physical buzzer, or Blynk/ThingSpeak cloud.
- Patient monitoring and IV monitoring must feed one fusion layer, not separate dashboards.
- ETA prediction must be a trained model, not a plain formula.
- If a decision isn't resolved in Section 9, ask — don't assume and proceed.

## 7. Tech stack (locked)
Python, OpenCV, PyTorch (pick one framework and don't switch mid-build), YOLOv8/v5, FastAPI or Streamlit for the dashboard. Prototype target: laptop webcam, no embedded hardware.

**Locked this repo:** PyTorch (when training starts), YOLOv8, FastAPI dashboard.

## 8. Dataset plan
| Module | Data source | Status |
|---|---|---|
| M1 Fluid-level | Mendeley SESOVERA saline bottles (CC BY 4.0); 2,479 local images across 4 classes (0: 1,034; 50: 593; 80: 304; 100: 548). The full 4,217-image fetch was interrupted by DNS/network errors. | MobileNetV3-Small checkpoint trained for 10 epochs; 100% accuracy on a random 495-image holdout. One real local sample per class also inferred correctly. Same-source bottle images may leak scene cues across the random split; this does not establish IV-bag or camera-rig accuracy. |
| M2 Drip-rate | Generated detector dataset plus three generated videos labeled 10, 20, and 30 dpm. | YOLO checkpoint trained; detector mAP50 0.9928 on synthetic data. Replay rate MAE 0.533 dpm on the three generated clips. Real rig footage remains required. |
| M3 Patient distress | FER2013 grayscale facial-expression dataset; not pain-labeled or IV-specific. | FER2013 EmotionCNN checkpoint wired into inference and fine-tuned for one additional epoch. Test accuracy: 66.75% (7 classes); derived distress accuracy: 83.42%, AUROC: 0.9193. The binary target is an expression-derived proxy, not clinical distress or pain. |

## 9. Open decisions (resolve here, don't leave silently assumed)
- Distress-detection approach: **Phase-I — FER2013 EmotionCNN expression model.** Haar face detection + 48x48 grayscale inference with horizontal-flip test-time averaging. This estimates expression-derived distress, not pain or clinical distress.
- Fusion logic: **Phase-I default — rule-based thresholds** in `models/fusion/risk.py`. A learned classifier is deferred until M1–M3 produce labeled fused outputs. Rules are a prototype, not a clinical protocol.
- ETA model architecture: **Phase-I default — sequence model (LSTM) trained later** on labeled bag-run time series. No ETA checkpoint or training sequence labels are present. **Do not substitute volume÷rate arithmetic.**

## 10. Current status (update every session)
- [x] M1 fluid-level CNN trained — 2,479 available bottle images; 100% random-split validation accuracy, not evidence of IV-bag generalization. Full download failed intermittently.
- [x] M2 detector/tracker built and replayed — 0.533 dpm MAE on three generated clips; no real rig video validation
- [x] M3 FER2013 expression checkpoint selected, wired, and fine-tuned — 66.75% 7-class test accuracy and 83.42% derived distress accuracy. Only the derived binary proxy exceeds 80%; this is not clinically validated distress accuracy.
- [x] M4 fusion logic built (rule-based prototype); missing required inputs now prevent a LOW result. Unit tests pass; a live camera frame reached fusion, but M2/M3 had no detections, so full real-output validation remains outstanding.
- [ ] M5 ETA model built (interface + time-series logger only; no trained weights)
- [x] M6 dashboard built (FastAPI; live frame + fused risk + dataset recorder)
- [x] Capture loop, pipeline, shared state, dataset collection paths
- Last updated: 2026-10-11
- Known issues / deviations from plan: M1 validation is limited to public saline bottles and a random same-source split; M2 clips are generated. Neither proves real-world accuracy. M3 uses expression as a distress proxy, not pain. A one-frame camera-to-dashboard pass succeeded, but M2/M3 were unavailable on that frame. M4 still needs a run with all models detecting their targets. M5 is blocked because no labeled bag-run time series exists. The remaining public full M1 download hit network/DNS failures; self-recorded camera/lighting data and clinical validation remain outstanding.

## 11. Repo structure
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
  pipeline.py     orchestrates M1–M5 per cycle
  dashboard/      FastAPI app (M6)
/tests/           per-module unit tests
PROJECT_MASTER.md
```

## 12. Build workflow (order matters)
1. Build M1 and M2 independently, validate each on held-out video before touching fusion.
2. Build M3 independently, same discipline.
3. Build M4 fusion logic against real M1–M3 outputs, not mocked ones.
4. Build M5 once M1+M2 produce a stable time series to train against.
5. Build M6 last — it only consumes outputs.
6. Update Section 10 after each module before moving to the next.

## 13. Runtime workflow
`capture.py` → `pipeline.py` (runs M1–M5 per cycle) → shared state → `dashboard/` polls/subscribes → renders + alerts. Keep every module independently callable so any one can be swapped or retrained without touching the others.

## 14. Team & context
Batch D3, team of 4, guide Mrs. N Divya, GNITS CSE (Data Science), Phase-I GNR-22 (2026–2027).

## 15. How to run (until models exist)
```
pip install -r requirements.txt
python -m app.dashboard.server
```
Open http://127.0.0.1:8000 — Start pipeline (webcam `0` or a video file path).

Labeled collection (same folders as the dashboard recorder):
```
python -m app.collect --module drip_rate --label 20dpm --seconds 60
python -m app.collect --module fluid_level --label 50 --frames 40 --interval 1
```
Drip clips go to `data/raw/drip_rate/<label>/`. Bag stills go to `data/raw/fluid_level/<label>/`. Time series for later ETA training logs to `data/timeseries/vision_log.csv`.

