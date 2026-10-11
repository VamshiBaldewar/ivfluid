# Project Context — AI-Driven Vision-Based IV Fluid Monitoring System

> Read this file fully before writing any code. Section 5 is hard constraints — never contradict them, never re-add removed hardware. Section 7 is the living status — update it after every session so the next tool (or the next session) doesn't have to guess or re-derive anything.

## 1. One-line pitch
Single-camera AI system that monitors IV fluid level, drip rate, and patient distress together, fusing them into one risk signal — replacing load-cell/microcontroller hardware with vision + ML.

## 2. Problem
Manual IV monitoring is error-prone and understaffed. Existing solutions (published + patented) do fluid-level detection OR drip-rate counting OR patient monitoring in isolation — never fused. None of them generalize well across bag types/lighting.

## 3. Modules (build order)
1. **Fluid-level estimation** — CNN on video frames of the IV bag → level class/estimate
2. **Drip-rate counting** — YOLO + tracker on the drip chamber → drops/min
3. **Patient visual monitoring** — distress/movement detection from the same or a second camera feed
4. **Fusion & risk scoring** — combine 1–3 into alert tiers (not independent alerts)
5. **Predictive ETA** — learned model estimating time-to-empty from noisy vision data (must be more than volume÷rate arithmetic)
6. **Dashboard/alerting** — web UI + notifications (no physical buzzer/LCD)

## 4. Tech stack (locked — don't re-decide mid-build)
- Python, OpenCV, PyTorch (or TF — pick one and stick to it)
- YOLOv8 or v5 for detection/tracking
- FastAPI or Streamlit for the dashboard
- Prototype target: laptop webcam, no embedded hardware

## 5. Non-negotiable constraints
- NO load cell, HX711, ESP8266/ESP32-CAM, physical buzzer, or Blynk/ThingSpeak cloud
- Patient monitoring and IV monitoring must feed one fusion layer, not separate dashboards
- ETA prediction must be a trained model, not a plain formula
- If a needed decision isn't in Section 6, ask — don't assume and proceed

## 6. Open decisions (fill in as you decide; don't leave silently assumed)
- Distress-detection approach: FER2013-trained EmotionCNN with Haar face detection; this is an expression-derived distress proxy, not a pain model.
- Fluid-level dataset source: Mendeley SESOVERA saline bottles for initial training; self-recorded IV-bag frames are still required for generalization.
- ETA model architecture: LSTM planned; training is blocked until complete bag-run sequences with remaining-time labels are collected.

## 7. Current status (update every session — this is what makes handoff work)
- [x] Initial M1 dataset and model — 2,479 Mendeley bottle images available, 1,984 train / 495 validation; 100% random-split validation accuracy. One image per class also passed inference. Same-source split; not validated on IV bags or camera-rig footage. Downloading the remaining source files was interrupted by network errors.
- [x] M2 YOLO detector and rate tracker — synthetic detector mAP50 0.9928; three generated 10/20/30 dpm videos replay at 0.533 dpm MAE. Real rig footage remains unvalidated.
- [x] M3 FER2013 expression model wired into inference and fine-tuned for one epoch — 66.75% 7-class test accuracy; derived binary distress accuracy 83.42% (AUROC 0.9193). The 80% result applies only to an expression-derived proxy, not clinical distress or pain.
- [x] M4 rule-based fusion implementation — unit tests pass; validation with simultaneous real M1-M3 camera outputs is outstanding.
- [ ] M5 ETA training — no labeled time-series data or trained weights; prediction remains unavailable.
- [x] Dashboard and capture pipeline — one real camera frame traversed the pipeline and `/api/state` returned the snapshot; M2/M3 had no detections, so full model coverage is not verified.
- Last updated by: GitHub Copilot, 2026-10-11
- Known issues / deviations from plan: M1/M2 metrics are from narrow source domains (bottles and generated clips); they do not establish real-world accuracy. M1's random split may contain same-source scene leakage. M3's FER expression score is only a distress proxy. Self-recorded camera/lighting data, ETA labels, simultaneous real-output fusion validation, and a live run with all targets in view remain outstanding. Python 3.8 dashboard compatibility requires `eval_type_backport` (declared conditionally in requirements).

## 8. Team & context
Batch D3, team of 4, guide Mrs. N Divya, GNITS CSE (Data Science), Phase-I GNR-22 (2026–2027).
