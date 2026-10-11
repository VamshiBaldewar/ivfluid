# IV Fluid AI: Team Presentation Brief

## Project in one minute

This is a laptop-camera prototype for monitoring an IV setup. It sends camera frames to vision models for fluid level, drip rate, and facial expression, combines available results into a risk tier, and displays the status in a local web dashboard. It is a research prototype, not a medical device or clinical decision system.

## Why we built it

IV bags and drip chambers need regular checks. The project explores whether one camera and AI models can help show fluid level, count drops, and surface possible risks together, instead of presenting each signal separately.

## How it works

1. Open the laptop camera and capture a frame.
2. Run M1, M2, and M3 on that frame.
3. Combine available results with M4 rules.
4. Send the frame and latest results to the FastAPI dashboard.
5. Log measurements over time for possible future ETA training.

The current pipeline passes the same full frame to all three models. Separate bag, chamber, and face crops are not implemented yet.

## Models and current results

| Module | What it does | Training and measured result |
|---|---|---|
| **M1: Fluid level** | MobileNetV3-Small classifies fill as 0%, 50%, 80%, or 100%. | Trained on 2,479 Mendeley saline-bottle images: 1,984 train and 495 validation. Validation accuracy was 100%; one local image from each class also ran through inference correctly. The holdout is a random split from the same source, not a test on real IV bags. |
| **M2: Drip rate** | YOLOv8n detects drops; a tracker estimates drops per minute over time. | Detector mAP50 was 99.28% on synthetic data. Three generated videos had a mean rate error of 0.533 drops/min at 10, 20, and 30 dpm. No real rig video has been validated. |
| **M3: Patient expression** | Haar face detection finds a face; an EmotionCNN classifies FER2013 expressions and derives a distress score from selected expressions. | FER2013 test accuracy: 66.75% across seven expressions. Derived binary distress accuracy: 83.42%, AUROC 0.9193. This is an expression proxy, not a pain or clinical distress detector. |
| **M4: Risk fusion** | Applies simple rules to M1-M3 outputs and returns a risk tier. | Prototype thresholds: low bag below 40%, high below 20%, critical below 10%; distress thresholds at 0.4, 0.6, and 0.8; drip at or below 0.5 dpm is treated as stopped. Missing inputs cannot produce a LOW tier; the result is UNKNOWN when low risk cannot be confirmed. |
| **M5: ETA** | Intended to estimate minutes remaining from a trained time-series model. | Not trained. There is no ETA checkpoint or labeled complete bag-run data, so ETA reports unavailable. It does not use a simple volume-divided-by-rate formula. |
| **M6: Dashboard** | Displays the live frame and latest model/risk results through a local FastAPI web app. | The camera-to-dashboard path was smoke-tested. A frame reached the API, but M2 and M3 produced no detections in that test; M5 was unavailable and fusion showed UNKNOWN. |

## Data and limitations

- M1's public dataset is saline **bottles**, not IV bags. The attempted full download was interrupted by network failures; 2,479 images are currently available, with uneven class counts.
- M2's model and rate test videos are synthetic. The reported scores do not establish performance on real drip-chamber footage.
- M3 uses FER2013 facial expressions. Facial expression is not a clinical measure of pain or distress.
- M4 is a rule-based prototype. It still needs validation with all three models detecting their targets together in a real camera scene.
- M5 needs complete labeled bag runs, including timestamps and remaining-time targets.
- A live demo depends on placing the bag, chamber, and a visible face in view. The models may return unavailable when they cannot detect their target.
- The full automated test suite passes: 13 tests. Passing tests verifies software behavior, not clinical accuracy.

## Demo instructions

From the project folder, run:

```powershell
.\.venv\Scripts\python -m app.dashboard.server
```

Open `http://127.0.0.1:8000` and start the pipeline. Show the live frame, module statuses, risk tier, and any available ETA. Be prepared for M2, M3, and M5 to report unavailable depending on the camera view and missing ETA data.

## Suggested presentation wording

> “We built a camera-based prototype that brings IV fluid level, drip counting, and facial-expression signals into one dashboard. Our initial tests show the models and pipeline can run, but the data is limited: fluid-level tests use bottles, drip tests use generated videos, and ETA still needs labeled bag-run recordings. The results are not clinical validation.”

## Claims to avoid

- Do not say the system is 100% accurate in real use; M1's 100% is only on a random bottle-image holdout.
- Do not say M3 diagnoses pain or patient distress.
- Do not present synthetic M2 test results as real-camera performance.
- Do not call the dashboard a replacement for nurse monitoring or claim clinical readiness.
