# IV Fluid AI

Vision-based prototype for estimating IV fluid level, drip rate, and an
expression-derived patient distress score from a camera feed. A local web
dashboard displays the results and a rule-based risk tier.

> **Research prototype only.** This software is not a medical device and must
> not be used to make clinical decisions or replace patient monitoring.

## Requirements

- Git
- 64-bit Python 3.8–3.11 (Python 3.10 or 3.11 recommended)
- A webcam for live monitoring, or a local video file for replay
- Internet access to install Python dependencies

The trained M1, M2, and M3 checkpoints are included in the repository; training
is not required to run the dashboard. A CPU is sufficient. The setup below
installs CPU-only PyTorch; a GPU is optional.

## 1. Clone the repository

```sh
git clone https://github.com/VamshiBaldewar/ivfluid.git
cd ivfluid
```

If you downloaded the repository as a ZIP instead, extract it and open a
terminal in the extracted `ivfluid` folder.

## 2. Create an environment and install dependencies

Use a virtual environment so the project dependencies do not affect other
Python projects. Run the commands for your operating system from the project
folder.

### Windows (PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

If Python 3.10 is not installed, install a 64-bit Python 3.8–3.11 release and
replace `-3.10` with its version (for example, `-3.11`). If PowerShell blocks
activation, open a new terminal with the appropriate policy or run the
environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

### macOS or Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

The dependency install, especially PyTorch, can take a while and needs several
gigabytes of free disk space. Keep the virtual environment activated in the
terminal used to run the following commands. To leave it later, run
`deactivate`.

## 3. Start the dashboard

From the repository folder, with the virtual environment active, run:

```sh
python -m uvicorn app.dashboard.server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> in a browser. The dashboard is bound to the local
machine only. Click **Start pipeline** to open the selected source; the source
field defaults to webcam `0`. Click **Stop** to stop camera processing. Stop
the web server with `Ctrl+C` in its terminal.

To replay a video instead of using a webcam, enter its path in the dashboard's
source field (for example, `C:\Videos\iv-demo.mp4` on Windows or
`/home/user/videos/iv-demo.mp4` on Linux), then click **Start pipeline**.

## 4. Verify the installation

In the activated environment, run the test suite:

```sh
python -m pytest
```

The tests exercise the dashboard API, shared state, inference fallback
behavior, and risk fusion. They do not validate medical or real-world model
accuracy.

## Optional: record training data

The dashboard includes a dataset recorder. Start the pipeline, select a module
and label, choose video or frames, then start and stop recording. Files are
saved under `data/raw/<module>/<label>/` and are intentionally excluded from
Git.

Recording can also be started from a terminal:

```sh
# Record 60 seconds of drip-rate video from webcam 0
python -m app.collect --module drip_rate --label 20dpm --seconds 60 --source 0

# Save 40 fluid-level frames, one second apart
python -m app.collect --module fluid_level --label 50 --frames 40 --interval 1
```

Use `python -m app.collect --help` to see all options. The source can be a
webcam index or a video file path.

## Troubleshooting

- **Camera will not start:** Close other apps using the webcam and check that
  the operating system has granted camera access to desktop applications.
  On Linux, check access to `/dev/video0`. Try another camera index (`1`, for
  example) or use a video file as the source.
- **Browser opens, but no frame appears:** Confirm the pipeline was started
  from the dashboard and check the terminal for errors. A camera can only be
  used by one process at a time.
- **`No module named ...`:** Activate the project's `.venv` and install the
  requirements again from the repository folder. On Windows, use
  `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` (add the
  CPU PyTorch `--extra-index-url` shown above if PyTorch is missing).
- **Model reports unavailable:** Confirm the clone contains
  `models/fluid_level/weights/fluid_level.pt`,
  `models/drip_rate/weights/drip_yolo.pt`, and
  `models/patient/weights/emotion_fer.pt`. These checkpoints are tracked in
  Git. If a model loads but cannot detect its target, improve the camera
  framing and lighting; the current models may not recognize every bag,
  chamber, or face.
- **ETA shows unavailable:** This is expected. M5 has no trained ETA checkpoint
  yet; the application logs time-series samples for future model development.

## Project information

See [fulldetails.md](fulldetails.md) for model results, demo guidance, and
limitations; [ARCHITECTURE.md](ARCHITECTURE.md) for the component/data flow;
and [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for current development status.
The reported model results use limited datasets and do not establish
real-world or clinical accuracy.
