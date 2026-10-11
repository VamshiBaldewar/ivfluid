from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_TIMESERIES = DATA_DIR / "timeseries"
APP_DIR = PROJECT_ROOT / "app"
DASHBOARD_STATIC = APP_DIR / "dashboard" / "static"
