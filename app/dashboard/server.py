from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.paths import DASHBOARD_STATIC
from app.pipeline import PIPELINE
from app.state import STATE

app = FastAPI(title="IV Fluid Vision Monitor", version="0.1.0")
app.mount("/static", StaticFiles(directory=DASHBOARD_STATIC), name="static")


class RecordRequest(BaseModel):
    module: str = Field(pattern="^(fluid_level|drip_rate|patient)$")
    label: str = Field(min_length=1, max_length=64)
    kind: str = Field(pattern="^(video|frames)$")
    interval: float = Field(default=1.0, ge=0.2, le=10.0)


class SourceRequest(BaseModel):
    source: str = Field(default="0", description="Webcam index like 0 or a video file path")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DASHBOARD_STATIC / "index.html")


@app.get("/api/state")
def api_state() -> dict:
    return STATE.get().public_dict()


@app.get("/api/frame")
def api_frame() -> Response:
    snap = STATE.get()
    if not snap.frame_jpeg:
        return Response(status_code=204)
    return Response(content=snap.frame_jpeg, media_type="image/jpeg")


@app.post("/api/pipeline/start")
def api_start(body: SourceRequest | None = None) -> dict:
    if body is not None:
        src: int | str = int(body.source) if body.source.isdigit() else body.source
        if PIPELINE.running:
            PIPELINE.stop()
        PIPELINE.source = src
        PIPELINE.camera.source = src
    if not PIPELINE.start():
        raise HTTPException(status_code=409, detail=f"Could not open source: {PIPELINE.source}")
    return {"ok": True, "running": True, "source": str(PIPELINE.source)}


@app.post("/api/pipeline/stop")
def api_stop() -> dict:
    PIPELINE.stop()
    return {"ok": True, "running": False}


@app.post("/api/record/start")
def api_record_start(body: RecordRequest) -> dict:
    if not PIPELINE.running and not PIPELINE.start():
        raise HTTPException(status_code=409, detail=f"Could not open source: {PIPELINE.source}")
    try:
        if body.kind == "video":
            path = PIPELINE.start_dataset_video(body.module, body.label)
        else:
            path = PIPELINE.start_dataset_frames(body.module, body.label, body.interval)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "path": path}


@app.post("/api/record/stop")
def api_record_stop() -> dict:
    return {"ok": True, **PIPELINE.stop_dataset()}


def run() -> None:
    import uvicorn

    PIPELINE.start()
    uvicorn.run("app.dashboard.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
