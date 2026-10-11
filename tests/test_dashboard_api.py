from fastapi.testclient import TestClient

import app.dashboard.server as server
from app.dashboard.server import app


def test_state_endpoint():
    client = TestClient(app)
    res = client.get("/api/state")
    assert res.status_code == 200
    body = res.json()
    assert "risk" in body
    assert "fluid" in body


def test_start_uses_default_source(monkeypatch):
    class PipelineStub:
        source = 0
        running = False

        def start(self):
            return True

    monkeypatch.setattr(server, "PIPELINE", PipelineStub())
    client = TestClient(app)

    res = client.post("/api/pipeline/start")

    assert res.status_code == 200
    assert res.json()["source"] == "0"


def test_record_start_reports_unavailable_source(monkeypatch):
    class PipelineStub:
        source = 0
        running = False

        def start(self):
            return False

    monkeypatch.setattr(server, "PIPELINE", PipelineStub())
    client = TestClient(app)

    res = client.post(
        "/api/record/start",
        json={"module": "drip_rate", "label": "20dpm", "kind": "video"},
    )

    assert res.status_code == 409
    assert "Could not open source" in res.json()["detail"]
