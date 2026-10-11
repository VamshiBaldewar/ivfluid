from fastapi.testclient import TestClient

from app.dashboard.server import app


def test_state_endpoint():
    client = TestClient(app)
    res = client.get("/api/state")
    assert res.status_code == 200
    body = res.json()
    assert "risk" in body
    assert "fluid" in body
