"""Dashboard backend smoke tests (call route handlers directly)."""
import json

from portfoliopilot import server


def test_health_endpoint():
    out = server.health()
    assert out["status"] == "ok"
    assert "data_ready" in out


def test_state_endpoint_returns_valid_payload(offline_data):
    resp = server.state()
    assert resp.status_code == 200
    payload = json.loads(resp.body)
    # with data ready, the controls block must be present
    assert "controls" in payload
    assert payload["controls"]["total_months"] == 120


def test_speed_control_updates_engine():
    out = server.speed(server.SpeedBody(seconds_per_month=1.0))
    assert out["ok"] is True
    assert out["seconds_per_month"] == 1.0


def test_checkpoint_endpoints(offline_data):
    server.engine.reset()
    server.engine.load()
    server.engine.step()
    server.engine.step()

    cps = server.checkpoints()
    assert "checkpoints" in cps
    assert [c["completed_month"] for c in cps["checkpoints"]] == [0, 1]

    out = server.restore(server.RestoreBody(completed_month=0))
    assert out["ok"] is True
    assert out["month_index"] == 1

    resumed = server.resume()
    assert resumed["ok"] is True
    server.engine.reset()
