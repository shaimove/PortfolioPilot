"""FastAPI server: dashboard backend + simulation controls.

Run with:  python -m portfoliopilot.server
Then open: http://127.0.0.1:8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .simulation.engine import SimulationEngine

app = FastAPI(title="PortfolioPilot", version="0.1.0")
engine = SimulationEngine()


class SpeedBody(BaseModel):
    seconds_per_month: float


class RestoreBody(BaseModel):
    completed_month: int


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "data_ready": engine.data_ready}


@app.get("/api/state")
def state() -> JSONResponse:
    if not engine.data_ready:
        return JSONResponse({
            "error": "data_not_ready",
            "message": "Run ingestion + build_features first "
                       "(see README / scripts).",
            "controls": {"data_ready": False, "month_index": 0,
                         "total_months": config.TOTAL_MONTHS, "running": False,
                         "finished": False, "progress": f"0 / {config.TOTAL_MONTHS}"},
        })
    return JSONResponse(engine.get_state())


@app.post("/api/start")
def start() -> dict:
    if not engine.data_ready:
        return {"ok": False, "message": "Data not ready. Run ingestion first."}
    engine.start()
    return {"ok": True}


@app.post("/api/pause")
def pause() -> dict:
    engine.pause()
    return {"ok": True}


@app.post("/api/reset")
def reset() -> dict:
    engine.reset()
    return {"ok": True}


@app.post("/api/step")
def step() -> dict:
    if not engine.data_ready:
        return {"ok": False, "message": "Data not ready."}
    rec = engine.step()
    return {"ok": True, "recorded": rec is not None}


@app.post("/api/speed")
def speed(body: SpeedBody) -> dict:
    engine.set_speed(body.seconds_per_month)
    return {"ok": True, "seconds_per_month": engine.state.seconds_per_month}


# --------------------------------------------------------------------------- #
# Checkpoints (point-in-time save / restore)
# --------------------------------------------------------------------------- #
@app.get("/api/checkpoints")
def checkpoints() -> dict:
    return {"checkpoints": engine.checkpointer.list_checkpoints()}


@app.post("/api/resume")
def resume() -> dict:
    ok = engine.resume_latest()
    return {"ok": ok, "month_index": engine.state.month_index}


@app.post("/api/restore")
def restore(body: RestoreBody) -> dict:
    ok = engine.restore_to(body.completed_month)
    return {"ok": ok, "month_index": engine.state.month_index}


# --------------------------------------------------------------------------- #
# Static dashboard
# --------------------------------------------------------------------------- #
DASHBOARD = Path(config.DASHBOARD_DIR)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DASHBOARD / "index.html")


if DASHBOARD.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD)), name="static")


def main() -> None:
    import uvicorn

    config.ensure_dirs()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
