"""Checkpoint saver.

Persists the simulation's runtime state at each completed month so a run can be
resumed after a restart, or rewound to an earlier point in time. Checkpoints are
small JSON files; the heavier monitor/memory state already lives in SQLite.

Layout:
    data/checkpoints/
        month_000.json
        month_001.json
        ...
        latest.json        (copy of the most recent month checkpoint)
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import config

CHECKPOINT_DIR = config.DATA_DIR / "checkpoints"


class CheckpointSaver:
    def __init__(self, directory: Path | str | None = None) -> None:
        self.dir = Path(directory or CHECKPOINT_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _month_path(self, completed_month: int) -> Path:
        return self.dir / f"month_{completed_month:03d}.json"

    @property
    def latest_path(self) -> Path:
        return self.dir / "latest.json"

    def save(self, completed_month: int, snapshot: dict) -> Path:
        """Write the checkpoint for a completed month (and update latest.json)."""
        payload = dict(snapshot)
        payload["completed_month"] = completed_month
        path = self._month_path(completed_month)
        path.write_text(json.dumps(payload, indent=2))
        self.latest_path.write_text(json.dumps(payload, indent=2))
        return path

    def load(self, completed_month: int) -> dict | None:
        p = self._month_path(completed_month)
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def load_latest(self) -> dict | None:
        if not self.latest_path.exists():
            return None
        return json.loads(self.latest_path.read_text())

    def list_checkpoints(self) -> list[dict]:
        """Return compact metadata for every saved month checkpoint."""
        out = []
        for p in sorted(self.dir.glob("month_*.json")):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            out.append({
                "completed_month": d.get("completed_month"),
                "date": d.get("current_date"),
                "month_index": d.get("month_index"),
                "portfolio_value": d.get("portfolio_value"),
                "simulation_id": d.get("simulation_id"),
            })
        return out

    def clear(self) -> None:
        for p in self.dir.glob("month_*.json"):
            p.unlink(missing_ok=True)
        self.latest_path.unlink(missing_ok=True)

    def prune_after(self, completed_month: int) -> None:
        """Delete checkpoints for months later than `completed_month` (used when
        rewinding to a point in time)."""
        for p in self.dir.glob("month_*.json"):
            try:
                idx = int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if idx > completed_month:
                p.unlink(missing_ok=True)
