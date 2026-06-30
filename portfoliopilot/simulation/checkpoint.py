"""Checkpoint saver (SQLite-backed).

Persists the simulation's runtime state at each completed month so a run can be
resumed after a restart, or rewound to an earlier point in time. State is stored
in a single SQLite database (``data/checkpoints.sqlite``), consistent with the
local monitor and memory stores. The heavier monitor/memory state already lives
in their own SQLite databases.

Schema (table ``checkpoints``):
    completed_month  INTEGER PRIMARY KEY   -- the month index that just finished
    simulation_id    TEXT
    month_index      INTEGER               -- next month to run
    current_date     TEXT
    portfolio_value  REAL
    snapshot         TEXT                  -- full JSON runtime snapshot

The class keeps the same public API the engine/server/tests use:
``save``, ``load``, ``load_latest``, ``list_checkpoints``, ``clear``,
``prune_after``.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .. import config


class SqliteSaver:
    """SQLite-backed point-in-time checkpoint store."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or config.CHECKPOINT_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                completed_month INTEGER PRIMARY KEY,
                simulation_id   TEXT,
                month_index     INTEGER,
                current_date    TEXT,
                portfolio_value REAL,
                snapshot        TEXT
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    def save(self, completed_month: int, snapshot: dict) -> int:
        """Write (upsert) the checkpoint for a completed month."""
        payload = dict(snapshot)
        payload["completed_month"] = completed_month
        self._conn.execute(
            """INSERT OR REPLACE INTO checkpoints
               (completed_month, simulation_id, month_index, current_date,
                portfolio_value, snapshot)
               VALUES (?,?,?,?,?,?)""",
            (
                completed_month,
                payload.get("simulation_id"),
                payload.get("month_index"),
                payload.get("current_date"),
                payload.get("portfolio_value"),
                json.dumps(payload),
            ),
        )
        self._conn.commit()
        return completed_month

    def load(self, completed_month: int) -> dict | None:
        row = self._conn.execute(
            "SELECT snapshot FROM checkpoints WHERE completed_month=?",
            (completed_month,),
        ).fetchone()
        return json.loads(row["snapshot"]) if row else None

    def load_latest(self) -> dict | None:
        row = self._conn.execute(
            "SELECT snapshot FROM checkpoints ORDER BY completed_month DESC LIMIT 1"
        ).fetchone()
        return json.loads(row["snapshot"]) if row else None

    def list_checkpoints(self) -> list[dict]:
        """Compact metadata for every saved checkpoint, oldest first."""
        rows = self._conn.execute(
            """SELECT completed_month, current_date, month_index, portfolio_value,
                      simulation_id
               FROM checkpoints ORDER BY completed_month"""
        ).fetchall()
        return [
            {
                "completed_month": r["completed_month"],
                "date": r["current_date"],
                "month_index": r["month_index"],
                "portfolio_value": r["portfolio_value"],
                "simulation_id": r["simulation_id"],
            }
            for r in rows
        ]

    def clear(self) -> None:
        self._conn.execute("DELETE FROM checkpoints")
        self._conn.commit()

    def prune_after(self, completed_month: int) -> None:
        """Delete checkpoints for months later than ``completed_month`` (rewind)."""
        self._conn.execute(
            "DELETE FROM checkpoints WHERE completed_month > ?", (completed_month,)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# Backwards-compatible alias.
CheckpointSaver = SqliteSaver
