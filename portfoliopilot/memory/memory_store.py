"""Structured local memory store (SQLite-backed).

Memory types: investment_thesis, decision_memory, risk_lesson, strategy_lesson,
asset_history, mistake_memory, stale_thesis, constraint_memory,
index_membership_event.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .. import config

MEMORY_TYPES = {
    "investment_thesis", "decision_memory", "risk_lesson", "strategy_lesson",
    "asset_history", "mistake_memory", "stale_thesis", "constraint_memory",
    "index_membership_event",
}


@dataclass
class Memory:
    memory_id: str
    date_created: str
    type: str
    related_assets: list[str]
    content: str
    status: str = "active"          # active | stale | retired
    confidence: float = 0.5
    source_trace_id: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def new_memory(type: str, content: str, related_assets: list[str] | None = None,
               date_created: str | None = None, confidence: float = 0.5,
               source_trace_id: str | None = None,
               valid_from: str | None = None, valid_until: str | None = None) -> Memory:
    today = date_created or dt.date.today().isoformat()
    return Memory(
        memory_id="mem_" + uuid.uuid4().hex[:10],
        date_created=today,
        type=type if type in MEMORY_TYPES else "decision_memory",
        related_assets=related_assets or [],
        content=content,
        confidence=float(confidence),
        source_trace_id=source_trace_id,
        valid_from=valid_from or today,
        valid_until=valid_until,
    )


class MemoryStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or config.MEMORY_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                date_created TEXT,
                type TEXT,
                related_assets TEXT,
                content TEXT,
                status TEXT,
                confidence REAL,
                source_trace_id TEXT,
                valid_from TEXT,
                valid_until TEXT
            )
            """
        )
        self._conn.commit()

    def add(self, mem: Memory) -> Memory:
        self._conn.execute(
            """INSERT OR REPLACE INTO memories VALUES
               (?,?,?,?,?,?,?,?,?,?)""",
            (
                mem.memory_id, mem.date_created, mem.type,
                json.dumps(mem.related_assets), mem.content, mem.status,
                mem.confidence, mem.source_trace_id, mem.valid_from, mem.valid_until,
            ),
        )
        self._conn.commit()
        return mem

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            memory_id=row["memory_id"],
            date_created=row["date_created"],
            type=row["type"],
            related_assets=json.loads(row["related_assets"] or "[]"),
            content=row["content"],
            status=row["status"],
            confidence=row["confidence"],
            source_trace_id=row["source_trace_id"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
        )

    def all(self) -> list[Memory]:
        rows = self._conn.execute("SELECT * FROM memories ORDER BY date_created").fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get(self, memory_id: str) -> Memory | None:
        r = self._conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        return self._row_to_memory(r) if r else None

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"])

    def count_by_status(self, status: str) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(*) AS c FROM memories WHERE status=?", (status,)).fetchone()["c"])

    def mark_stale(self, memory_id: str) -> None:
        self._conn.execute("UPDATE memories SET status='stale' WHERE memory_id=?", (memory_id,))
        self._conn.commit()

    def mark_stale_before(self, cutoff_date: str, types: set[str] | None = None) -> int:
        """Mark active memories created before cutoff as stale (thesis decay)."""
        rows = self._conn.execute(
            "SELECT memory_id, type, date_created FROM memories WHERE status='active'"
        ).fetchall()
        n = 0
        for r in rows:
            if r["date_created"] < cutoff_date and (types is None or r["type"] in types):
                self.mark_stale(r["memory_id"])
                n += 1
        return n

    def reset(self) -> None:
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
