"""Local monitor (SQLite).

Persists per-month step records and incident alerts, and computes the aggregate
metrics the dashboard displays. Survives process restarts.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

from .. import config
from . import metrics as M


class LocalMonitor:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or config.MONITOR_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS steps (
                month_index INTEGER PRIMARY KEY,
                date TEXT,
                record TEXT
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                month_index INTEGER,
                date TEXT,
                severity TEXT,
                type TEXT,
                message TEXT
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    def record_step(self, month_index: int, date: str, record: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO steps (month_index, date, record) VALUES (?,?,?)",
            (month_index, date, json.dumps(record)),
        )
        self._conn.commit()

    def record_alert(self, month_index: int, date: str, severity: str,
                     type_: str, message: str) -> None:
        self._conn.execute(
            "INSERT INTO alerts (month_index, date, severity, type, message) VALUES (?,?,?,?,?)",
            (month_index, date, severity, type_, message),
        )
        self._conn.commit()

    def steps(self) -> list[dict]:
        rows = self._conn.execute("SELECT record FROM steps ORDER BY month_index").fetchall()
        return [json.loads(r["record"]) for r in rows]

    def alerts(self, limit: int = 200) -> list[dict]:
        rows = self._conn.execute(
            "SELECT month_index, date, severity, type, message FROM alerts "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_after(self, month_index: int) -> None:
        """Remove steps and alerts for months later than `month_index` (rewind)."""
        self._conn.execute("DELETE FROM steps WHERE month_index > ?", (month_index,))
        self._conn.execute("DELETE FROM alerts WHERE month_index > ?", (month_index,))
        self._conn.commit()

    def reset(self) -> None:
        self._conn.executescript("DELETE FROM steps; DELETE FROM alerts;")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ #
    def aggregates(self) -> dict:
        """Compute the dashboard's rolled-up metrics from stored steps."""
        steps = self.steps()
        if not steps:
            return _empty_aggregates()

        port_vals = [s["portfolio_value"] for s in steps]
        bench_vals = [s["benchmark_value"] for s in steps]
        total_costs = sum(s.get("transaction_cost", 0.0) for s in steps)

        def _sum(key: str) -> float:
            return sum(s.get(key, 0) or 0 for s in steps)

        def _avg_judge(key: str) -> float:
            vals = [s["judge"].get(key) for s in steps if s.get("judge")]
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        no_action = sum(1 for s in steps if s.get("action") in ("no_action", "hold"))

        return {
            "financial": {
                "portfolio_value": round(port_vals[-1], 2),
                "benchmark_value": round(bench_vals[-1], 2),
                "total_return": round(M.total_return(port_vals), 4),
                "benchmark_return": round(M.total_return(bench_vals), 4),
                "excess_return": round(M.excess_return(port_vals, bench_vals), 4),
                "max_drawdown": round(M.max_drawdown(port_vals), 4),
                "turnover_avg": round(M.average([s.get("turnover", 0.0) for s in steps]), 4),
                "transaction_cost_drag": round(
                    M.transaction_cost_drag(total_costs, config.STARTING_CAPITAL), 4),
            },
            "agent": {
                "transaction_count": int(_sum("transaction_count")),
                "buy_count": int(_sum("buy_count")),
                "sell_count": int(_sum("sell_count")),
                "forced_sell_count": int(_sum("forced_sell_count")),
                "changed_positions_count": int(_sum("changed_positions_count")),
                "risk_blocked_count": int(_sum("risk_blocked")),
                "risk_modified_count": int(_sum("risk_modified")),
                "constraint_violation_count": int(_sum("constraint_violation_count")),
                "same_asset_flip_count": int(_sum("same_asset_flip_count")),
                "no_action_rate": round(no_action / len(steps), 4),
                "invalid_json_count": int(_sum("invalid_json_count")),
                "retry_count": int(_sum("retry_count")),
            },
            "judge": {
                "groundedness_score": _avg_judge("groundedness_score"),
                "hallucination_risk": _avg_judge("hallucination_risk"),
                "constraint_awareness_score": _avg_judge("constraint_awareness_score"),
                "memory_use_score": _avg_judge("memory_use_score"),
                "decision_consistency_score": _avg_judge("decision_consistency_score"),
                "unsupported_claim_count": int(_sum("unsupported_claim_count")),
                "judge_warning_count": int(_sum("judge_warning_count")),
                "judge_critical_count": int(_sum("judge_critical_count")),
            },
        }


def _empty_aggregates() -> dict:
    return {
        "financial": {
            "portfolio_value": config.STARTING_CAPITAL,
            "benchmark_value": config.STARTING_CAPITAL,
            "total_return": 0.0, "benchmark_return": 0.0, "excess_return": 0.0,
            "max_drawdown": 0.0, "turnover_avg": 0.0, "transaction_cost_drag": 0.0,
        },
        "agent": {
            "transaction_count": 0, "buy_count": 0, "sell_count": 0,
            "forced_sell_count": 0, "changed_positions_count": 0,
            "risk_blocked_count": 0, "risk_modified_count": 0,
            "constraint_violation_count": 0, "same_asset_flip_count": 0,
            "no_action_rate": 0.0, "invalid_json_count": 0, "retry_count": 0,
        },
        "judge": {
            "groundedness_score": None, "hallucination_risk": None,
            "constraint_awareness_score": None, "memory_use_score": None,
            "decision_consistency_score": None, "unsupported_claim_count": 0,
            "judge_warning_count": 0, "judge_critical_count": 0,
        },
    }
