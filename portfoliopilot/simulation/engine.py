"""Monthly simulation engine.

Runs up to 120 monthly steps from local cached data only (no network). Each
month executes the full pipeline: universe -> snapshot -> agent -> risk ->
execution -> benchmark -> memory -> judge -> monitor -> dashboard state.

The engine is controllable (start / pause / reset / speed) and runs its loop on
a background thread so the dashboard stays responsive.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
import uuid
from dataclasses import dataclass, field

import pandas as pd

from .. import config
from ..agent.decision_agent import DecisionAgent
from ..config import Constraints
from ..data import cache
from ..data.universe import Universe
from ..execution import broker_simulator as broker
from ..execution import risk_engine
from ..features.feature_engine import FeatureStore
from ..memory import retriever
from ..memory.memory_store import MemoryStore, new_memory
from ..monitoring import alerts as alert_engine
from ..monitoring import metrics as M
from ..monitoring.judge import Judge
from ..monitoring.langsmith_tracing import Tracer
from ..monitoring.local_monitor import LocalMonitor
from ..utils import to_jsonable
from .checkpoint import SqliteSaver

STALE_MEMORY_MONTHS = 36


@dataclass
class EngineState:
    running: bool = False
    finished: bool = False
    month_index: int = 0
    total_months: int = config.TOTAL_MONTHS
    seconds_per_month: float = config.DEFAULT_SECONDS_PER_MONTH
    current_date: str | None = None
    simulation_id: str = field(default_factory=lambda: "sim_" + uuid.uuid4().hex[:8])


class SimulationEngine:
    def __init__(self, constraints: Constraints | None = None) -> None:
        self.constraints = constraints or config.DEFAULT_CONSTRAINTS
        self.state = EngineState()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # data stores (loaded lazily so the server can start before ingestion)
        self._loaded = False
        self.universe: Universe | None = None
        self.features: FeatureStore | None = None
        self.month_ends: list[dt.date] = []
        self.sectors: dict[str, str] = {}
        self.valid_flags: dict[str, bool] = {}
        self.benchmark_series: dict[dt.date, float] = {}

        # runtime components
        self.agent = DecisionAgent(self.constraints)
        self.judge = Judge()
        self.tracer = Tracer()
        self.monitor = LocalMonitor()
        self.memory = MemoryStore()
        self.checkpointer = SqliteSaver()

        # portfolio runtime
        self.portfolio = broker.Portfolio.initial(config.STARTING_CAPITAL)
        self.benchmark_shares = 0.0
        self.history: list[dict] = []
        self._last_trade_sides: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        with self._lock:
            self.universe = Universe.load()
            self.features = FeatureStore.load()
            self.month_ends = self.universe.month_ends[: self.state.total_months]

            meta = cache.read_parquet(config.METADATA_PARQUET)
            if meta is not None and not meta.empty:
                if "sector" in meta.columns:
                    self.sectors = {r.ticker: r.sector for r in meta.itertuples()
                                    if pd.notna(getattr(r, "sector", None))}
                self.valid_flags = {r.ticker: bool(getattr(r, "valid_history", True))
                                    for r in meta.itertuples()}

            # benchmark monthly series from feature store
            self.benchmark_series = {}
            for me in self.month_ends:
                px = self.features.price_on(me, config.BENCHMARK_TICKER)
                if px is not None:
                    self.benchmark_series[me] = px
            self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    @property
    def data_ready(self) -> bool:
        try:
            return (cache.read_parquet(config.MONTHLY_FEATURES_PARQUET) is not None
                    and cache.read_parquet(config.CONSTITUENTS_MONTHLY_PARQUET) is not None)
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Controls
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._ensure_loaded()
        with self._lock:
            if self.state.running or self.state.finished:
                if self.state.finished:
                    return
            self.state.running = True
            self._stop.clear()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run_loop, daemon=True)
                self._thread.start()

    def pause(self) -> None:
        with self._lock:
            self.state.running = False

    def reset(self) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        with self._lock:
            self.state = EngineState(seconds_per_month=self.state.seconds_per_month)
            self.portfolio = broker.Portfolio.initial(config.STARTING_CAPITAL)
            self.benchmark_shares = 0.0
            self.history = []
            self._last_trade_sides = {}
            self.monitor.reset()
            self.memory.reset()
            self.checkpointer.clear()
            self._thread = None
            self._stop.clear()

    def set_speed(self, seconds_per_month: float) -> None:
        with self._lock:
            self.state.seconds_per_month = max(config.MIN_SECONDS_PER_MONTH,
                                               float(seconds_per_month))

    # ------------------------------------------------------------------ #
    # Run loop
    # ------------------------------------------------------------------ #
    def _run_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                running = self.state.running
                idx = self.state.month_index
                spm = self.state.seconds_per_month
            if not running:
                time.sleep(0.1)
                continue
            if idx >= min(self.state.total_months, len(self.month_ends)):
                with self._lock:
                    self.state.running = False
                    self.state.finished = True
                return
            self.step()
            # pace the simulated month (interruptible)
            slept = 0.0
            while slept < spm and not self._stop.is_set():
                with self._lock:
                    if not self.state.running:
                        break
                time.sleep(min(0.1, spm - slept))
                slept += 0.1

    # ------------------------------------------------------------------ #
    # One month
    # ------------------------------------------------------------------ #
    def step(self) -> dict | None:
        self._ensure_loaded()
        with self._lock:
            idx = self.state.month_index
            if idx >= min(self.state.total_months, len(self.month_ends)):
                self.state.finished = True
                return None
            me = self.month_ends[idx]
            prev_me = self.month_ends[idx - 1] if idx > 0 else None

        feats = self.features.features_on(me)
        prices = {t: float(feats.loc[t, "current_price"]) for t in feats.index
                  if pd.notna(feats.loc[t, "current_price"])}

        members = self.universe.members_on(me)
        held = {t for t, sh in self.portfolio.shares.items() if sh > 0}
        forced_sells = held - members
        new_entrants = (self.universe.added_between(prev_me, me) if prev_me else set())

        eligible = {t for t in members
                    if t in feats.index and bool(feats.loc[t, "valid_history"])}

        # ---- build compact agent input ----
        pre_value = self.portfolio.value(prices)
        pre_weights = self.portfolio.weights(prices)
        running_dd = self._running_drawdown(pre_value)
        turnover_3m = sum(s.get("turnover", 0.0) for s in self.history[-3:])

        candidates = []
        for t in sorted(eligible | held):
            if t not in feats.index:
                continue
            row = feats.loc[t]
            candidates.append({
                "ticker": t,
                "sector": self.sectors.get(t),
                "current_weight": round(pre_weights.get(t, 0.0), 4),
                "return_3m": _num(row.get("ret_3m")),
                "return_12m": _num(row.get("ret_12m")),
                "volatility_3m": _num(row.get("vol_3m")),
                "drawdown": _num(row.get("drawdown")),
                "trend": row.get("ma_trend") or "unknown",
            })

        # ---- retrieve memories ----
        self.memory.mark_stale_before(self._cutoff_date(me))
        mems = retriever.retrieve(
            self.memory, as_of=me,
            tickers=list(held) + [c["ticker"] for c in candidates[:10]],
            keywords="diversification drawdown turnover momentum",
            k=5,
        )
        relevant_memories = [{"memory_id": m.memory_id, "type": m.type,
                              "content": m.content, "status": m.status} for m in mems]

        agent_input = to_jsonable({
            "date": me.isoformat(),
            "portfolio_state": {
                "value": round(pre_value, 2),
                "cash_weight": round(pre_weights.get("CASH", 0.0), 4),
                "positions": {t: round(w, 4) for t, w in pre_weights.items() if t != "CASH"},
                "drawdown": round(running_dd, 4),
                "turnover_last_3m": round(turnover_3m, 4),
            },
            "constraints": self.constraints.as_dict(),
            "eligible_candidates": candidates,
            "forced_actions": [{"ticker": t, "reason": "removed_from_sp500", "action": "sell"}
                               for t in sorted(forced_sells)],
            "relevant_memories": relevant_memories,
        })

        # ---- agent decision ----
        decision = self.agent.decide(agent_input)

        # ---- risk validation / repair ----
        risk = risk_engine.validate_and_repair(
            target_weights=decision.output.get("target_weights", {}),
            current_weights=pre_weights,
            eligible=eligible,
            sectors=self.sectors,
            valid_history=self.valid_flags,
            forced_sells=forced_sells,
            constraints=self.constraints,
        )

        # ---- execution ----
        exec_res = broker.execute(
            self.portfolio,
            target_weights=risk.final_weights,
            prices=prices,
            forced_sells=forced_sells,
            prev_weights=pre_weights,
            new_entrants=new_entrants,
        )

        # ---- benchmark ----
        bench_value = self._benchmark_value(me)

        # ---- forced sell verification ----
        still_held_forced = sum(1 for t in forced_sells if self.portfolio.shares.get(t, 0.0) > 1e-9)

        # ---- same-asset flip detection ----
        flips = 0
        cur_sides = {tr["ticker"]: tr["side"] for tr in exec_res.trades}
        for tkr, side in cur_sides.items():
            prev_side = self._last_trade_sides.get(tkr)
            if prev_side and prev_side != side:
                flips += 1
        self._last_trade_sides = cur_sides

        # ---- memory writes ----
        self._write_memories(decision, me)
        for t in sorted(forced_sells):
            self.memory.add(new_memory("index_membership_event",
                            f"{t} removed from S&P 500 universe; forced sell at {me.isoformat()}.",
                            related_assets=[t], date_created=me.isoformat()))
        for t in sorted(new_entrants):
            self.memory.add(new_memory("index_membership_event",
                            f"{t} added to S&P 500 universe; eligible from {me.isoformat()}.",
                            related_assets=[t], date_created=me.isoformat()))

        # ---- judge ----
        judge_input = to_jsonable({
            "agent_input": agent_input,
            "agent_output": decision.output,
            "risk_result": {"violations": risk.violations,
                            "violation_count": risk.violation_count,
                            "turnover": risk.turnover, "modified": risk.modified},
            "executed_trades": exec_res.trades,
            "relevant_memories": relevant_memories,
            "market_snapshot": {"date": me.isoformat(), "n_eligible": len(eligible)},
            "constraints": self.constraints.as_dict(),
        })
        judge_res = self.judge.evaluate(judge_input)

        # ---- assemble step record ----
        hard_block_types = {"ineligible_ticker", "forced_sell_ignored",
                            "invalid_history_buy", "negative_weight"}
        risk_blocked = 1 if any(v["type"] in hard_block_types for v in risk.violations) else 0

        record = to_jsonable({
            "month_index": idx,
            "date": me.isoformat(),
            "action": decision.output.get("action", "rebalance"),
            "portfolio_value": round(exec_res.new_value, 2),
            "benchmark_value": round(bench_value, 2),
            "cash_weight": round(exec_res.new_holdings.get("CASH", 0.0), 4),
            "max_position_weight": round(M.max_position_weight(exec_res.new_holdings), 4),
            "sector_concentration": round(M.sector_concentration(exec_res.new_holdings, self.sectors), 4),
            "drawdown": round(self._running_drawdown(exec_res.new_value), 4),
            "turnover": round(exec_res.turnover, 4),
            "transaction_cost": round(exec_res.transaction_cost, 2),
            "transaction_count": exec_res.transaction_count,
            "buy_count": exec_res.buy_count,
            "sell_count": exec_res.sell_count,
            "forced_sell_count": exec_res.forced_sell_count,
            "new_entry_buy_count": exec_res.new_entry_buy_count,
            "changed_positions_count": exec_res.changed_positions_count,
            "forced_sell_not_executed": still_held_forced,
            "same_asset_flip_count": flips,
            "new_entrants_considered": sorted(new_entrants),
            "forced_sells": sorted(forced_sells),
            "holdings": {t: round(w, 4) for t, w in exec_res.new_holdings.items()},
            "trades": exec_res.trades,
            # agent behavior
            "agent_source": decision.source,
            "used_fallback": decision.used_fallback,
            "invalid_json_count": decision.invalid_json_count,
            "retry_count": decision.retry_count,
            "risk_blocked": risk_blocked,
            "risk_modified": 1 if risk.modified else 0,
            "constraint_violation_count": risk.violation_count,
            "risk_violations": risk.violations,
            "rationale": decision.output.get("rationale", []),
            # judge
            "judge": judge_res.output,
            "unsupported_claim_count": judge_res.unsupported_claim_count,
            "judge_warning_count": judge_res.warning_count,
            "judge_critical_count": judge_res.critical_count,
            # llm usage
            "latency_ms": round((decision.llm_usage.get("latency_ms", 0.0)
                                 + judge_res.llm_usage.get("latency_ms", 0.0)), 1),
            "cost_usd": round((decision.llm_usage.get("cost_usd", 0.0)
                               + judge_res.llm_usage.get("cost_usd", 0.0)), 6),
            "tokens": int(decision.llm_usage.get("total_tokens", 0)
                          + judge_res.llm_usage.get("total_tokens", 0)),
        })

        # ---- alerts ----
        fired = alert_engine.evaluate(record, self.history)
        for a in fired:
            self.monitor.record_alert(idx, me.isoformat(), a["severity"], a["type"], a["message"])

        # ---- persist + trace ----
        self.monitor.record_step(idx, me.isoformat(), record)
        self.tracer.log_month(
            name="monthly_rebalance",
            inputs={"agent_input": agent_input},
            outputs={"agent_output": decision.output, "judge_output": judge_res.output},
            metadata={
                "simulation_id": self.state.simulation_id,
                "simulated_month": idx,
                "agent_version": config.LLM.agent_version,
                "judge_version": config.LLM.judge_version,
                "prompt_version": config.LLM.prompt_version,
                "model_version": config.LLM.model,
                "universe_name": "sp500_sample",
                "retry_count": decision.retry_count,
                "invalid_json_count": decision.invalid_json_count,
                "tokens": record["tokens"],
                "cost_usd": record["cost_usd"],
                "latency_ms": record["latency_ms"],
            },
        )

        with self._lock:
            self.history.append(record)
            self.state.month_index = idx + 1
            self.state.current_date = me.isoformat()
            if self.state.month_index >= min(self.state.total_months, len(self.month_ends)):
                self.state.finished = True
                self.state.running = False

        # ---- checkpoint this completed month (point-in-time save) ----
        self.checkpointer.save(idx, self._snapshot(record["portfolio_value"]))
        return record

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def _snapshot(self, portfolio_value: float | None = None) -> dict:
        """Serializable runtime state for a checkpoint."""
        with self._lock:
            st = self.state
            return {
                "simulation_id": st.simulation_id,
                "month_index": st.month_index,
                "current_date": st.current_date,
                "finished": st.finished,
                "seconds_per_month": st.seconds_per_month,
                "portfolio_value": portfolio_value,
                "portfolio": {"cash": self.portfolio.cash,
                              "shares": dict(self.portfolio.shares)},
                "benchmark_shares": self.benchmark_shares,
                "last_trade_sides": dict(self._last_trade_sides),
            }

    def _apply_snapshot(self, snap: dict) -> None:
        """Restore runtime state from a checkpoint dict."""
        with self._lock:
            self.portfolio = broker.Portfolio(
                cash=float(snap["portfolio"]["cash"]),
                shares={k: float(v) for k, v in snap["portfolio"]["shares"].items()},
            )
            self.benchmark_shares = float(snap.get("benchmark_shares", 0.0))
            self._last_trade_sides = dict(snap.get("last_trade_sides", {}))
            self.state.simulation_id = snap.get("simulation_id", self.state.simulation_id)
            self.state.month_index = int(snap.get("month_index", 0))
            self.state.current_date = snap.get("current_date")
            self.state.finished = bool(snap.get("finished", False))
            self.state.running = False
            self.history = self.monitor.steps()

    def resume_latest(self) -> bool:
        """Resume from the most recent checkpoint on disk (if any). Returns True
        if a checkpoint was loaded."""
        self._ensure_loaded()
        snap = self.checkpointer.load_latest()
        if not snap:
            return False
        self._apply_snapshot(snap)
        return True

    def restore_to(self, completed_month: int) -> bool:
        """Rewind the simulation to the end of `completed_month` (point in time).

        Truncates monitor steps/alerts and checkpoints after that month, so the
        run can continue forward from the restored state.
        """
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        with self._lock:
            self._thread = None
            self._stop.clear()
        self._ensure_loaded()
        snap = self.checkpointer.load(completed_month)
        if not snap:
            return False
        self.monitor.delete_after(completed_month)
        self.checkpointer.prune_after(completed_month)
        self._apply_snapshot(snap)
        return True

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _running_drawdown(self, current_value: float) -> float:
        vals = [s["portfolio_value"] for s in self.history] + [current_value]
        return M.max_drawdown(vals)

    def _benchmark_value(self, me: dt.date) -> float:
        px = self.benchmark_series.get(me)
        if px is None:
            return self.history[-1]["benchmark_value"] if self.history else config.STARTING_CAPITAL
        if self.benchmark_shares <= 0:
            self.benchmark_shares = config.STARTING_CAPITAL / px
        return self.benchmark_shares * px

    def _cutoff_date(self, me: dt.date) -> str:
        cutoff = pd.Timestamp(me) - pd.DateOffset(months=STALE_MEMORY_MONTHS)
        return cutoff.date().isoformat()

    def _write_memories(self, decision, me: dt.date) -> None:
        for cand in (decision.output.get("new_memory_candidates", []) or [])[:2]:
            self.memory.add(new_memory(
                type=cand.get("type", "strategy_lesson"),
                content=cand.get("content", ""),
                date_created=me.isoformat(),
                confidence=0.5,
            ))

    # ------------------------------------------------------------------ #
    # Dashboard state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict:
        with self._lock:
            st = self.state
            history = list(self.history)
        aggregates = self.monitor.aggregates()
        latest = history[-1] if history else None

        series = {
            "dates": [s["date"] for s in history],
            "portfolio_value": [s["portfolio_value"] for s in history],
            "benchmark_value": [s["benchmark_value"] for s in history],
        }
        return {
            "controls": {
                "running": st.running,
                "finished": st.finished,
                "month_index": st.month_index,
                "total_months": st.total_months,
                "progress": f"{st.month_index} / {st.total_months}",
                "current_date": st.current_date,
                "seconds_per_month": st.seconds_per_month,
                "simulation_id": st.simulation_id,
                "data_ready": self.data_ready,
            },
            "langsmith": self.tracer.status(),
            "llm": {"enabled": config.LLM.enabled, "model": config.LLM.model},
            "series": series,
            "aggregates": aggregates,
            "latest": latest,
            "alerts": self.monitor.alerts(limit=100),
            "memory": {
                "total": self.memory.count(),
                "stale": self.memory.count_by_status("stale"),
                "active": self.memory.count_by_status("active"),
            },
        }


def _num(v) -> float | None:
    try:
        if v is None or pd.isna(v):
            return None
        return round(float(v), 6)
    except Exception:
        return None
