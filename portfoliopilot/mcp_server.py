"""MCP server exposing PortfolioPilot's deterministic tools.

Every stock feature, risk check, execution calculation, metric, and the judge is
computed by deterministic Python. This module advertises those as Model Context
Protocol (MCP) tools so an external client/agent can *consume* them instead of
re-implementing (or hallucinating) the math.

Run (stdio transport):
    python -m portfoliopilot.mcp_server

Register in an MCP client (example):
    {
      "mcpServers": {
        "portfoliopilot": {
          "command": "/path/to/.venv/bin/python",
          "args": ["-m", "portfoliopilot.mcp_server"],
          "cwd": "/path/to/PortfolioPilot"
        }
      }
    }

All tools read only from the local cache (run the ingestion scripts first).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
from mcp.server.fastmcp import FastMCP

from . import config
from .agent import schemas
from .config import Constraints
from .data import cache
from .data.universe import Universe
from .execution import broker_simulator as broker
from .execution import risk_engine
from .features import feature_engine
from .features.feature_engine import FeatureStore
from .memory import retriever
from .memory.memory_store import MemoryStore
from .monitoring import metrics as M
from .monitoring.judge import Judge
from .monitoring.local_monitor import LocalMonitor
from .simulation.checkpoint import CheckpointSaver
from .utils import to_jsonable

mcp = FastMCP("portfoliopilot")


def _to_date(s: str) -> dt.date:
    return pd.to_datetime(s).date()


def _constraints(d: dict | None) -> Constraints:
    if not d:
        return config.DEFAULT_CONSTRAINTS
    base = Constraints().as_dict()
    base.update({k: v for k, v in d.items() if k in base})
    return Constraints(**base)


# --------------------------------------------------------------------------- #
# Universe / membership
# --------------------------------------------------------------------------- #
@mcp.tool()
def list_universe(month_end: str) -> dict:
    """Return the point-in-time S&P 500 membership for a given month-end (YYYY-MM-DD)."""
    uni = Universe.load()
    members = sorted(uni.members_on(_to_date(month_end)))
    return {"month_end": month_end, "n_members": len(members), "members": members}


@mcp.tool()
def membership_changes(prev_month_end: str, curr_month_end: str) -> dict:
    """Return tickers removed (forced sells) and added (new entrants) between two month-ends."""
    uni = Universe.load()
    prev, curr = _to_date(prev_month_end), _to_date(curr_month_end)
    return {
        "removed": sorted(uni.removed_between(prev, curr)),
        "added": sorted(uni.added_between(prev, curr)),
    }


@mcp.tool()
def list_month_ends() -> dict:
    """Return all simulation month-end dates available in the local cache."""
    uni = Universe.load()
    return {"month_ends": [d.isoformat() for d in uni.month_ends]}


# --------------------------------------------------------------------------- #
# Deterministic features
# --------------------------------------------------------------------------- #
@mcp.tool()
def get_stock_features(month_end: str) -> dict:
    """Return all deterministically-computed stock features for a month-end
    (from the cached monthly_features table)."""
    store = FeatureStore.load()
    feats = store.features_on(_to_date(month_end))
    rows = [] if feats.empty else feats.reset_index().to_dict(orient="records")
    return to_jsonable({"month_end": month_end, "count": len(rows), "features": rows})


@mcp.tool()
def compute_features_asof(month_end: str) -> dict:
    """Recompute stock features as-of a month-end directly from cached daily prices
    (no look-ahead: uses only data dated <= month_end). Deterministic."""
    prices = cache.read_parquet(config.PRICES_PARQUET)
    if prices is None or prices.empty:
        return {"error": "prices.parquet missing. Run ingest_prices first."}
    meta = cache.read_parquet(config.METADATA_PARQUET)
    sectors, valid = {}, {}
    if meta is not None and not meta.empty:
        if "sector" in meta.columns:
            sectors = {r.ticker: r.sector for r in meta.itertuples()
                       if pd.notna(getattr(r, "sector", None))}
        valid = {r.ticker: bool(getattr(r, "valid_history", True)) for r in meta.itertuples()}
    feats = feature_engine.compute_features_asof(prices, _to_date(month_end),
                                                 sectors=sectors, valid_flags=valid)
    rows = [] if feats.empty else feats.reset_index().to_dict(orient="records")
    return to_jsonable({"month_end": month_end, "count": len(rows), "features": rows})


# --------------------------------------------------------------------------- #
# Risk + execution
# --------------------------------------------------------------------------- #
@mcp.tool()
def validate_target_weights(
    target_weights: dict,
    current_weights: dict | None = None,
    eligible: list[str] | None = None,
    sectors: dict | None = None,
    valid_history: dict | None = None,
    forced_sells: list[str] | None = None,
    constraints: dict | None = None,
) -> dict:
    """Validate/repair proposed target weights with the deterministic risk engine.

    Returns the feasible weight vector, detected violations, turnover, and flags.
    """
    res = risk_engine.validate_and_repair(
        target_weights=target_weights,
        current_weights=current_weights or {risk_engine.CASH: 1.0},
        eligible=set(eligible or target_weights.keys()),
        sectors=sectors or {},
        valid_history=valid_history or {},
        forced_sells=set(forced_sells or []),
        constraints=_constraints(constraints),
    )
    return to_jsonable({
        "final_weights": res.final_weights,
        "violations": res.violations,
        "violation_count": res.violation_count,
        "turnover": res.turnover,
        "modified": res.modified,
    })


@mcp.tool()
def simulate_execution(
    target_weights: dict,
    cash: float,
    shares: dict,
    prices: dict,
    forced_sells: list[str] | None = None,
    prev_weights: dict | None = None,
    new_entrants: list[str] | None = None,
    cost_bps: float = config.TRANSACTION_COST_BPS,
) -> dict:
    """Simulate executing target weights against a portfolio (deterministic broker).

    Returns execution metrics (turnover, cost, counts), new holdings and value.
    """
    pf = broker.Portfolio(cash=float(cash), shares={k: float(v) for k, v in shares.items()})
    res = broker.execute(
        pf, target_weights=target_weights, prices={k: float(v) for k, v in prices.items()},
        forced_sells=set(forced_sells or []), prev_weights=prev_weights,
        new_entrants=set(new_entrants or []), cost_bps=cost_bps,
    )
    return to_jsonable({
        "transaction_count": res.transaction_count, "buy_count": res.buy_count,
        "sell_count": res.sell_count, "forced_sell_count": res.forced_sell_count,
        "new_entry_buy_count": res.new_entry_buy_count,
        "changed_positions_count": res.changed_positions_count,
        "turnover": res.turnover, "transaction_cost": res.transaction_cost,
        "cash_after": res.cash_after, "new_value": res.new_value,
        "new_holdings": res.new_holdings, "trades": res.trades,
    })


@mcp.tool()
def compute_portfolio_metrics(portfolio_values: list[float],
                              benchmark_values: list[float] | None = None) -> dict:
    """Deterministically compute total return, max drawdown, and (if a benchmark
    series is given) excess return."""
    out = {
        "total_return": M.total_return(portfolio_values),
        "max_drawdown": M.max_drawdown(portfolio_values),
    }
    if benchmark_values:
        out["benchmark_return"] = M.total_return(benchmark_values)
        out["excess_return"] = M.excess_return(portfolio_values, benchmark_values)
    return to_jsonable(out)


# --------------------------------------------------------------------------- #
# Judge (monitoring)
# --------------------------------------------------------------------------- #
@mcp.tool()
def run_judge(judge_input: dict) -> dict:
    """Run the LLM-as-judge (deterministic fallback if no LLM is configured) over a
    decision. Returns scores, issues, and approval. The judge never trades."""
    res = Judge().evaluate(judge_input)
    return to_jsonable({
        "output": res.output, "source": res.source,
        "unsupported_claim_count": res.unsupported_claim_count,
        "warning_count": res.warning_count, "critical_count": res.critical_count,
    })


@mcp.tool()
def validate_agent_output(agent_output: dict) -> dict:
    """Validate a candidate agent output against the strict Pydantic schema.
    Returns {valid, errors}."""
    errors = schemas.validate_agent_output(agent_output)
    return {"valid": not errors, "errors": errors}


# --------------------------------------------------------------------------- #
# Memory + monitoring + checkpoints
# --------------------------------------------------------------------------- #
@mcp.tool()
def retrieve_memories(as_of: str, tickers: list[str] | None = None,
                      keywords: str = "", k: int = 5) -> dict:
    """Retrieve top-k relevant memories as of a date (ticker/recency/keyword scored)."""
    store = MemoryStore()
    mems = retriever.retrieve(store, as_of=_to_date(as_of), tickers=tickers or [],
                              keywords=keywords, k=k)
    return {"memories": [m.as_dict() for m in mems]}


@mcp.tool()
def get_monitoring_metrics() -> dict:
    """Return the local monitor's rolled-up financial, agent-behavior, and judge metrics."""
    return to_jsonable(LocalMonitor().aggregates())


@mcp.tool()
def list_checkpoints() -> dict:
    """List point-in-time simulation checkpoints saved on disk."""
    return {"checkpoints": CheckpointSaver().list_checkpoints()}


def main() -> None:
    config.ensure_dirs()
    mcp.run()


if __name__ == "__main__":
    main()
