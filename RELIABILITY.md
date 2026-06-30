# PortfolioPilot — Reliability, Output Checks & Monitoring

This document details how PortfolioPilot keeps an LLM agent *trustworthy* inside a
sequential decision loop: what is deterministic vs. model-driven, every output
check the agent's response passes through, the stability mechanisms (fallbacks,
checkpoints, retries), the MCP tool surface, and the full monitoring metric
catalog.

> Companion to the main [`README.md`](README.md). This is a simulation /
> observability project — **not financial advice, not a real trading system.**

---

## 1. Deterministic vs. LLM responsibilities

**All quantitative work is deterministic Python. The LLM only proposes weights
and prose, and is judged by deterministic + LLM checks afterward.**

| Concern | Owner | Module |
|---|---|---|
| Daily prices, caching | deterministic | `data/ingestion.py`, `data/cache.py` |
| Point-in-time S&P 500 universe | deterministic | `data/universe.py` |
| **Every stock feature** (returns, vols, drawdown, MA/volume trend, validity) | **deterministic** | `features/feature_engine.py` |
| Risk validation / repair | deterministic | `execution/risk_engine.py` |
| Simulated execution, costs, turnover | deterministic | `execution/broker_simulator.py` |
| Portfolio metrics | deterministic | `monitoring/metrics.py` |
| Target weights + rationale | **LLM agent** (deterministic fallback) | `agent/decision_agent.py` |
| Decision review / scoring | **LLM judge** (deterministic fallback) | `monitoring/judge.py` |

Because features are 100% deterministic, they are exposed as **MCP tools** (§5)
so any external agent consumes the same numbers rather than recomputing or
hallucinating them. The LLM never receives raw price history — only compact
summaries.

---

## 2. Output checks on the agent's response

Every agent decision flows through a layered gauntlet. Each layer is independent,
so a failure at one layer is caught by the next.

```
LLM output ─► JSON extract ─► Pydantic schema ─► retry loop ─► deterministic
              (llm.py)         (agent/schemas)   (decision_agent)  fallback
                                     │
                                     ▼
                          Risk engine validate/repair ─► Broker execution
                                     │
                                     ▼
                          LLM-as-judge review (every month)
```

### 2.1 Strict JSON + Pydantic schema validation

The agent must return strict JSON. It is parsed and validated against a
**Pydantic v2 model** (`agent/schemas.py::AgentOutput`):

- `action` ∈ `{rebalance, hold, raise_cash, no_action}` (rejected otherwise);
- `target_weights` is a non-empty map of ticker → **finite** float (NaN/Inf
  rejected); tickers are upper-cased;
- `rationale` is a non-empty list of strings;
- `memories_used`, `risk_notes`, `new_memory_candidates` are validated
  (each memory candidate needs `type` + `content`);
- unknown extra fields are ignored, not fatal.

JSON-Schema views (`AGENT_OUTPUT_SCHEMA`, `JUDGE_OUTPUT_SCHEMA`) are *derived
from* the Pydantic models so documentation and MCP descriptions never drift.

### 2.2 Retry on invalid output

On invalid JSON or schema failure the agent retries up to `LLM.max_retries`
(default 2), feeding the specific validation errors back into the prompt. Counts
are recorded: `invalid_json_count`, `retry_count`.

### 2.3 Deterministic fallback

If the LLM is not configured, errors out, or never produces valid output, the
agent falls back to a deterministic momentum rule-based portfolio
(`rule_based_decision`) that is valid by construction. The system therefore
**always produces a feasible decision**, with or without an LLM.

### 2.4 Risk engine (always-feasible repair)

The (post-agent) weights are validated and repaired by `risk_engine`:

- weights sum to 1.0; no negative weights; long-only;
- cash within `[min_cash, max_cash]`;
- no stock above `max_asset_weight`; no sector above `max_sector_weight`;
- turnover below `max_turnover_per_month` (**forced sells are exempt** —
  index-removal liquidations always complete);
- only stocks eligible in that month's universe; stocks with invalid history
  cannot be bought; index-removed stocks are forced to zero.

The engine **always returns a feasible vector** and records each intervention as
a violation (used for `risk_modified`, `risk_blocked`, `constraint_violation_count`).

### 2.5 LLM-as-judge — runs after **every** month

After execution, the judge (`monitoring/judge.py`) reviews the decision on every
single month (unconditionally in `simulation/engine.py::step`). It scores
groundedness, hallucination risk, constraint awareness, memory use, and
decision/output consistency, and flags issues:

- **`unsupported_claim`** — rationale references facts absent from the input
  (earnings, revenue, valuation, analyst ratings, news, guidance, fundamentals);
- **`forced_sell_ignored`** (critical) — a forced sell still has weight;
- **`decision_consistency`** — a ticker praised in prose has ~0 weight;
- **`stale_memory`** — a stale memory was used;
- **`constraint_awareness`** — agent proposed an infeasible weight (auto-repaired);
- **`overconfidence`** — excessive new-memory claims.

The judge output is itself **Pydantic-validated**; hard structural checks
(forced-sell / constraints) are always merged in, so an LLM judge can never hide
a detectable problem. Deterministic fallback runs when no LLM is configured.

---

## 3. Stability mechanisms

| Mechanism | What it guarantees |
|---|---|
| Deterministic agent fallback | A valid decision every month, no-LLM or LLM-down. |
| Deterministic judge fallback | Monitoring continues without an LLM. |
| Risk-engine repair | The portfolio can never enter an infeasible/illegal state. |
| Retry loop | Transient invalid JSON self-heals before falling back. |
| **No network in the loop** | Simulation reads only local Parquet/DuckDB/SQLite. |
| **No look-ahead** | Features at month `t` use only data ≤ `t` (`test_no_lookahead`). |
| LangSmith optional | App never crashes when tracing is missing/unreachable. |
| **Checkpoints** (§4) | Crash-safe resume + point-in-time rewind. |
| Reproducibility | Synthetic data is seeded per-ticker → identical runs. |

---

## 4. Checkpointing (point-in-time save / restore)

`simulation/checkpoint.py` writes a small JSON checkpoint after **every completed
month** to `data/checkpoints/month_XXX.json` (+ `latest.json`). Heavier monitor /
memory state already lives in SQLite.

A checkpoint captures: `simulation_id`, `month_index`, `current_date`, portfolio
(`cash` + `shares`), `benchmark_shares`, and `last_trade_sides`.

Engine API:

- `resume_latest()` — continue a run after a process restart.
- `restore_to(completed_month)` — **rewind** to a past month: truncates monitor
  steps/alerts and prunes later checkpoints, so the run continues forward from
  that exact state (deterministically reproducible).

HTTP API:

```
GET  /api/checkpoints          -> list saved months
POST /api/resume               -> resume from latest checkpoint
POST /api/restore  {completed_month: N}   -> rewind to end of month N
```

`test_checkpoint.py` verifies per-month saving, rewind, truncation, and that
stepping forward after a rewind reproduces the same portfolio value.

---

## 5. MCP server — deterministic tools to consume

`python -m portfoliopilot.mcp_server` starts a Model Context Protocol server
(stdio) exposing the deterministic toolset so an external client/agent uses the
same engine instead of re-deriving numbers.

| Tool | Purpose |
|---|---|
| `list_universe(month_end)` | Point-in-time S&P 500 membership. |
| `membership_changes(prev, curr)` | Forced sells (removed) + new entrants (added). |
| `list_month_ends()` | All simulation month-ends. |
| `get_stock_features(month_end)` | Cached deterministic features for a month. |
| `compute_features_asof(month_end)` | Recompute features from prices (no look-ahead). |
| `validate_target_weights(...)` | Risk-engine validate/repair. |
| `simulate_execution(...)` | Broker execution metrics + new holdings. |
| `compute_portfolio_metrics(...)` | Total/excess return, max drawdown. |
| `run_judge(judge_input)` | LLM-as-judge review (deterministic fallback). |
| `validate_agent_output(obj)` | Strict Pydantic validation of an agent response. |
| `retrieve_memories(...)` | Top-k memory retrieval. |
| `get_monitoring_metrics()` | Rolled-up local monitor metrics. |
| `list_checkpoints()` | Saved point-in-time checkpoints. |

Example client registration:

```json
{
  "mcpServers": {
    "portfoliopilot": {
      "command": "/abs/path/PortfolioPilot/.venv/bin/python",
      "args": ["-m", "portfoliopilot.mcp_server"],
      "cwd": "/abs/path/PortfolioPilot"
    }
  }
}
```

(Run the ingestion scripts first so the tools have local data to read.)

---

## 6. Monitoring metric catalog

### 6.1 Financial (local monitor)

`portfolio_value`, `benchmark_value`, `total_return`, `benchmark_return`,
`excess_return`, `max_drawdown`, `turnover_avg`, `transaction_cost_drag`.

### 6.2 Agent behavior

`transaction_count`, `buy_count`, `sell_count`, `forced_sell_count`,
`changed_positions_count`, `risk_blocked_count`, `risk_modified_count`,
`constraint_violation_count`, `same_asset_flip_count`, `no_action_rate`,
`invalid_json_count`, `retry_count`.

### 6.3 Judge

`groundedness_score`, `hallucination_risk`, `constraint_awareness_score`,
`memory_use_score`, `decision_consistency_score`, `unsupported_claim_count`,
`judge_warning_count`, `judge_critical_count`.

### 6.4 LangSmith trace fields (when enabled)

Monthly rebalance trace with agent/judge I/O, `latency_ms`, `tokens`, `cost_usd`,
errors, `retry_count`, `prompt_version`, `model_version`, `agent_version`,
`judge_version`, `simulation_id`, `simulated_month`, `universe_name`.

### 6.5 Alerts / incident timeline

Overtrading, turnover > 30%, `risk_blocked > 0`, `constraint_violation > 0`,
`hallucination_risk > 0.30`, `unsupported_claim > 0`, same-asset flips,
forced-sell-not-executed, invalid-JSON spike, transaction-count spike, latency
spike, cost spike.

---

## 7. Test coverage map

| Area | Test file |
|---|---|
| Risk validation, caps, turnover, sector, eligibility | `test_risk_engine.py` |
| Execution: turnover, costs, forced sells, growth | `test_execution_engine.py` |
| Features present + as-of correctness | `test_feature_engine.py` |
| **No look-ahead** | `test_no_lookahead.py` |
| Judge flags (unsupported / mismatch / forced sell) | `test_judge.py` |
| Agent invalid-JSON retry + fallback | `test_decision_agent.py` |
| **Pydantic output validation** | `test_pydantic_schema.py` |
| **Checkpoint save / rewind** | `test_checkpoint.py` |
| **MCP tools** | `test_mcp_tools.py` |
| 120-step run, membership changes, monitor, dashboard API | `test_sp500_membership_changes.py` |
| Dashboard backend | `test_server.py` |

Run all: `pytest -q` (offline, deterministic; no network or LLM required).
