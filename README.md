# PortfolioPilot

**A local paper-trading / backtesting system for a monitored portfolio agent.**

PortfolioPilot simulates monthly portfolio decisions over 10 years using S&P 500
**stocks** (not ETFs), tracks portfolio performance, monitors abnormal agent
behavior, and uses an **LLM-as-a-judge** to evaluate whether the portfolio
agent's decisions are grounded and non-hallucinated. Everything is shown in a
clean white-background dashboard (no Streamlit).

> ⚠️ **Disclaimer.** This is a **simulation and observability project**. It is
> **not financial advice** and **not a real trading system**. All trades are
> simulated ("paper") against locally cached historical data.

---

## Why it exists

It is a sandbox for studying *agent reliability* in a realistic-feeling
sequential decision loop:

- A single **LLM portfolio decision agent** proposes monthly target weights.
- Deterministic Python owns everything that must be correct: data, features,
  risk checks, execution, and metrics.
- An **LLM judge** reviews each decision *after the fact* for groundedness,
  hallucination, constraint awareness, and consistency.
- Two monitoring layers (**LangSmith** for tracing + a **local monitor** for
  financial/agent/judge metrics) surface problems on a live dashboard.

The system **runs even if LangSmith and the LLM are not configured**, using
deterministic fallback logic for both the agent and the judge.

---

## Architecture

```
                ┌─────────────────────────────────────────────────────────┐
ingestion  ───► │ local cache: prices.parquet / monthly_features.parquet / │
(yfinance /     │ constituents_monthly.parquet / fundamentals / metadata   │
 synthetic)     │ + portfoliopilot.duckdb                                  │
                └───────────────────────────┬─────────────────────────────┘
                                            │ (read-only during simulation)
                                            ▼
  ┌──────────────────────── monthly simulation loop (×120) ───────────────────────┐
  │ universe (point-in-time S&P 500) → market snapshot → memory retrieval →        │
  │ DECISION AGENT (LLM or deterministic, strict JSON + retry) →                   │
  │ RISK ENGINE (validate/repair) → BROKER (simulated execution + costs) →         │
  │ benchmark update → memory writes → JUDGE (LLM or deterministic) →              │
  │ LangSmith trace + local monitor + alerts → dashboard state                     │
  └───────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                      FastAPI server  ◄──►  white dashboard (HTML/CSS/JS)
```

### Key design rules (enforced)

- **No online APIs during the simulation loop.** All data is downloaded/cached
  *before* simulation; the loop reads only local Parquet/DuckDB/SQLite.
- **No look-ahead bias.** Features for month-end `t` use only data dated `≤ t`
  (guarded by `tests/test_no_lookahead.py`).
- **No raw price history to the LLM.** Python computes signals/returns/risk; the
  agent receives only compact JSON summaries and must output strict JSON.
- The **judge never trades** — it only evaluates decisions after they are made.

---

## Repo layout

```
portfoliopilot/
  data/        ingestion.py providers.py cache.py universe.py
  features/    feature_engine.py
  agent/       decision_agent.py prompts.py schemas.py
  execution/   risk_engine.py broker_simulator.py
  memory/      memory_store.py retriever.py
  monitoring/  langsmith_tracing.py local_monitor.py judge.py metrics.py alerts.py
  simulation/  engine.py
  server.py    llm.py config.py utils.py
dashboard/     index.html styles.css app.js
scripts/       ingest_constituents.py ingest_prices.py ingest_fundamentals.py
               build_features.py run_smoke_test.py
tests/
```

---

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional integrations (the app runs without them):

```bash
cp .env.example .env      # then fill in OPENAI_API_KEY / LANGSMITH_API_KEY
```

---

## Quickstart

### Option A — fully offline, deterministic (recommended first run)

No network needed. Uses a deterministic synthetic price generator and a
point-in-time membership table with built-in index additions/removals.

```bash
python scripts/run_smoke_test.py        # ingests, builds features, runs 120 months
python -m portfoliopilot.server         # then open http://127.0.0.1:8000
```

### Option B — real historical data (yfinance)

```bash
python scripts/ingest_constituents.py --years 10
python scripts/ingest_prices.py --years 10
python scripts/ingest_fundamentals.py
python scripts/build_features.py
python -m portfoliopilot.server
```

Open the dashboard, press **Start**, and watch the 120-month simulation run from
local cached data. Use **Pause**, **Reset**, **Step ▸**, and the **seconds /
month** control (default 30s; set to 1s for fast testing).

---

## Data ingestion

`scripts/ingest_*` download/build and **cache everything locally** before any
simulation:

| Output | Description |
|---|---|
| `data/raw/prices/<ticker>.parquet` | per-ticker daily adjusted OHLCV cache |
| `data/processed/prices.parquet` | consolidated long price table |
| `data/processed/constituents_monthly.parquet` | point-in-time S&P 500 membership |
| `data/processed/fundamentals.parquet` | sector (and room for more) |
| `data/processed/metadata.parquet` | per-ticker validity / first/last date |
| `data/processed/monthly_features.parquet` | per-stock monthly features |
| `data/portfoliopilot.duckdb` | DuckDB views over the processed parquet |

**Price data** uses yfinance for MVP adjusted OHLCV; adjusted prices are used for
returns. Missing prices, delistings, and short histories are tracked, and a
stock with invalid/insufficient history is **not eligible** to be bought.

**Fundamentals** use a provider abstraction (`data/providers.py`). The MVP ships
a deterministic stub (sector only) and an optional yfinance provider. The design
lets you drop in **SEC EDGAR Companyfacts, Alpha Vantage, FMP, or Tiingo** later
by subclassing `FundamentalsProvider`.

---

## S&P 500 membership handling

The universe is modeled as **point-in-time** monthly membership
(`constituents_monthly`: rows of `month_end, ticker`). At each rebalance:

- the tradable universe is the membership known at that date;
- a held stock that **leaves** the index is force-sold at the next rebalance
  (logged, and exempt from the turnover cap);
- stocks that **enter** the index become eligible candidates from the next
  rebalance (logged);
- all forced sells and new entrants are recorded as `index_membership_event`
  memories and shown on the dashboard.

### Survivorship-bias limitation (fallback)

Real point-in-time membership data is **not bundled**. Three builders exist:

- `pit` — load `data/raw/constituents/membership.csv` (`month_end,ticker`) if you
  provide a real point-in-time dataset (preferred);
- `synthetic` — deterministic additions/removals (used offline / in tests);
- `current` — a single static current-constituent list applied to every month.

> **The `current` fallback creates survivorship bias** — today's members are
> assumed to have always been members, which inflates results. It is the default
> only for the online MVP when no point-in-time CSV is present. The data model
> and all logic are point-in-time ready; replace the CSV to remove the bias.

---

## Monthly features (no look-ahead)

Per eligible stock/month: `ret_1m/3m/6m/12m`, `vol_3m/6m`, `drawdown`,
`ma_trend`, `volume_trend`, `current_price`, `valid_history`, `sector`.

Portfolio-level metrics (value, monthly/benchmark/excess return, max drawdown,
cash weight, turnover (1m / trailing 3m), transaction-cost drag, transaction
counts, changed positions, max position weight, sector concentration) are
computed during simulation in `monitoring/metrics.py` and the engine.

Benchmark: `^GSPC` (or SPY) is used **only as a benchmark**, never as a tradable
asset.

---

## Agent, risk & execution

- **Decision agent** (`agent/decision_agent.py`): receives compact JSON, returns
  strict JSON (schema-validated with retry on invalid JSON). On repeated invalid
  output or no LLM, it falls back to a deterministic momentum rule-based
  portfolio. Invalid-JSON and retry counts are logged.
- **Risk engine** (`execution/risk_engine.py`): enforces sum-to-1, long-only,
  per-asset cap, sector cap, cash bounds, turnover cap, eligibility, forced
  sells, and invalid-history bans — always returning a feasible weight vector.
- **Broker** (`execution/broker_simulator.py`): simulates trades on a
  share-based portfolio and applies **10 bps** transaction cost per traded
  dollar; reports turnover, costs, and full trade/holdings metrics.

---

## Monitoring

### LLM-as-judge design

The judge (`monitoring/judge.py`) reviews each decision using the agent input,
agent output, risk result, executed trades, memories, market snapshot, and
constraints. It scores groundedness, hallucination risk, constraint awareness,
memory use, and decision/output consistency, and emits structured `issues`.

It flags: rationale referencing facts **not present in the input** (earnings,
revenue, valuation, analyst ratings, news, fundamentals → `unsupported_claim`);
explanation/trade mismatches; **ignored forced sells** (critical); stale-memory
use; overconfidence; and constraint breaches the agent proposed. Hard structural
checks are always merged in, so an LLM judge can never hide a detectable problem.
Output is strict JSON (schema-validated; deterministic fallback otherwise).

### Metrics tracked

- **LangSmith** (optional): monthly rebalance trace with agent/judge I/O,
  latency, tokens, cost, errors, retry count, prompt/model/agent versions,
  `simulation_id`, `simulated_month`, `universe_name`. The app never crashes if
  LangSmith is missing.
- **Local monitor** (SQLite): financial metrics (value, benchmark, total/excess
  return, max drawdown, turnover, cost drag), agent behavior (transaction/buy/
  sell/forced-sell/changed counts, risk blocked/modified, constraint violations,
  same-asset flips, no-action rate, invalid JSON, retries), and judge metrics
  (the five scores, unsupported claims, warnings, criticals).
- **Alerts** (`monitoring/alerts.py`): overtrading, turnover > 30%, risk blocks,
  constraint violations, hallucination risk > 0.30, unsupported claims,
  same-asset flips, forced-sell-not-executed, invalid-JSON spikes, and
  latency/cost spikes — shown on the incident timeline.

---

## Dashboard

White, professional, non-Streamlit (FastAPI + vanilla HTML/CSS/JS, self-contained
canvas chart). Sections: Simulation Controls, Portfolio Overview, Holdings &
Transactions, Agent Behavior, Judge Monitoring, Memory Health, Incident Timeline.

Controls: Start / Pause / Reset / Step, and seconds-per-simulated-month
(default 30s, min 1s).

---

## Reliability, MCP tools & checkpoints

See **[`RELIABILITY.md`](RELIABILITY.md)** for the full detail. In short:

- **Deterministic everything-quantitative.** All stock features, risk, execution,
  and metrics are deterministic Python; the LLM only proposes weights/prose.
- **Output checks.** Agent responses are validated with a strict **Pydantic v2**
  schema (finite weights, valid action/lists), retried on failure, and fall back
  to a deterministic portfolio. The risk engine then guarantees a feasible state.
- **LLM-as-judge runs after every month**, with Pydantic-validated output and a
  deterministic fallback.
- **Checkpoints.** A SQLite-backed `SqliteSaver` writes a point-in-time
  checkpoint after every completed month (`data/checkpoints.sqlite`); the engine
  can `resume_latest()` or `restore_to(month)` (rewind), exposed via
  `/api/checkpoints`, `/api/resume`, `/api/restore`.
- **MCP server.** `python -m portfoliopilot.mcp_server` exposes the deterministic
  tools (features, universe, risk, execution, metrics, judge, memory, checkpoints)
  over the Model Context Protocol so external agents consume them instead of
  recomputing/hallucinating.

```bash
python -m portfoliopilot.mcp_server      # stdio MCP server (run ingestion first)
```

## Tests

```bash
pytest -q
```

Covers: 120-step simulation; index removal forces a sell; new entrant becomes
eligible; risk engine rejects invalid weights and caps positions; execution
turnover & transaction-cost effects; feature engine look-ahead safety; invalid
JSON → retry/fallback; judge flags unsupported claims and explanation/trade
mismatches; local monitor writes events; dashboard API returns valid state.

---

## Enabling LangSmith / the LLM

Set the variables in `.env` (see `.env.example`). With `OPENAI_API_KEY` set, the
agent and judge use the configured model (`PORTFOLIOPILOT_MODEL`, default
`gpt-4o-mini`). With `LANGSMITH_API_KEY` set, monthly runs are traced. The
dashboard shows the live status of both.

---

## Known limitations

- The bundled offline universe is a **curated ~50-name subset** of large S&P 500
  stocks (fast to synthesize); the logic is identical for the full list.
- Without a point-in-time CSV, the online MVP falls back to **current
  constituents → survivorship bias** (documented above).
- Synthetic offline prices are **not real**; they only exercise the pipeline.
  Use Option B for real history.
- Transaction costs are a flat 10 bps; no slippage/market-impact/borrow modeling.
- The MVP memory retriever is keyword/recency/ticker based — **not** a vector
  store (by design for v1).
