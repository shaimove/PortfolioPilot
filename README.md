<div align="center">

# PortfolioPilot

### A monitored, observable agentic system for portfolio decision-making

*A local paper-trading / backtesting platform where an **LLM portfolio agent** makes monthly S&P 500 decisions over 10 years — wrapped in deterministic guardrails, **LLM-as-a-judge** evaluation, LangSmith tracing, and a live observability dashboard.*

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![LangSmith](https://img.shields.io/badge/Observability-LangSmith-1C3C3C)
![MCP](https://img.shields.io/badge/MCP-Server-6E56CF)
![Tests](https://img.shields.io/badge/tests-63%20passing-2ea44f)

</div>

> ⚠️ **Disclaimer.** This is a **simulation and observability project** — **not financial advice and not a real trading system.** All trades are simulated ("paper") against locally cached historical data.

---

## Why this project

Shipping an LLM agent is easy. Making it **trustworthy** in a long, sequential decision loop is the hard part. PortfolioPilot is built around that exact problem: it puts a single LLM portfolio agent inside a realistic 120-step backtest and surrounds it with the machinery you'd want in production —

- **deterministic guardrails** so the model can never push the system into an illegal state,
- an **LLM-as-a-judge** that audits every decision for hallucinations and inconsistencies,
- **two monitoring layers** (LangSmith tracing + a local metrics/alerting monitor),
- **point-in-time correctness** (no look-ahead bias, real S&P 500 membership changes),
- **crash-safe checkpointing** and a clean, framework-free dashboard.

It's a focused demonstration of **agent reliability, evaluation, and observability** — not another chatbot.

## What this project demonstrates

| Area | In this repo |
|---|---|
| **Agentic system design** | One LLM decision agent with strict-JSON I/O, retries, and a deterministic fallback. |
| **LLM evaluation** | An LLM-as-a-judge scoring groundedness, hallucination risk, constraint awareness, memory use, and consistency — every month. |
| **Observability** | LangSmith tracing (latency/tokens/cost/versions) + a local monitor with financial, agent-behavior, and judge metrics, plus an incident timeline. |
| **Guardrails / safety** | A deterministic risk engine that validates and *repairs* every decision; forced-sell handling; long-only / cap / turnover / cash constraints. |
| **Data engineering** | Local-first ingestion → Parquet/DuckDB; point-in-time universe; no-look-ahead monthly feature engine. |
| **Schema & contracts** | **Pydantic v2** validation for agent and judge outputs (JSON-Schema derived from the models). |
| **Tooling / interop** | An **MCP server** exposing the deterministic toolset so external agents can consume it. |
| **Reliability** | SQLite-backed checkpointing with resume + point-in-time rewind. |
| **Testing** | 63 deterministic, offline tests (risk, execution, no-look-ahead, judge, membership changes, MCP, schema, checkpoints). |
| **Product polish** | A white, professional, dependency-free dashboard (FastAPI + vanilla JS). |

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
  │ point-in-time universe → market snapshot → memory retrieval →                  │
  │ DECISION AGENT (LLM or deterministic, strict JSON + retry) →                   │
  │ RISK ENGINE (validate/repair) → BROKER (simulated execution + costs) →         │
  │ benchmark update → memory writes → JUDGE (LLM or deterministic) →              │
  │ LangSmith trace + local monitor + alerts + checkpoint → dashboard state        │
  └───────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                      FastAPI server  ◄──►  white dashboard (HTML/CSS/JS)
                            ▲
                            └── MCP server exposes the deterministic tools
```

**Design principles enforced (and tested):** no online APIs in the loop · no look-ahead bias · no raw price history sent to the LLM · the judge never trades · the system runs fully **without** any LLM or LangSmith via deterministic fallbacks. Deep dive: **[`RELIABILITY.md`](RELIABILITY.md)**.

---

## Quickstart

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run it fully offline (deterministic, no network, no API keys)

```bash
python scripts/run_smoke_test.py     # ingest synthetic data → features → run 120 months
python -m portfoliopilot.server      # open http://127.0.0.1:8000 and press Start
```

### Run on real historical data (yfinance)

```bash
python scripts/ingest_constituents.py --years 10
python scripts/ingest_prices.py --years 10
python scripts/ingest_fundamentals.py
python scripts/build_features.py
python -m portfoliopilot.server
```

Optional integrations (the app runs without them): copy `.env.example` → `.env` and set `OPENAI_API_KEY` (enables the LLM agent + judge) and/or `LANGSMITH_API_KEY` (enables tracing).

---

## The dashboard

A white, professional, **non-Streamlit** UI (FastAPI + vanilla HTML/CSS/JS, self-contained canvas chart). Controls: **Start / Pause / Reset / Step** and seconds-per-simulated-month (default 30s, min 1s). Seven panels:

1. **Simulation Controls** — progress, date, LangSmith status
2. **Portfolio Overview** — value vs. benchmark, return, drawdown, turnover, cost drag
3. **Holdings & Transactions** — holdings, buys/sells, forced sells, new entrants
4. **Agent Behavior** — risk blocks/modifications, violations, invalid JSON, retries, flips
5. **Judge Monitoring** — groundedness, hallucination risk, unsupported claims, consistency
6. **Memory Health** — totals, stale usage, judge warnings/criticals
7. **Incident Timeline** — overtrading, risk, hallucination, latency/cost alerts

---

## Highlights

- **LLM-as-a-judge after every month.** Flags rationale that references facts not in the input (earnings, valuation, analyst ratings, news…), explanation/trade mismatches, ignored forced sells, and stale-memory use. Pydantic-validated, with hard structural checks always merged in.
- **Always-feasible risk engine.** Sum-to-1, long-only, per-asset & sector caps, cash bounds, turnover throttle (forced sells exempt), eligibility & invalid-history bans — the portfolio can never enter an illegal state.
- **Point-in-time S&P 500 universe.** Handles index additions/removals through time; forced sells on removal, new entrants become eligible. (Documented survivorship-bias fallback when point-in-time data isn't supplied.)
- **No look-ahead, by construction & test.** Features at month `t` use only data dated `≤ t` (`tests/test_no_lookahead.py`).
- **MCP server.** `python -m portfoliopilot.mcp_server` exposes 13 deterministic tools (features, universe, risk, execution, metrics, judge, memory, checkpoints).
- **Checkpointing.** SQLite-backed `SqliteSaver`; `resume_latest()` and `restore_to(month)` (rewind), exposed via `/api/checkpoints`, `/api/resume`, `/api/restore`.

---

## Tech stack

**Python 3.13** · pandas / numpy · **DuckDB** + Parquet · **FastAPI** + Uvicorn · **Pydantic v2** · **LangSmith** (optional) · **OpenAI** (optional) · **MCP** (Model Context Protocol) · SQLite · pytest · vanilla HTML/CSS/JS dashboard.

## Repository layout

```
portfoliopilot/
  data/        ingestion · providers · cache · universe
  features/    feature_engine (no look-ahead)
  agent/       decision_agent · prompts · schemas (Pydantic)
               research_agent · research_sources   ← experimental, not activated
  execution/   risk_engine · broker_simulator
  memory/      memory_store · retriever
  monitoring/  langsmith_tracing · local_monitor · judge · metrics · alerts
  simulation/  engine · checkpoint (SqliteSaver)
  server.py · mcp_server.py · llm.py · config.py · utils.py
dashboard/     index.html · styles.css · app.js
scripts/       ingest_* · build_features · run_smoke_test · run_research_agent
tests/         63 offline, deterministic tests
```

---

## Roadmap

- 🔬 **Fundamental research agent** *(staged, not yet activated — `agent/research_agent.py`)*: reads company fundamental reports and produces buy/hold/avoid recommendations with cited metrics and risks. Built with the same discipline as the decision agent (strict Pydantic JSON, deterministic fallback) and a provider abstraction ready for SEC EDGAR / FMP / Alpha Vantage. It will eventually feed *candidate idea hints* (never direct trades) into the pipeline, with its own judge coverage for the fundamental claims it introduces. Try it: `python scripts/run_research_agent.py`.
- Vector-based memory retrieval (the MVP is keyword/recency/ticker).
- Real point-in-time S&P 500 constituents dataset (removes the survivorship-bias fallback).
- Slippage / market-impact modeling beyond the flat 10 bps cost.

---

## Tests

```bash
pytest -q        # 63 tests, fully offline & deterministic (no network or LLM required)
```

See **[`RELIABILITY.md`](RELIABILITY.md)** for the full output-check pipeline, stability mechanisms, MCP tool catalog, and the complete monitoring-metric list.

---

<div align="center">
<sub>Built as a portfolio project on agentic systems — multi-agent orchestration, learning, and production-grade monitoring/evaluation. Not financial advice.</sub>
</div>
