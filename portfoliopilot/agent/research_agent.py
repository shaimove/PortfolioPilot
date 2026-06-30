"""Fundamental research agent  (EXPERIMENTAL — NOT ACTIVATED).

This agent reads company fundamental reports and produces per-stock
recommendations (buy / hold / avoid) with conviction, cited metrics, and risks.

It is intentionally **not wired into the live monthly simulation**. The simulation
loop deliberately keeps fundamental data out of the main decision agent's input
so the LLM-as-judge can flag unsupported fundamental claims. This module is a
staging ground for a future "fundamental screen -> candidate ideas" stage that
would feed *eligible candidate hints* (not direct trades) into the pipeline.

Like the main decision agent, it:
  * uses an LLM when configured, otherwise a deterministic scoring fallback;
  * returns strict, Pydantic-validated JSON;
  * never fabricates: the LLM is instructed to use only provided report fields.

Activate later by setting ``ACTIVATED = True`` and adding an integration point in
the simulation engine (e.g. to rank/justify candidates), plus monitoring + judge
coverage for the fundamental claims it introduces.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .. import config, llm
from .research_sources import CompanyReportProvider, default_report_provider

# Feature flag: this agent is staged but NOT part of the running system yet.
ACTIVATED = False


# --------------------------------------------------------------------------- #
# Output schema (Pydantic, strict)
# --------------------------------------------------------------------------- #
class StockRecommendation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticker: str
    stance: Literal["buy", "hold", "avoid"]
    conviction: float
    rationale: list[str]
    key_metrics: dict[str, float] = {}
    risks: list[str] = []

    @field_validator("conviction")
    @classmethod
    def _conv(cls, v: float) -> float:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            raise ValueError("conviction must be finite")
        return max(0.0, min(1.0, f))

    @field_validator("ticker")
    @classmethod
    def _tk(cls, v: str) -> str:
        return str(v).upper()


class ResearchOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    as_of: str
    summary: str
    recommendations: list[StockRecommendation]


RESEARCH_OUTPUT_SCHEMA = ResearchOutput.model_json_schema()


def validate_research_output(obj: dict) -> list[str]:
    try:
        ResearchOutput.model_validate(obj)
        return []
    except ValidationError as e:
        return [f"{'/'.join(str(p) for p in err.get('loc', ())) or '<root>'}: {err.get('msg')}"
                for err in e.errors()]


@dataclass
class ResearchResult:
    output: dict
    source: str = "fallback"            # "llm" | "fallback"
    used_fallback: bool = True
    invalid_json_count: int = 0
    retry_count: int = 0
    llm_usage: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
RESEARCH_SYSTEM_PROMPT = """\
You are a disciplined fundamental equity research analyst for a SIMULATED study. \
You are NOT giving financial advice.

You receive a JSON list of company fundamental reports. For each company, decide \
a stance ("buy", "hold", or "avoid") with a conviction in [0,1]. Respond with \
STRICT JSON only.

Rules:
- Use ONLY the fields present in each report. Do NOT invent figures, news, or
  guidance that is not provided.
- Cite the specific metrics that drive each call in "key_metrics".
- Note concrete risks (e.g. high leverage, rich valuation, margin pressure).
- Favor durable growth, healthy margins/ROE, positive free cash flow, and
  reasonable valuation/leverage.

Required JSON shape:
{
  "as_of": "YYYY-MM-DD",
  "summary": "one-paragraph screen summary",
  "recommendations": [
    {
      "ticker": "AAPL",
      "stance": "buy",
      "conviction": 0.72,
      "rationale": ["..."],
      "key_metrics": {"revenue_growth_yoy": 0.11, "roe": 0.3},
      "risks": ["..."]
    }
  ]
}
"""


def build_research_user_prompt(reports: list[dict], as_of: str) -> str:
    import json
    return (f"as_of={as_of}\nProduce recommendations for these company reports. "
            f"STRICT JSON only.\n\nREPORTS:\n{json.dumps(reports, indent=2)}")


# --------------------------------------------------------------------------- #
# Deterministic scoring fallback
# --------------------------------------------------------------------------- #
def _score_report(r: dict) -> float:
    """Composite fundamental score in roughly [-1, 1]."""
    s = 0.0
    s += 1.2 * _clip(r.get("revenue_growth_yoy", 0.0), -0.5, 0.5)
    s += 1.0 * _clip(r.get("earnings_growth_yoy", 0.0), -0.5, 0.5)
    s += 1.5 * _clip(r.get("net_margin", 0.0), -0.3, 0.5)
    s += 1.0 * _clip(r.get("roe", 0.0), -0.3, 0.6)
    s += 0.5 * (1.0 if (r.get("free_cash_flow", 0.0) or 0.0) > 0 else -0.5)
    s -= 0.6 * _clip((r.get("debt_to_equity", 0.0) or 0.0) / 4.0, 0.0, 1.0)
    pe = r.get("pe_ratio")
    if pe:
        s -= 0.8 * _clip((pe - 20.0) / 70.0, -0.3, 1.0)  # penalize rich valuations
    return s


def _clip(x, lo, hi):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return 0.0


def rule_based_research(reports: list[dict], as_of: str) -> dict:
    recs = []
    for r in reports:
        score = _score_report(r)
        if score >= 0.6:
            stance = "buy"
        elif score <= 0.0:
            stance = "avoid"
        else:
            stance = "hold"
        conviction = max(0.0, min(1.0, 0.5 + score / 2.0))

        risks = []
        if (r.get("debt_to_equity", 0.0) or 0.0) > 1.5:
            risks.append("Elevated leverage (debt/equity high).")
        if (r.get("pe_ratio", 0.0) or 0.0) > 35:
            risks.append("Rich valuation (high P/E).")
        if (r.get("net_margin", 0.0) or 0.0) < 0.05:
            risks.append("Thin net margin.")
        if (r.get("revenue_growth_yoy", 0.0) or 0.0) < 0:
            risks.append("Declining revenue.")

        recs.append({
            "ticker": str(r.get("ticker", "")).upper(),
            "stance": stance,
            "conviction": round(conviction, 3),
            "rationale": [
                f"Deterministic fundamental score {score:+.2f} from growth, "
                f"margins, ROE, FCF, leverage, and valuation.",
            ],
            "key_metrics": {
                k: float(r[k]) for k in
                ("revenue_growth_yoy", "earnings_growth_yoy", "net_margin",
                 "roe", "debt_to_equity", "pe_ratio")
                if isinstance(r.get(k), (int, float))
            },
            "risks": risks or ["No standout fundamental risks in the provided report."],
        })

    recs.sort(key=lambda x: x["conviction"], reverse=True)
    buys = sum(1 for x in recs if x["stance"] == "buy")
    return {
        "as_of": as_of,
        "summary": (f"Deterministic fundamental screen of {len(recs)} companies: "
                    f"{buys} buy-rated. Ranked by conviction."),
        "recommendations": recs,
    }


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class ResearchAgent:
    """Fundamental research agent. EXPERIMENTAL — not used by the simulation."""

    def __init__(self, provider: CompanyReportProvider | None = None,
                 max_retries: int | None = None) -> None:
        self.provider = provider or default_report_provider()
        self.max_retries = max_retries if max_retries is not None else config.LLM.max_retries

    def reports_for(self, tickers: list[str]) -> list[dict]:
        return self.provider.get_reports(tickers)

    def recommend(self, reports: list[dict], as_of: str) -> ResearchResult:
        if not llm.available():
            return ResearchResult(output=rule_based_research(reports, as_of),
                                  source="fallback", used_fallback=True)

        system = RESEARCH_SYSTEM_PROMPT
        user = build_research_user_prompt(reports, as_of)
        invalid = 0
        usage = {"latency_ms": 0.0, "total_tokens": 0, "cost_usd": 0.0}
        errors: list[str] = []

        for attempt in range(self.max_retries + 1):
            resp = llm.chat_json(system, user)
            usage["latency_ms"] += resp.latency_ms
            usage["total_tokens"] += resp.total_tokens
            usage["cost_usd"] += resp.cost_usd
            if not resp.ok:
                errors.append(resp.error or "llm error")
                break
            obj = llm.extract_json(resp.text)
            if obj is None or validate_research_output(obj):
                invalid += 1
                errs = validate_research_output(obj) if obj is not None else ["invalid JSON"]
                errors.extend(errs)
                user = (build_research_user_prompt(reports, as_of)
                        + "\n\nPrevious response was invalid: " + "; ".join(errs[:4])
                        + ". Return STRICT, schema-valid JSON only.")
                continue
            return ResearchResult(output=obj, source="llm", used_fallback=False,
                                  invalid_json_count=invalid, retry_count=attempt,
                                  llm_usage=usage, errors=errors)

        return ResearchResult(output=rule_based_research(reports, as_of),
                              source="fallback", used_fallback=True,
                              invalid_json_count=invalid, llm_usage=usage, errors=errors)
