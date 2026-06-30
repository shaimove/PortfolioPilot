"""Portfolio decision agent.

Builds the compact agent input, calls the LLM (if configured) with strict-JSON
validation + retry, and falls back to a deterministic momentum rule-based
portfolio if the LLM is unavailable or keeps producing invalid output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import config, llm
from ..config import Constraints
from . import prompts, schemas


@dataclass
class AgentDecision:
    output: dict                       # validated agent output JSON
    used_fallback: bool = False
    retry_count: int = 0
    invalid_json_count: int = 0
    source: str = "fallback"           # "llm" | "fallback"
    llm_usage: dict = field(default_factory=dict)
    raw_responses: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Deterministic rule-based fallback (also the default when no LLM configured)
# --------------------------------------------------------------------------- #
def _candidate_score(c: dict) -> float:
    r12 = c.get("return_12m")
    r3 = c.get("return_3m")
    base = 0.0
    if r12 is not None:
        base += r12
    if r3 is not None:
        base += 0.5 * r3
    if c.get("trend") == "positive":
        base += 0.05
    # penalize high volatility a touch
    vol = c.get("volatility_3m")
    if vol is not None:
        base -= 0.25 * vol
    return base


def rule_based_decision(agent_input: dict, constraints: Constraints) -> dict:
    candidates = list(agent_input.get("eligible_candidates", []))
    forced = {f["ticker"].upper() for f in agent_input.get("forced_actions", [])
              if f.get("action") == "sell"}
    candidates = [c for c in candidates if c["ticker"].upper() not in forced]

    ranked = sorted(candidates, key=_candidate_score, reverse=True)

    target_cash = max(constraints.min_cash, min(0.10, constraints.max_cash))
    investable = 1.0 - target_cash
    # pick enough names that each stays under the per-asset cap
    min_names = max(1, int(round(investable / constraints.max_asset_weight)) + 2)
    n = min(len(ranked), max(min_names, 15))
    chosen = ranked[:n] if n > 0 else []

    weights: dict[str, float] = {}
    if chosen:
        per = min(constraints.max_asset_weight, investable / len(chosen))
        for c in chosen:
            weights[c["ticker"].upper()] = round(per, 6)
    # residual to cash
    stock_sum = sum(weights.values())
    weights["CASH"] = round(max(target_cash, 1.0 - stock_sum), 6)

    return {
        "action": "rebalance" if chosen else "raise_cash",
        "target_weights": weights,
        "rationale": [
            "Deterministic fallback: equal-weight the top momentum-ranked eligible "
            "stocks while respecting the per-asset cap.",
            "Hold a cash buffer to limit drawdown and keep turnover moderate.",
        ],
        "memories_used": [],
        "risk_notes": [
            "Forced sells from index removals are excluded from candidates.",
            "Weights respect max_asset_weight and cash bounds by construction.",
        ],
        "new_memory_candidates": [],
    }


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class DecisionAgent:
    def __init__(self, constraints: Constraints | None = None,
                 max_retries: int | None = None) -> None:
        self.constraints = constraints or config.DEFAULT_CONSTRAINTS
        self.max_retries = max_retries if max_retries is not None else config.LLM.max_retries

    def decide(self, agent_input: dict) -> AgentDecision:
        if not llm.available():
            out = rule_based_decision(agent_input, self.constraints)
            return AgentDecision(output=out, used_fallback=True, source="fallback")

        system = prompts.AGENT_SYSTEM_PROMPT
        user = prompts.build_agent_user_prompt(agent_input)

        invalid = 0
        retries = 0
        raws: list[str] = []
        errors: list[str] = []
        usage = {"latency_ms": 0.0, "total_tokens": 0, "cost_usd": 0.0}

        for attempt in range(self.max_retries + 1):
            resp = llm.chat_json(system, user)
            usage["latency_ms"] += resp.latency_ms
            usage["total_tokens"] += resp.total_tokens
            usage["cost_usd"] += resp.cost_usd
            if not resp.ok:
                errors.append(resp.error or "llm error")
                break  # LLM unusable -> fallback

            raws.append(resp.text)
            obj = llm.extract_json(resp.text)
            if obj is None:
                invalid += 1
                retries = attempt
                user = (prompts.build_agent_user_prompt(agent_input)
                        + "\n\nYour previous response was not valid JSON. Return STRICT JSON only.")
                continue

            obj = schemas.normalize_agent_weights(obj)
            errs = schemas.validate_agent_output(obj)
            if errs:
                invalid += 1
                retries = attempt
                errors.extend(errs)
                user = (prompts.build_agent_user_prompt(agent_input)
                        + "\n\nYour previous JSON failed validation: "
                        + "; ".join(errs[:5]) + ". Return STRICT, schema-valid JSON only.")
                continue

            return AgentDecision(
                output=obj, used_fallback=False, retry_count=attempt,
                invalid_json_count=invalid, source="llm",
                llm_usage=usage, raw_responses=raws, errors=errors,
            )

        # all attempts failed -> deterministic fallback
        out = rule_based_decision(agent_input, self.constraints)
        return AgentDecision(
            output=out, used_fallback=True, retry_count=retries,
            invalid_json_count=invalid, source="fallback",
            llm_usage=usage, raw_responses=raws, errors=errors,
        )
