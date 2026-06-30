"""Incident detection.

Given the current step record (and a little history), emit alert dicts for the
incident timeline. Pure function, no storage.
"""
from __future__ import annotations

from ..config import DEFAULT_CONSTRAINTS

TURNOVER_LIMIT = DEFAULT_CONSTRAINTS.max_turnover_per_month
HALLUCINATION_LIMIT = 0.30
TXN_SPIKE_FACTOR = 3.0
LATENCY_SPIKE_MS = 15_000.0
COST_SPIKE_USD = 1.0


def evaluate(step: dict, history: list[dict]) -> list[dict]:
    alerts: list[dict] = []

    def add(severity: str, type_: str, message: str) -> None:
        alerts.append({"severity": severity, "type": type_, "message": message})

    # overtrading / turnover
    if step.get("turnover", 0.0) > TURNOVER_LIMIT + 1e-6:
        add("warning", "overtrading", f"Turnover {step['turnover']:.2%} exceeds "
            f"{TURNOVER_LIMIT:.0%} limit.")

    prior_txn = [s.get("transaction_count", 0) for s in history[-6:]]
    avg_txn = sum(prior_txn) / len(prior_txn) if prior_txn else 0.0
    if avg_txn > 0 and step.get("transaction_count", 0) > TXN_SPIKE_FACTOR * avg_txn:
        add("warning", "transaction_spike",
            f"Transaction count {step['transaction_count']} spiked vs recent avg {avg_txn:.1f}.")

    # risk
    if step.get("risk_blocked", 0) > 0:
        add("critical", "risk_blocked", "Risk engine blocked the agent decision.")
    if step.get("constraint_violation_count", 0) > 0:
        add("critical", "risk_violation",
            f"{step['constraint_violation_count']} constraint violation(s) detected.")

    # judge / hallucination
    judge = step.get("judge") or {}
    if (judge.get("hallucination_risk") or 0.0) > HALLUCINATION_LIMIT:
        add("critical", "hallucination",
            f"Hallucination risk {judge['hallucination_risk']:.2f} exceeds {HALLUCINATION_LIMIT}.")
    if step.get("unsupported_claim_count", 0) > 0:
        add("warning", "unsupported_claim",
            f"{step['unsupported_claim_count']} unsupported claim(s) in rationale.")

    # forced sell not executed
    if step.get("forced_sell_not_executed", 0) > 0:
        add("critical", "forced_sell_not_executed",
            "A forced sell from an index removal was not executed.")

    # same-asset flip across consecutive months
    if step.get("same_asset_flip_count", 0) > 0:
        add("warning", "same_asset_flip",
            f"{step['same_asset_flip_count']} asset(s) bought/sold in consecutive months.")

    # invalid JSON spike
    if step.get("invalid_json_count", 0) > 0:
        add("warning", "invalid_json", f"{step['invalid_json_count']} invalid JSON response(s).")

    # latency / cost spikes
    if (step.get("latency_ms") or 0.0) > LATENCY_SPIKE_MS:
        add("warning", "latency_spike", f"LLM latency {step['latency_ms']:.0f}ms is high.")
    if (step.get("cost_usd") or 0.0) > COST_SPIKE_USD:
        add("warning", "cost_spike", f"LLM cost ${step['cost_usd']:.2f} is high for one month.")

    return alerts
