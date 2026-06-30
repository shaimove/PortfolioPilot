"""Agent invalid-JSON retry + deterministic fallback behavior."""
import json

from portfoliopilot import llm
from portfoliopilot.agent.decision_agent import DecisionAgent, rule_based_decision
from portfoliopilot.config import Constraints
from portfoliopilot.llm import LLMResponse


def _agent_input():
    return {
        "date": "2021-06-30",
        "portfolio_state": {"value": 100000, "cash_weight": 1.0, "positions": {},
                            "drawdown": 0.0, "turnover_last_3m": 0.0},
        "constraints": Constraints().as_dict(),
        "eligible_candidates": [
            {"ticker": "AAPL", "sector": "Technology", "current_weight": 0.0,
             "return_3m": 0.05, "return_12m": 0.2, "volatility_3m": 0.2,
             "drawdown": -0.05, "trend": "positive"},
            {"ticker": "MSFT", "sector": "Technology", "current_weight": 0.0,
             "return_3m": 0.04, "return_12m": 0.15, "volatility_3m": 0.18,
             "drawdown": -0.04, "trend": "positive"},
            {"ticker": "JPM", "sector": "Financials", "current_weight": 0.0,
             "return_3m": 0.02, "return_12m": 0.1, "volatility_3m": 0.22,
             "drawdown": -0.06, "trend": "positive"},
        ],
        "forced_actions": [],
        "relevant_memories": [],
    }


def test_rule_based_decision_is_valid():
    out = rule_based_decision(_agent_input(), Constraints())
    w = out["target_weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v >= 0 for v in w.values())
    assert "CASH" in w
    assert all(v <= Constraints().max_asset_weight + 1e-9 for k, v in w.items() if k != "CASH")


def test_no_llm_uses_fallback(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: False)
    agent = DecisionAgent(Constraints())
    decision = agent.decide(_agent_input())
    assert decision.used_fallback is True
    assert decision.source == "fallback"


def test_invalid_json_triggers_retry_then_fallback(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    calls = {"n": 0}

    def fake_chat(system, user, **kw):
        calls["n"] += 1
        return LLMResponse(text="not json at all", ok=True, total_tokens=10)

    monkeypatch.setattr(llm, "chat_json", fake_chat)
    agent = DecisionAgent(Constraints(), max_retries=2)
    decision = agent.decide(_agent_input())
    assert decision.invalid_json_count >= 1
    assert decision.used_fallback is True
    assert calls["n"] == 3  # initial + 2 retries


def test_invalid_then_valid_json_uses_llm(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    seq = [
        "garbage",
        json.dumps({
            "action": "rebalance",
            "target_weights": {"AAPL": 0.06, "MSFT": 0.06, "CASH": 0.88},
            "rationale": ["diversified"],
        }),
    ]
    state = {"i": 0}

    def fake_chat(system, user, **kw):
        text = seq[min(state["i"], len(seq) - 1)]
        state["i"] += 1
        return LLMResponse(text=text, ok=True, total_tokens=10)

    monkeypatch.setattr(llm, "chat_json", fake_chat)
    agent = DecisionAgent(Constraints(), max_retries=2)
    decision = agent.decide(_agent_input())
    assert decision.source == "llm"
    assert decision.used_fallback is False
    assert decision.retry_count >= 1
