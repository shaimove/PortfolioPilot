from portfoliopilot.monitoring.judge import Judge, deterministic_judge


def _base_input():
    return {
        "agent_input": {
            "eligible_candidates": [
                {"ticker": "AAPL", "sector": "Technology", "return_3m": 0.05,
                 "return_12m": 0.18, "volatility_3m": 0.2, "drawdown": -0.05, "trend": "positive"},
                {"ticker": "MSFT", "sector": "Technology", "return_3m": 0.03,
                 "return_12m": 0.10, "volatility_3m": 0.18, "drawdown": -0.03, "trend": "positive"},
            ],
            "forced_actions": [],
        },
        "agent_output": {
            "action": "rebalance",
            "target_weights": {"AAPL": 0.06, "MSFT": 0.06, "CASH": 0.88},
            "rationale": ["Keep diversified large-cap exposure with positive momentum."],
            "memories_used": [],
            "risk_notes": [],
            "new_memory_candidates": [],
        },
        "risk_result": {"violations": [], "violation_count": 0},
        "executed_trades": [],
        "relevant_memories": [],
        "constraints": {"max_asset_weight": 0.08, "max_sector_weight": 0.30,
                        "max_turnover_per_month": 0.30, "min_cash": 0.02,
                        "max_cash": 0.30, "long_only": True},
    }


def test_judge_flags_unsupported_claim():
    ji = _base_input()
    ji["agent_output"]["rationale"] = [
        "AAPL earnings improved sharply and analyst ratings were upgraded."
    ]
    out = deterministic_judge(ji)
    types = {i["type"] for i in out["issues"]}
    assert "unsupported_claim" in types
    assert out["_unsupported_claim_count"] >= 1
    assert out["hallucination_risk"] > 0


def test_judge_flags_explanation_trade_mismatch():
    ji = _base_input()
    # praise MSFT but give it zero weight
    ji["agent_output"]["target_weights"] = {"AAPL": 0.08, "CASH": 0.92}
    ji["agent_output"]["rationale"] = ["We strongly favor MSFT for its momentum."]
    out = deterministic_judge(ji)
    types = {i["type"] for i in out["issues"]}
    assert "decision_consistency" in types
    assert out["decision_consistency_score"] < 1.0


def test_judge_flags_ignored_forced_sell():
    ji = _base_input()
    ji["agent_input"]["forced_actions"] = [
        {"ticker": "XYZ", "reason": "removed_from_sp500", "action": "sell"}
    ]
    ji["agent_output"]["target_weights"] = {"XYZ": 0.05, "AAPL": 0.05, "CASH": 0.90}
    out = deterministic_judge(ji)
    types = {i["type"] for i in out["issues"]}
    assert "forced_sell_ignored" in types
    assert out["approved_for_monitoring"] is False


def test_judge_clean_decision_approved():
    out = deterministic_judge(_base_input())
    assert out["approved_for_monitoring"] is True
    assert out["hallucination_risk"] <= 0.30


def test_judge_wrapper_counts_and_schema():
    res = Judge().evaluate(_base_input())
    assert set([
        "groundedness_score", "hallucination_risk", "constraint_awareness_score",
        "memory_use_score", "decision_consistency_score", "issues",
        "approved_for_monitoring",
    ]).issubset(res.output.keys())
    assert isinstance(res.warning_count, int)
    assert isinstance(res.critical_count, int)
