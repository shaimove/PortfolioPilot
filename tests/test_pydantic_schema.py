"""Pydantic validation of agent (and judge) output."""
import pytest
from pydantic import ValidationError

from portfoliopilot.agent import schemas
from portfoliopilot.agent.schemas import AgentOutput, parse_agent_output


def _valid():
    return {
        "action": "rebalance",
        "target_weights": {"aapl": 0.06, "msft": 0.06, "CASH": 0.88},
        "rationale": ["Diversified large-cap exposure."],
    }


def test_valid_output_parses_and_uppercases_tickers():
    model = parse_agent_output(_valid())
    assert isinstance(model, AgentOutput)
    assert "AAPL" in model.target_weights and "aapl" not in model.target_weights
    assert schemas.validate_agent_output(_valid()) == []


def test_invalid_action_rejected():
    obj = _valid()
    obj["action"] = "yolo"
    errors = schemas.validate_agent_output(obj)
    assert errors and any("action" in e for e in errors)


def test_missing_required_fields_rejected():
    assert schemas.validate_agent_output({"action": "hold"})  # no weights/rationale


def test_non_finite_weight_rejected():
    obj = _valid()
    obj["target_weights"]["AAPL"] = float("nan")
    with pytest.raises(ValidationError):
        parse_agent_output(obj)


def test_empty_rationale_rejected():
    obj = _valid()
    obj["rationale"] = []
    assert schemas.validate_agent_output(obj)


def test_empty_weights_rejected():
    obj = _valid()
    obj["target_weights"] = {}
    assert schemas.validate_agent_output(obj)


def test_judge_schema_validates():
    good = {
        "groundedness_score": 0.9, "hallucination_risk": 0.1,
        "constraint_awareness_score": 0.9, "memory_use_score": 0.8,
        "decision_consistency_score": 0.9, "issues": [], "approved_for_monitoring": True,
    }
    assert schemas.validate_judge_output(good) == []
    bad = dict(good)
    del bad["hallucination_risk"]
    assert schemas.validate_judge_output(bad)


def test_json_schema_exports_present():
    assert schemas.AGENT_OUTPUT_SCHEMA["type"] == "object"
    assert "target_weights" in schemas.AGENT_OUTPUT_SCHEMA["properties"]
