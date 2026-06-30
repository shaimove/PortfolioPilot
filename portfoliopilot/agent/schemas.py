"""Pydantic schemas + validation for agent and judge outputs.

Pydantic is the canonical, strict validator (with retry-on-failure in the
agent). JSON-Schema dicts are derived from the Pydantic models so they can still
be advertised (e.g. via the MCP server) without duplicating definitions.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


# --------------------------------------------------------------------------- #
# Agent output models
# --------------------------------------------------------------------------- #
class NewMemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str
    content: str


class AgentOutput(BaseModel):
    """Strict schema for the portfolio decision agent's response."""
    model_config = ConfigDict(extra="ignore")

    action: Literal["rebalance", "hold", "raise_cash", "no_action"]
    target_weights: dict[str, float]
    rationale: list[str]
    memories_used: list[str] = []
    risk_notes: list[str] = []
    new_memory_candidates: list[NewMemoryCandidate] = []

    @field_validator("target_weights")
    @classmethod
    def _check_weights(cls, v: dict) -> dict:
        if not v:
            raise ValueError("target_weights must contain at least one entry")
        out: dict[str, float] = {}
        for k, val in v.items():
            try:
                f = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"weight for {k!r} is not a number")
            if math.isnan(f) or math.isinf(f):
                raise ValueError(f"weight for {k!r} is not finite")
            out[str(k).upper()] = f
        return out

    @field_validator("rationale")
    @classmethod
    def _check_rationale(cls, v: list) -> list:
        if not v:
            raise ValueError("rationale must contain at least one statement")
        return v


# --------------------------------------------------------------------------- #
# Judge output models
# --------------------------------------------------------------------------- #
class JudgeIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")
    severity: str
    type: str
    message: str


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    groundedness_score: float
    hallucination_risk: float
    constraint_awareness_score: float
    memory_use_score: float
    decision_consistency_score: float
    issues: list[JudgeIssue] = []
    approved_for_monitoring: bool


# JSON-Schema views derived from the Pydantic models (used for documentation /
# the MCP server tool descriptions).
AGENT_OUTPUT_SCHEMA = AgentOutput.model_json_schema()
JUDGE_OUTPUT_SCHEMA = JudgeOutput.model_json_schema()


# --------------------------------------------------------------------------- #
# Validation helpers (return human-readable error lists; [] means valid)
# --------------------------------------------------------------------------- #
def _format_errors(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        loc = "/".join(str(p) for p in err.get("loc", ())) or "<root>"
        out.append(f"{loc}: {err.get('msg', 'invalid')}")
    return out


def validate_agent_output(obj: dict) -> list[str]:
    try:
        AgentOutput.model_validate(obj)
        return []
    except ValidationError as e:
        return _format_errors(e)


def parse_agent_output(obj: dict) -> AgentOutput:
    """Strictly parse into an AgentOutput model (raises ValidationError)."""
    return AgentOutput.model_validate(obj)


def validate_judge_output(obj: dict) -> list[str]:
    try:
        JudgeOutput.model_validate(obj)
        return []
    except ValidationError as e:
        return _format_errors(e)


def normalize_agent_weights(obj: dict) -> dict:
    """Uppercase tickers and coerce weights to float; keep CASH as-is.

    Best-effort: malformed entries are dropped here and caught later by the
    Pydantic validator with a clear error message.
    """
    tw = obj.get("target_weights", {}) or {}
    norm: dict[str, float] = {}
    for k, v in tw.items():
        if v is None:
            continue
        try:
            norm[str(k).upper()] = float(v)
        except (TypeError, ValueError):
            continue
    obj["target_weights"] = norm
    return obj
