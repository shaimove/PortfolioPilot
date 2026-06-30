"""LLM-as-a-judge monitoring component.

The judge reviews a decision AFTER it is produced. It never trades. It scores
groundedness, hallucination risk, constraint awareness, memory use, and
decision/output consistency, and emits structured issues.

A fully deterministic judge is always available (and is the fallback when no LLM
is configured). When an LLM is configured, its JSON output is validated and we
fall back to the deterministic judge on failure. We also always merge in the
deterministic structural checks (forced-sell, missed-violation) so the LLM can
never hide a hard, detectable problem.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .. import llm
from ..agent import prompts, schemas
from ..execution import risk_engine

# Fundamental / news concepts that are NEVER provided in the agent input.
# Any mention in the rationale is therefore an unsupported claim.
_FORBIDDEN_PATTERNS = {
    "earnings": r"\bearnings?\b",
    "revenue": r"\brevenue[s]?\b",
    "profit": r"\bprofit(s|ability)?\b",
    "valuation": r"\bvaluation[s]?\b|\bp/e\b|\bpe ratio\b|\bprice[- ]to[- ]earnings\b",
    "eps": r"\beps\b|\bearnings per share\b",
    "analyst": r"\banalyst[s]?\b|\bprice target[s]?\b|\bupgrade[d]?\b|\bdowngrade[d]?\b|\brating[s]?\b",
    "news": r"\bnews\b|\bheadline[s]?\b|\bannounc(e|ed|ement)\b",
    "guidance": r"\bguidance\b|\bforecast(s|ed)?\b|\boutlook\b",
    "fundamentals": r"\bfundamental[s]?\b|\bbalance sheet\b|\bcash flow\b|\bmargin[s]?\b|\bdividend[s]?\b|\bbuyback[s]?\b|\bsales\b",
}


@dataclass
class JudgeResult:
    output: dict
    source: str = "deterministic"      # "llm" | "deterministic"
    llm_usage: dict = field(default_factory=dict)
    unsupported_claim_count: int = 0
    warning_count: int = 0
    critical_count: int = 0


def _scan_unsupported_claims(rationale: list[str], risk_notes: list[str]) -> list[dict]:
    text = " ".join((rationale or []) + (risk_notes or [])).lower()
    issues: list[dict] = []
    for name, pattern in _FORBIDDEN_PATTERNS.items():
        if re.search(pattern, text):
            issues.append({
                "severity": "warning",
                "type": "unsupported_claim",
                "message": f"Rationale references '{name}', which was not provided in the "
                           f"agent input (only price-derived features were available).",
            })
    return issues


def _mentioned_tickers(rationale: list[str], candidate_tickers: set[str]) -> set[str]:
    text = " ".join(rationale or [])
    found = set(re.findall(r"\b[A-Z]{1,5}(?:-[A-Z])?\b", text))
    return {t for t in found if t in candidate_tickers}


def deterministic_judge(judge_input: dict) -> dict:
    agent_in = judge_input.get("agent_input", {})
    agent_out = judge_input.get("agent_output", {})
    risk = judge_input.get("risk_result", {})
    memories = judge_input.get("relevant_memories", [])
    constraints = judge_input.get("constraints", {})

    rationale = agent_out.get("rationale", []) or []
    risk_notes = agent_out.get("risk_notes", []) or []
    target_weights = agent_out.get("target_weights", {}) or {}
    memories_used = set(agent_out.get("memories_used", []) or [])

    issues: list[dict] = []

    # 1) unsupported claims (hallucinated facts)
    unsupported = _scan_unsupported_claims(rationale, risk_notes)
    issues.extend(unsupported)
    unsupported_count = len(unsupported)

    # 2) ignored forced sells
    forced = {f["ticker"].upper() for f in agent_in.get("forced_actions", [])
              if f.get("action") == "sell"}
    ignored_forced = [t for t in forced if target_weights.get(t, 0.0) > 1e-6]
    for t in ignored_forced:
        issues.append({
            "severity": "critical",
            "type": "forced_sell_ignored",
            "message": f"{t} was flagged for forced sale (removed from index) but still "
                       f"has a non-zero target weight.",
        })

    # 3) decision/output consistency: tickers praised in rationale but absent/zero
    cand_tickers = {c["ticker"].upper() for c in agent_in.get("eligible_candidates", [])}
    mentioned = _mentioned_tickers(rationale, cand_tickers)
    inconsistent = [t for t in mentioned if target_weights.get(t, 0.0) <= 1e-6]
    for t in inconsistent:
        issues.append({
            "severity": "warning",
            "type": "decision_consistency",
            "message": f"Rationale discusses {t} but the target weights give it ~0 weight.",
        })

    # 4) stale memory usage
    stale_ids = {m.get("memory_id") for m in memories if m.get("status") and m["status"] != "active"}
    used_stale = memories_used & {s for s in stale_ids if s}
    for mid in used_stale:
        issues.append({
            "severity": "warning",
            "type": "stale_memory",
            "message": f"Decision used stale memory {mid}.",
        })

    # 5) constraint violations missed by the risk engine (independent re-check)
    eligible = cand_tickers
    sectors = {c["ticker"].upper(): c.get("sector") for c in agent_in.get("eligible_candidates", [])}
    cons = risk_engine.Constraints(**{k: constraints.get(k, getattr(risk_engine.Constraints(), k))
                                      for k in risk_engine.Constraints().as_dict()})
    missed = risk_engine.check_only(target_weights, cons, eligible, sectors)
    risk_violation_count = int(risk.get("violation_count", len(risk.get("violations", []))))
    for v in missed:
        # These were auto-repaired by the deterministic risk engine, so they are
        # constraint-awareness warnings (the agent proposed an infeasible weight),
        # not critical execution failures.
        issues.append({
            "severity": "warning",
            "type": "constraint_awareness",
            "message": "Agent output needed risk-engine repair: " + v.get("message", v.get("type", "")),
        })

    # 6) overconfidence: lots of new memory candidates with strong claims
    new_mems = agent_out.get("new_memory_candidates", []) or []
    if len(new_mems) >= 4:
        issues.append({
            "severity": "info",
            "type": "overconfidence",
            "message": "Agent proposed many new memories from a single decision.",
        })

    # ----- scores -----
    hallucination_risk = min(1.0, 0.25 * unsupported_count + 0.4 * len(ignored_forced))
    groundedness = max(0.0, 1.0 - 0.2 * unsupported_count - 0.3 * len(ignored_forced))
    consistency = max(0.0, 1.0 - 0.25 * len(inconsistent) - 0.3 * len(ignored_forced))
    constraint_awareness = max(0.0, 1.0 - 0.15 * len(missed) - 0.1 * risk_violation_count)
    if memories:
        mem_score = max(0.0, 1.0 - 0.4 * len(used_stale)) if memories_used else 0.6
    else:
        mem_score = 1.0 if not memories_used else 0.7

    warning_count = sum(1 for i in issues if i["severity"] == "warning")
    critical_count = sum(1 for i in issues if i["severity"] == "critical")

    approved = hallucination_risk <= 0.30 and critical_count == 0

    return {
        "groundedness_score": round(groundedness, 4),
        "hallucination_risk": round(hallucination_risk, 4),
        "constraint_awareness_score": round(constraint_awareness, 4),
        "memory_use_score": round(mem_score, 4),
        "decision_consistency_score": round(consistency, 4),
        "issues": issues,
        "approved_for_monitoring": bool(approved),
        "_unsupported_claim_count": unsupported_count,
    }


class Judge:
    def evaluate(self, judge_input: dict) -> JudgeResult:
        deterministic = deterministic_judge(judge_input)

        output = deterministic
        source = "deterministic"
        usage: dict = {}

        if llm.available():
            resp = llm.chat_json(prompts.JUDGE_SYSTEM_PROMPT,
                                 prompts.build_judge_user_prompt(judge_input))
            usage = {"latency_ms": resp.latency_ms, "total_tokens": resp.total_tokens,
                     "cost_usd": resp.cost_usd, "ok": resp.ok, "error": resp.error}
            if resp.ok:
                obj = llm.extract_json(resp.text)
                if obj is not None and not schemas.validate_judge_output(obj):
                    # merge: keep LLM scores but force-include hard structural issues
                    merged_issues = list(obj.get("issues", []))
                    hard = [i for i in deterministic["issues"]
                            if i["type"] in ("forced_sell_ignored", "constraint_violation")]
                    merged_issues.extend(hard)
                    obj["issues"] = merged_issues
                    if any(i["severity"] == "critical" for i in hard):
                        obj["approved_for_monitoring"] = False
                    obj["_unsupported_claim_count"] = deterministic["_unsupported_claim_count"]
                    output = obj
                    source = "llm"

        issues = output.get("issues", [])
        warning_count = sum(1 for i in issues if i.get("severity") == "warning")
        critical_count = sum(1 for i in issues if i.get("severity") == "critical")
        unsupported = int(output.pop("_unsupported_claim_count",
                                     sum(1 for i in issues if i.get("type") == "unsupported_claim")))

        return JudgeResult(
            output=output, source=source, llm_usage=usage,
            unsupported_claim_count=unsupported,
            warning_count=warning_count, critical_count=critical_count,
        )
