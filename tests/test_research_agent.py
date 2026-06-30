"""Tests for the EXPERIMENTAL (not-activated) fundamental research agent."""
from portfoliopilot.agent import research_agent
from portfoliopilot.agent.research_agent import (
    ACTIVATED,
    ResearchAgent,
    rule_based_research,
    validate_research_output,
)
from portfoliopilot.agent.research_sources import StubCompanyReportProvider


def test_agent_is_not_activated():
    # Safety: this agent must stay out of the live system until explicitly enabled.
    assert ACTIVATED is False


def test_stub_reports_are_deterministic():
    p = StubCompanyReportProvider()
    r1 = p.get_report("AAPL")
    r2 = p.get_report("AAPL")
    assert r1 == r2
    assert {"ticker", "revenue", "net_margin", "pe_ratio", "roe"}.issubset(r1)


def test_rule_based_output_is_schema_valid():
    p = StubCompanyReportProvider()
    reports = p.get_reports(["AAPL", "MSFT", "JPM", "XOM", "NVDA"])
    out = rule_based_research(reports, as_of="2024-01-31")
    assert validate_research_output(out) == []
    assert len(out["recommendations"]) == 5
    for rec in out["recommendations"]:
        assert rec["stance"] in ("buy", "hold", "avoid")
        assert 0.0 <= rec["conviction"] <= 1.0


def test_recommendations_sorted_by_conviction():
    p = StubCompanyReportProvider()
    reports = p.get_reports(["AAPL", "MSFT", "JPM", "XOM", "NVDA", "KO", "BA"])
    out = rule_based_research(reports, as_of="2024-01-31")
    convs = [r["conviction"] for r in out["recommendations"]]
    assert convs == sorted(convs, reverse=True)


def test_agent_recommend_uses_fallback_without_llm(monkeypatch):
    monkeypatch.setattr(research_agent.llm, "available", lambda: False)
    agent = ResearchAgent()
    reports = agent.reports_for(["AAPL", "MSFT"])
    res = agent.recommend(reports, as_of="2024-01-31")
    assert res.used_fallback is True
    assert res.source == "fallback"
    assert validate_research_output(res.output) == []


def test_invalid_research_output_detected():
    bad = {"as_of": "2024-01-31", "summary": "x",
           "recommendations": [{"ticker": "AAPL", "stance": "strong_buy",
                                "conviction": 0.5, "rationale": []}]}
    assert validate_research_output(bad)  # 'strong_buy' is not a valid stance
