"""MCP server tool functions (called directly; FastMCP returns the wrapped fn)."""
from portfoliopilot import mcp_server as mcp


def test_list_month_ends_and_universe(offline_data):
    months = mcp.list_month_ends()["month_ends"]
    assert len(months) == 120
    uni = mcp.list_universe(months[-1])
    assert uni["n_members"] >= 1
    assert isinstance(uni["members"], list)


def test_membership_changes(offline_data):
    months = mcp.list_month_ends()["month_ends"]
    # synthetic membership swaps every 24 months -> month 24 has changes
    res = mcp.membership_changes(months[23], months[24])
    assert "removed" in res and "added" in res


def test_get_and_compute_features(offline_data):
    months = mcp.list_month_ends()["month_ends"]
    me = months[len(months) // 2]
    cached = mcp.get_stock_features(me)
    recomputed = mcp.compute_features_asof(me)
    assert cached["count"] > 0
    assert recomputed["count"] > 0


def test_validate_target_weights_tool():
    res = mcp.validate_target_weights(
        target_weights={"AAPL": 0.9, "CASH": 0.1},
        eligible=["AAPL"],
    )
    assert abs(sum(res["final_weights"].values()) - 1.0) < 1e-6
    assert res["final_weights"]["AAPL"] <= 0.08 + 1e-6  # capped by default constraints


def test_simulate_execution_tool():
    res = mcp.simulate_execution(
        target_weights={"AAPL": 0.5, "CASH": 0.5},
        cash=100000.0, shares={}, prices={"AAPL": 100.0},
        prev_weights={"CASH": 1.0},
    )
    assert res["buy_count"] == 1
    assert res["transaction_cost"] > 0


def test_portfolio_metrics_tool():
    res = mcp.compute_portfolio_metrics([100.0, 110.0, 99.0], [100.0, 105.0, 102.0])
    assert "total_return" in res and "excess_return" in res
    assert res["max_drawdown"] <= 0.0


def test_run_judge_tool_flags_unsupported_claim():
    ji = {
        "agent_input": {"eligible_candidates": [{"ticker": "AAPL"}], "forced_actions": []},
        "agent_output": {"action": "rebalance",
                         "target_weights": {"AAPL": 0.06, "CASH": 0.94},
                         "rationale": ["AAPL earnings beat estimates."],
                         "memories_used": [], "risk_notes": [], "new_memory_candidates": []},
        "risk_result": {"violations": [], "violation_count": 0},
        "executed_trades": [], "relevant_memories": [],
        "constraints": {"max_asset_weight": 0.08, "max_sector_weight": 0.30,
                        "max_turnover_per_month": 0.30, "min_cash": 0.02,
                        "max_cash": 0.30, "long_only": True},
    }
    res = mcp.run_judge(ji)
    assert res["unsupported_claim_count"] >= 1


def test_validate_agent_output_tool():
    ok = mcp.validate_agent_output({
        "action": "rebalance", "target_weights": {"AAPL": 0.5, "CASH": 0.5},
        "rationale": ["ok"],
    })
    assert ok["valid"] is True
    bad = mcp.validate_agent_output({"action": "rebalance"})
    assert bad["valid"] is False and bad["errors"]
