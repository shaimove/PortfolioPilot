from portfoliopilot.config import Constraints
from portfoliopilot.execution import risk_engine
from portfoliopilot.execution.risk_engine import CASH, validate_and_repair


def _eligible(*tk):
    return set(tk)


def test_rejects_and_repairs_invalid_sum_and_negatives():
    c = Constraints()
    # weights don't sum to 1 and include a negative weight (long-only violation)
    target = {"AAPL": 0.5, "MSFT": -0.2, "JPM": 0.3}
    res = validate_and_repair(
        target_weights=target,
        current_weights={CASH: 1.0},
        eligible=_eligible("AAPL", "MSFT", "JPM"),
        sectors={}, valid_history={}, forced_sells=set(), constraints=c,
    )
    total = sum(res.final_weights.values())
    assert abs(total - 1.0) < 1e-6
    assert all(w >= -1e-9 for w in res.final_weights.values())
    assert any(v["type"] == "negative_weight" for v in res.violations)


def test_caps_max_position():
    c = Constraints(max_asset_weight=0.08)
    target = {"AAPL": 0.9, CASH: 0.1}
    res = validate_and_repair(
        target_weights=target,
        current_weights={CASH: 1.0},
        eligible=_eligible("AAPL"),
        sectors={}, valid_history={}, forced_sells=set(), constraints=c,
    )
    assert res.final_weights.get("AAPL", 0.0) <= c.max_asset_weight + 1e-6
    assert any(v["type"] == "max_asset_weight" for v in res.violations)


def test_cash_bounds_enforced():
    c = Constraints(min_cash=0.02, max_cash=0.30, max_asset_weight=1.0,
                    max_turnover_per_month=1.0)
    # all into one stock -> cash would be 0, below min_cash
    res = validate_and_repair(
        target_weights={"AAPL": 1.0},
        current_weights={CASH: 1.0},
        eligible=_eligible("AAPL"),
        sectors={}, valid_history={}, forced_sells=set(), constraints=c,
    )
    assert res.final_weights[CASH] >= c.min_cash - 1e-6
    assert res.final_weights[CASH] <= c.max_cash + 1e-6


def test_drops_ineligible_and_forced_sells():
    c = Constraints(max_asset_weight=1.0)
    res = validate_and_repair(
        target_weights={"AAPL": 0.4, "ZZZ": 0.3, "OLD": 0.3},
        current_weights={CASH: 1.0},
        eligible=_eligible("AAPL", "OLD"),     # ZZZ not eligible
        sectors={}, valid_history={}, forced_sells={"OLD"}, constraints=c,
    )
    assert "ZZZ" not in res.final_weights
    assert res.final_weights.get("OLD", 0.0) == 0.0 or "OLD" not in res.final_weights
    types = {v["type"] for v in res.violations}
    assert "ineligible_ticker" in types
    assert "forced_sell_ignored" in types


def test_invalid_history_cannot_be_bought():
    c = Constraints(max_asset_weight=1.0)
    res = validate_and_repair(
        target_weights={"BAD": 0.5},
        current_weights={CASH: 1.0},
        eligible=_eligible("BAD"),
        sectors={}, valid_history={"BAD": False}, forced_sells=set(), constraints=c,
    )
    assert "BAD" not in res.final_weights
    assert any(v["type"] == "invalid_history_buy" for v in res.violations)


def test_turnover_throttled():
    c = Constraints(max_turnover_per_month=0.10, max_asset_weight=1.0, min_cash=0.0)
    current = {"AAPL": 1.0}
    res = validate_and_repair(
        target_weights={"MSFT": 1.0},   # 100% switch -> huge turnover
        current_weights=current,
        eligible=_eligible("AAPL", "MSFT"),
        sectors={}, valid_history={}, forced_sells=set(), constraints=c,
    )
    assert res.turnover <= c.max_turnover_per_month + 1e-3
    assert any(v["type"] == "max_turnover" for v in res.violations)


def test_sector_cap_enforced():
    c = Constraints(max_sector_weight=0.30, max_asset_weight=0.30, min_cash=0.0, max_cash=0.5)
    sectors = {"A": "Tech", "B": "Tech", "C": "Tech"}
    res = validate_and_repair(
        target_weights={"A": 0.3, "B": 0.3, "C": 0.3},
        current_weights={CASH: 1.0},
        eligible=_eligible("A", "B", "C"),
        sectors=sectors, valid_history={}, forced_sells=set(), constraints=c,
    )
    tech = sum(res.final_weights.get(t, 0.0) for t in ("A", "B", "C"))
    assert tech <= c.max_sector_weight + 1e-3
