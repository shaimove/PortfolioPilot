from portfoliopilot.execution import broker_simulator as broker
from portfoliopilot.execution.broker_simulator import Portfolio, execute
from portfoliopilot.execution.risk_engine import CASH


def test_initial_value_is_capital():
    p = Portfolio.initial(100_000)
    assert p.value({}) == 100_000


def test_execute_buys_and_computes_turnover():
    p = Portfolio.initial(100_000)
    prices = {"AAPL": 100.0, "MSFT": 200.0}
    res = execute(p, {"AAPL": 0.5, "MSFT": 0.4, CASH: 0.1}, prices,
                  prev_weights={CASH: 1.0})
    assert res.buy_count == 2
    assert res.transaction_count == 2
    # turnover from all-cash to 90% invested ~ 0.9 one-way
    assert 0.85 < res.turnover <= 0.95
    w = p.weights(prices)
    assert abs(w["AAPL"] - 0.5) < 0.02
    assert abs(w.get(CASH, 0) - 0.1) < 0.02


def test_transaction_cost_reduces_value():
    p = Portfolio.initial(100_000)
    prices = {"AAPL": 100.0}
    res = execute(p, {"AAPL": 0.9, CASH: 0.1}, prices, prev_weights={CASH: 1.0},
                  cost_bps=10.0)
    assert res.transaction_cost > 0
    # value after trading should be starting capital minus cost (prices unchanged)
    assert res.new_value < 100_000
    assert abs((100_000 - res.new_value) - res.transaction_cost) < 1e-6


def test_forced_sell_executes_to_zero():
    p = Portfolio.initial(100_000)
    prices = {"OLD": 50.0, "AAPL": 100.0}
    # first establish an OLD position
    execute(p, {"OLD": 0.5, "AAPL": 0.4, CASH: 0.1}, prices, prev_weights={CASH: 1.0})
    assert p.shares.get("OLD", 0) > 0
    # now force-sell OLD (not in target, in forced_sells)
    res = execute(p, {"AAPL": 0.8, CASH: 0.2}, prices,
                  forced_sells={"OLD"}, prev_weights=p.weights(prices))
    assert p.shares.get("OLD", 0.0) == 0.0
    assert res.forced_sell_count == 1


def test_price_growth_increases_value():
    p = Portfolio.initial(100_000)
    execute(p, {"AAPL": 0.9, CASH: 0.1}, {"AAPL": 100.0}, prev_weights={CASH: 1.0})
    v1 = p.value({"AAPL": 100.0})
    v2 = p.value({"AAPL": 110.0})   # price up 10%
    assert v2 > v1
