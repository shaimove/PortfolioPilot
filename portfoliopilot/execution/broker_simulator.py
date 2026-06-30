"""Simulated broker / execution engine.

Translates target weights into simulated trades against a share-based portfolio,
applies transaction costs, and reports execution metrics. Deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import TRANSACTION_COST_BPS
from .risk_engine import CASH

EPS = 1e-9


@dataclass
class Portfolio:
    cash: float
    shares: dict[str, float] = field(default_factory=dict)

    @classmethod
    def initial(cls, capital: float) -> "Portfolio":
        return cls(cash=float(capital), shares={})

    def position_value(self, prices: dict[str, float]) -> dict[str, float]:
        out = {}
        for t, sh in self.shares.items():
            px = prices.get(t)
            if px is None or sh <= EPS:
                continue
            out[t] = sh * px
        return out

    def value(self, prices: dict[str, float]) -> float:
        return self.cash + sum(self.position_value(prices).values())

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        total = self.value(prices)
        if total <= EPS:
            return {CASH: 1.0}
        w = {t: v / total for t, v in self.position_value(prices).items()}
        w[CASH] = self.cash / total
        return w


@dataclass
class ExecutionResult:
    transaction_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    forced_sell_count: int = 0
    new_entry_buy_count: int = 0
    changed_positions_count: int = 0
    turnover: float = 0.0
    transaction_cost: float = 0.0
    cash_after: float = 0.0
    new_value: float = 0.0
    new_holdings: dict[str, float] = field(default_factory=dict)   # ticker -> weight
    trades: list[dict] = field(default_factory=list)               # per-ticker trade detail


def _one_way_turnover(new_w: dict[str, float], old_w: dict[str, float]) -> float:
    keys = set(new_w) | set(old_w)
    return 0.5 * sum(abs(new_w.get(k, 0.0) - old_w.get(k, 0.0)) for k in keys)


def execute(
    portfolio: Portfolio,
    target_weights: dict[str, float],
    prices: dict[str, float],
    forced_sells: set[str] | None = None,
    prev_weights: dict[str, float] | None = None,
    new_entrants: set[str] | None = None,
    cost_bps: float = TRANSACTION_COST_BPS,
) -> ExecutionResult:
    """Execute trades to reach `target_weights`. Mutates `portfolio` in place."""
    forced_sells = forced_sells or set()
    new_entrants = new_entrants or set()
    cost_fraction = cost_bps / 10_000.0

    total_value = portfolio.value(prices)
    old_values = portfolio.position_value(prices)
    old_weights = prev_weights if prev_weights is not None else portfolio.weights(prices)

    # target dollar per stock (exclude CASH; cash is the residual)
    targets = {t.upper(): w for t, w in target_weights.items() if t.upper() != CASH}
    involved = set(old_values) | set(targets)

    dollar_threshold = max(1.0, 1e-4 * total_value)
    trades: list[dict] = []
    new_shares: dict[str, float] = {}
    traded_dollars = 0.0
    buy_count = sell_count = forced_sell_count = new_entry_buy_count = changed = 0
    invested = 0.0

    for ticker in sorted(involved):
        px = prices.get(ticker)
        cur_val = old_values.get(ticker, 0.0)
        if px is None or px <= EPS:
            # cannot price -> keep existing shares, no trade
            if portfolio.shares.get(ticker, 0.0) > EPS:
                new_shares[ticker] = portfolio.shares[ticker]
            continue

        tgt_w = targets.get(ticker, 0.0)
        tgt_val = tgt_w * total_value
        trade = tgt_val - cur_val

        if abs(trade) > dollar_threshold:
            traded_dollars += abs(trade)
            if trade > 0:
                buy_count += 1
                if cur_val <= dollar_threshold:
                    new_entry_buy_count += 1 if ticker in new_entrants or cur_val <= EPS else 0
            else:
                sell_count += 1
                if ticker in forced_sells and tgt_val <= EPS:
                    forced_sell_count += 1
            changed += 1
            trades.append({
                "ticker": ticker,
                "side": "buy" if trade > 0 else "sell",
                "dollars": round(trade, 2),
                "forced": ticker in forced_sells,
            })

        if tgt_val > EPS:
            new_shares[ticker] = tgt_val / px
            invested += tgt_val

    cost = cost_fraction * traded_dollars
    new_cash = total_value - invested - cost
    portfolio.shares = {t: s for t, s in new_shares.items() if s > EPS}
    portfolio.cash = new_cash

    new_value = portfolio.value(prices)
    new_weights = portfolio.weights(prices)
    turnover = _one_way_turnover(new_weights, old_weights)

    return ExecutionResult(
        transaction_count=buy_count + sell_count,
        buy_count=buy_count,
        sell_count=sell_count,
        forced_sell_count=forced_sell_count,
        new_entry_buy_count=new_entry_buy_count,
        changed_positions_count=changed,
        turnover=turnover,
        transaction_cost=cost,
        cash_after=new_cash,
        new_value=new_value,
        new_holdings=new_weights,
        trades=trades,
    )
