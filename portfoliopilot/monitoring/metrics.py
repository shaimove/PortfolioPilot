"""Pure financial / portfolio metric helpers.

Used by the simulation engine and the local monitor. No side effects.
"""
from __future__ import annotations


def returns_from_values(values: list[float]) -> list[float]:
    out = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        out.append(values[i] / prev - 1.0 if prev else 0.0)
    return out


def total_return(values: list[float]) -> float:
    if len(values) < 2 or values[0] == 0:
        return 0.0
    return values[-1] / values[0] - 1.0


def max_drawdown(values: list[float]) -> float:
    """Most negative peak-to-trough drawdown over the series (<= 0)."""
    if not values:
        return 0.0
    peak = values[0]
    mdd = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            dd = v / peak - 1.0
            mdd = min(mdd, dd)
    return mdd


def excess_return(port_values: list[float], bench_values: list[float]) -> float:
    return total_return(port_values) - total_return(bench_values)


def transaction_cost_drag(total_costs: float, starting_capital: float) -> float:
    if starting_capital == 0:
        return 0.0
    return total_costs / starting_capital


def average(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def sector_concentration(weights: dict[str, float], sectors: dict[str, str]) -> float:
    """Largest single-sector weight (Herfindahl-style max), ignoring CASH."""
    agg: dict[str, float] = {}
    for tkr, w in weights.items():
        if tkr == "CASH":
            continue
        sec = sectors.get(tkr)
        if sec:
            agg[sec] = agg.get(sec, 0.0) + w
    return max(agg.values()) if agg else 0.0


def max_position_weight(weights: dict[str, float]) -> float:
    stock = [w for k, w in weights.items() if k != "CASH"]
    return max(stock) if stock else 0.0
