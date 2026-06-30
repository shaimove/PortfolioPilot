"""Deterministic risk engine.

Validates and repairs the agent's proposed target weights against hard
constraints. Always returns a feasible weight vector (the agent can never push
the simulated portfolio into an invalid state).

Enforced rules:
    * weights sum to 1.0 (stocks + CASH)
    * no negative weights / long-only
    * cash within [min_cash, max_cash]
    * no stock above max_asset_weight
    * no sector above max_sector_weight (if sector data exists)
    * monthly turnover below max_turnover_per_month
    * only stocks eligible in that month's S&P 500 universe
    * stocks removed from the index are forced to weight 0
    * stocks with invalid price history cannot be bought
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Constraints

CASH = "CASH"
EPS = 1e-9


@dataclass
class RiskResult:
    final_weights: dict[str, float]
    violations: list[dict] = field(default_factory=list)
    blocked: bool = False          # agent output rejected (fell back to repair)
    modified: bool = False         # weights changed from agent proposal
    turnover: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def violation_count(self) -> int:
        return len(self.violations)


def _one_way_turnover(new_w: dict[str, float], old_w: dict[str, float]) -> float:
    keys = set(new_w) | set(old_w)
    return 0.5 * sum(abs(new_w.get(k, 0.0) - old_w.get(k, 0.0)) for k in keys)


def _normalize_with_cash(stock_w: dict[str, float], constraints: Constraints) -> dict[str, float]:
    """Scale stock weights so that stock_sum + cash == 1, with cash clamped to
    [min_cash, max_cash]. Returns dict including CASH."""
    stock_sum = sum(stock_w.values())
    cash = 1.0 - stock_sum
    if cash < constraints.min_cash:
        # too little cash -> scale stocks down to free up cash
        target_stock = 1.0 - constraints.min_cash
        scale = target_stock / stock_sum if stock_sum > EPS else 0.0
        stock_w = {k: v * scale for k, v in stock_w.items()}
        cash = constraints.min_cash
    elif cash > constraints.max_cash:
        # too much cash -> scale stocks up (bounded by per-asset cap handled later)
        target_stock = 1.0 - constraints.max_cash
        scale = target_stock / stock_sum if stock_sum > EPS else 0.0
        stock_w = {k: v * scale for k, v in stock_w.items()}
        cash = constraints.max_cash
    out = dict(stock_w)
    out[CASH] = cash
    return out


def validate_and_repair(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    eligible: set[str],
    sectors: dict[str, str] | None,
    valid_history: dict[str, bool] | None,
    forced_sells: set[str],
    constraints: Constraints,
) -> RiskResult:
    sectors = sectors or {}
    valid_history = valid_history or {}
    violations: list[dict] = []
    notes: list[str] = []

    raw = dict(target_weights)
    cash_in = raw.pop(CASH, None)

    # 1) eligibility / forced sells / invalid history / negatives
    cleaned: dict[str, float] = {}
    for tkr, w in raw.items():
        ticker = tkr.upper()
        if ticker == CASH:
            continue
        if w is None:
            continue
        if w < -EPS:
            violations.append({"type": "negative_weight", "ticker": ticker,
                               "message": f"Negative weight {w:.4f} for {ticker} set to 0 (long-only)."})
            w = 0.0
        if w <= EPS:
            continue
        if ticker in forced_sells:
            violations.append({"type": "forced_sell_ignored", "ticker": ticker,
                               "message": f"{ticker} was removed from the index and must be sold; weight forced to 0."})
            continue
        if ticker not in eligible:
            violations.append({"type": "ineligible_ticker", "ticker": ticker,
                               "message": f"{ticker} is not in the S&P 500 universe this month; dropped."})
            continue
        held = current_weights.get(ticker, 0.0) > EPS
        if not valid_history.get(ticker, True) and not held:
            violations.append({"type": "invalid_history_buy", "ticker": ticker,
                               "message": f"{ticker} has invalid/insufficient price history and cannot be bought."})
            continue
        cleaned[ticker] = float(w)

    # 2) per-asset cap
    for ticker, w in list(cleaned.items()):
        if w > constraints.max_asset_weight + EPS:
            violations.append({"type": "max_asset_weight", "ticker": ticker,
                               "message": f"{ticker} weight {w:.4f} exceeds cap {constraints.max_asset_weight}; capped."})
            cleaned[ticker] = constraints.max_asset_weight

    # 3) sector cap (if sector data exists)
    if sectors:
        by_sector: dict[str, list[str]] = {}
        for ticker in cleaned:
            sec = sectors.get(ticker)
            if sec:
                by_sector.setdefault(sec, []).append(ticker)
        for sec, members in by_sector.items():
            sec_sum = sum(cleaned[m] for m in members)
            if sec_sum > constraints.max_sector_weight + EPS:
                violations.append({"type": "max_sector_weight",
                                   "message": f"Sector {sec} weight {sec_sum:.4f} exceeds cap "
                                              f"{constraints.max_sector_weight}; scaled down."})
                scale = constraints.max_sector_weight / sec_sum
                for m in members:
                    cleaned[m] *= scale

    # 4) normalize with cash bounds
    final = _normalize_with_cash(cleaned, constraints)

    # 4b) re-apply per-asset cap after normalization (scaling up can breach it)
    changed = True
    guard = 0
    while changed and guard < 5:
        guard += 1
        changed = False
        for ticker, w in list(final.items()):
            if ticker == CASH:
                continue
            if w > constraints.max_asset_weight + EPS:
                final[ticker] = constraints.max_asset_weight
                changed = True
        # re-normalize residual into cash
        stock_sum = sum(v for k, v in final.items() if k != CASH)
        final[CASH] = 1.0 - stock_sum

    # 5) turnover throttle: blend toward current weights to satisfy the cap.
    #    Forced sells are MANDATORY and exempt from throttling (always go to 0).
    turnover = _one_way_turnover(final, current_weights)
    if turnover > constraints.max_turnover_per_month + EPS and turnover > EPS:
        alpha = constraints.max_turnover_per_month / turnover  # in (0,1)
        keys = set(final) | set(current_weights)
        blended = {k: current_weights.get(k, 0.0) + alpha * (final.get(k, 0.0) - current_weights.get(k, 0.0))
                   for k in keys}
        # forced sells must fully liquidate regardless of the turnover cap
        for t in forced_sells:
            blended[t] = 0.0
        # renormalize: scale stocks under the cash floor, push residual to cash
        stock_sum = sum(v for k, v in blended.items() if k != CASH and v > EPS)
        max_stock = 1.0 - constraints.min_cash
        if stock_sum > max_stock and stock_sum > EPS:
            scale = max_stock / stock_sum
            for k in list(blended):
                if k != CASH:
                    blended[k] *= scale
        blended[CASH] = 1.0 - sum(v for k, v in blended.items() if k != CASH and v > EPS)
        violations.append({"type": "max_turnover",
                           "message": f"Turnover {turnover:.4f} exceeds cap "
                                      f"{constraints.max_turnover_per_month}; voluntary trades throttled."})
        final = {k: v for k, v in blended.items() if v > EPS or k == CASH}
        turnover = _one_way_turnover(final, current_weights)

    # clean near-zero noise
    final = {k: round(v, 8) for k, v in final.items() if abs(v) > 1e-6 or k == CASH}
    # final normalization safety
    s = sum(final.values())
    if abs(s - 1.0) > 1e-6 and s > EPS:
        final = {k: v / s for k, v in final.items()}

    modified = _weights_differ(final, target_weights)
    return RiskResult(
        final_weights=final,
        violations=violations,
        blocked=False,
        modified=modified,
        turnover=turnover,
        notes=notes,
    )


def _weights_differ(a: dict[str, float], b: dict[str, float], tol: float = 1e-4) -> bool:
    keys = set(a) | set(b)
    return any(abs(a.get(k, 0.0) - b.get(k, 0.0)) > tol for k in keys)


def check_only(weights: dict[str, float], constraints: Constraints,
               eligible: set[str], sectors: dict[str, str] | None = None) -> list[dict]:
    """Pure validation (no repair). Returns a list of violations. Used by the
    judge to detect issues the repair step silently fixed."""
    sectors = sectors or {}
    v: list[dict] = []
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-3:
        v.append({"type": "weights_sum", "message": f"Weights sum to {total:.4f}, not 1.0."})
    for k, w in weights.items():
        if w < -1e-6:
            v.append({"type": "negative_weight", "ticker": k, "message": f"{k} negative weight."})
        if k != CASH and w > constraints.max_asset_weight + 1e-6:
            v.append({"type": "max_asset_weight", "ticker": k, "message": f"{k} over asset cap."})
        if k != CASH and k not in eligible:
            v.append({"type": "ineligible_ticker", "ticker": k, "message": f"{k} not eligible."})
    cash = weights.get(CASH, 0.0)
    if cash < constraints.min_cash - 1e-6 or cash > constraints.max_cash + 1e-6:
        v.append({"type": "cash_bounds", "message": f"Cash {cash:.4f} outside bounds."})
    return v
