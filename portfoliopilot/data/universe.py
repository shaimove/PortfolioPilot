"""Point-in-time S&P 500 membership.

Data model
----------
`constituents_monthly` is a long table with columns ``month_end`` (date) and
``ticker``. A row means "this ticker was an S&P 500 member at this month-end".
This supports true point-in-time membership: additions and removals over time.

Fallbacks
---------
Real point-in-time membership data is not bundled. Two builders are provided:

* ``build_membership_from_current`` - uses a single static constituent list for
  every month. This is the documented FALLBACK and introduces *survivorship
  bias* (today's members are assumed to have always been members). The code is
  structured so this can be swapped for a real point-in-time dataset later.

* ``build_synthetic_membership`` - deterministically injects additions and
  removals over the window so the removal/addition handling can be exercised
  offline and in tests.

If a CSV exists at ``data/raw/constituents/membership.csv`` with columns
``month_end,ticker`` it is loaded as the authoritative point-in-time source.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from .. import config
from . import cache


# A curated subset of large, well-known S&P 500 names used as the MVP "current
# constituents" fallback universe. Not the full 500 — kept small so offline
# synthetic ingestion is fast, but the logic is identical for the full list.
SAMPLE_SP500: list[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "BAC", "XOM", "DIS", "ADBE", "CRM",
    "NFLX", "KO", "PEP", "CSCO", "INTC", "WMT", "MRK", "ABT", "CVX", "WFC",
    "MCD", "COST", "TMO", "ORCL", "ACN", "NKE", "LLY", "DHR", "TXN", "PM",
    "IBM", "QCOM", "HON", "UNP", "LOW", "AMD", "GS", "CAT", "BA", "GE",
]


def membership_csv_path() -> Path:
    return config.RAW_CONSTITUENTS / "membership.csv"


def current_constituents() -> list[str]:
    """Return the static fallback constituent list (survivorship-biased)."""
    return list(SAMPLE_SP500)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_membership_from_current(month_ends: list[dt.date],
                                  tickers: list[str] | None = None) -> pd.DataFrame:
    """Constant membership for every month-end. FALLBACK -> survivorship bias."""
    tickers = tickers or current_constituents()
    rows = [{"month_end": me, "ticker": t} for me in month_ends for t in tickers]
    return pd.DataFrame(rows)


def build_synthetic_membership(month_ends: list[dt.date],
                               tickers: list[str] | None = None,
                               n_initial: int = 40) -> pd.DataFrame:
    """Deterministic point-in-time membership with additions/removals over time.

    Starts with the first ``n_initial`` tickers, then every ~24 months removes
    one current member and adds one previously-excluded ticker. This guarantees
    the simulation sees real forced sells and new entrants.
    """
    tickers = tickers or current_constituents()
    initial = tickers[:n_initial]
    reserve = tickers[n_initial:]  # candidates that join later

    current = list(initial)
    reserve_q = list(reserve)
    rows: list[dict] = []

    for i, me in enumerate(month_ends):
        # every 24 months, swap one out and one in (if reserve available)
        if i > 0 and i % 24 == 0 and reserve_q and len(current) > 1:
            removed = current.pop(0)          # deterministic removal (oldest slot)
            added = reserve_q.pop(0)           # deterministic addition
            current.append(added)
            # `removed` simply disappears from membership from this month on
            _ = removed
        for t in current:
            rows.append({"month_end": me, "ticker": t})
    return pd.DataFrame(rows)


def load_membership_csv() -> pd.DataFrame | None:
    p = membership_csv_path()
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["month_end"] = pd.to_datetime(df["month_end"]).dt.date
    return df[["month_end", "ticker"]]


# --------------------------------------------------------------------------- #
# Universe accessor
# --------------------------------------------------------------------------- #
class Universe:
    """Read-only accessor over the processed monthly membership table."""

    def __init__(self, df: pd.DataFrame) -> None:
        d = df.copy()
        d["month_end"] = pd.to_datetime(d["month_end"]).dt.date
        self._df = d
        self._by_month: dict[dt.date, set[str]] = {
            me: set(g["ticker"]) for me, g in d.groupby("month_end")
        }
        self._month_ends = sorted(self._by_month.keys())

    @classmethod
    def load(cls) -> "Universe":
        df = cache.read_parquet(config.CONSTITUENTS_MONTHLY_PARQUET)
        if df is None or df.empty:
            raise FileNotFoundError(
                "constituents_monthly.parquet not found. Run ingest_constituents first."
            )
        return cls(df)

    @property
    def month_ends(self) -> list[dt.date]:
        return list(self._month_ends)

    def members_on(self, month_end: dt.date) -> set[str]:
        return set(self._by_month.get(month_end, set()))

    def all_tickers(self) -> list[str]:
        return sorted({t for s in self._by_month.values() for t in s})

    def removed_between(self, prev: dt.date, curr: dt.date) -> set[str]:
        """Tickers that were members at `prev` but not at `curr` (forced sells)."""
        return self.members_on(prev) - self.members_on(curr)

    def added_between(self, prev: dt.date, curr: dt.date) -> set[str]:
        """Tickers that are new members at `curr` vs `prev` (new entrants)."""
        return self.members_on(curr) - self.members_on(prev)
