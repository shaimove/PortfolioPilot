"""Provider abstraction for market data.

Two provider families:

* PriceProvider     -> daily adjusted OHLCV.
* FundamentalsProvider -> per-ticker fundamentals (stub-friendly).

The default MVP implementation uses yfinance. A fully deterministic
``SyntheticPriceProvider`` is also provided so the whole system (ingestion,
features, simulation, tests) can run offline with no network access.

IMPORTANT: providers are only ever used during *ingestion*, never inside the
simulation loop. The simulation reads exclusively from local Parquet/DuckDB.
"""
from __future__ import annotations

import abc
import datetime as dt
import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Price providers
# --------------------------------------------------------------------------- #
class PriceProvider(abc.ABC):
    """Returns daily adjusted OHLCV for a ticker over [start, end]."""

    name: str = "base"

    @abc.abstractmethod
    def get_daily_prices(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Return a DataFrame indexed by date with columns:
        open, high, low, close, adj_close, volume.
        Returns an empty frame if data is unavailable.
        """
        raise NotImplementedError


class YFinancePriceProvider(PriceProvider):
    """Historical adjusted OHLCV via yfinance (MVP default)."""

    name = "yfinance"

    def get_daily_prices(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        import yfinance as yf  # imported lazily; only needed at ingestion time

        raw = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + dt.timedelta(days=1)).isoformat(),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            return _empty_price_frame()

        # yfinance may return a MultiIndex when given a single ticker in newer versions
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = pd.DataFrame(index=raw.index)
        df["open"] = raw.get("Open")
        df["high"] = raw.get("High")
        df["low"] = raw.get("Low")
        df["close"] = raw.get("Close")
        df["adj_close"] = raw.get("Adj Close", raw.get("Close"))
        df["volume"] = raw.get("Volume")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        return df.dropna(subset=["adj_close"])


class SyntheticPriceProvider(PriceProvider):
    """Deterministic geometric-brownian-motion price generator.

    Used for offline runs and tests. Each ticker gets a stable seed derived from
    its symbol, so output is reproducible across runs.
    """

    name = "synthetic"

    def __init__(self, annual_drift: float = 0.08, annual_vol: float = 0.25,
                 start_price_range: tuple[float, float] = (20.0, 400.0)) -> None:
        self.annual_drift = annual_drift
        self.annual_vol = annual_vol
        self.start_price_range = start_price_range

    def _seed(self, ticker: str) -> int:
        h = hashlib.sha256(ticker.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def get_daily_prices(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        rng = np.random.default_rng(self._seed(ticker))
        # business days only
        dates = pd.bdate_range(start=start, end=end)
        n = len(dates)
        if n == 0:
            return _empty_price_frame()

        lo, hi = self.start_price_range
        start_price = float(rng.uniform(lo, hi))
        # per-ticker drift/vol jitter so names diverge
        drift = self.annual_drift * float(rng.uniform(0.3, 1.7)) - 0.04
        vol = self.annual_vol * float(rng.uniform(0.6, 1.6))

        # Give the benchmark a steady, realistic broad-market profile so the
        # offline demo is sensible (real ingestion uses actual index data).
        if ticker == "^GSPC":
            start_price = 2000.0
            drift = 0.085
            vol = 0.15

        dt_step = 1.0 / 252.0
        shocks = rng.normal(
            loc=(drift - 0.5 * vol**2) * dt_step,
            scale=vol * np.sqrt(dt_step),
            size=n,
        )
        log_path = np.cumsum(shocks)
        adj_close = start_price * np.exp(log_path)

        intraday = np.abs(rng.normal(0.0, 0.01, size=n))
        high = adj_close * (1.0 + intraday)
        low = adj_close * (1.0 - intraday)
        open_ = np.concatenate([[adj_close[0]], adj_close[:-1]])
        volume = rng.integers(500_000, 20_000_000, size=n)

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": adj_close,        # treat synthetic close == adj_close
                "adj_close": adj_close,
                "volume": volume,
            },
            index=pd.DatetimeIndex(dates, name="date"),
        )
        return df


def _empty_price_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        columns=["open", "high", "low", "close", "adj_close", "volume"]
    )
    df.index = pd.DatetimeIndex([], name="date")
    return df


# --------------------------------------------------------------------------- #
# Fundamentals providers
# --------------------------------------------------------------------------- #
class FundamentalsProvider(abc.ABC):
    """Returns a flat dict of fundamental fields for a ticker.

    Designed so SEC EDGAR Companyfacts / Alpha Vantage / FMP / Tiingo can be
    dropped in later by subclassing.
    """

    name: str = "base"

    @abc.abstractmethod
    def get_fundamentals(self, ticker: str) -> dict:
        raise NotImplementedError


class StubFundamentalsProvider(FundamentalsProvider):
    """Returns only a sector classification (deterministic) and nothing else.

    Crucially this returns NO earnings/revenue/valuation fields, so the judge can
    correctly flag any agent rationale that references such facts.
    """

    name = "stub"

    SECTORS = [
        "Technology", "Financials", "Health Care", "Consumer Discretionary",
        "Consumer Staples", "Industrials", "Energy", "Utilities",
        "Materials", "Real Estate", "Communication Services",
    ]

    def get_fundamentals(self, ticker: str) -> dict:
        idx = int(hashlib.sha256(ticker.encode()).hexdigest()[:8], 16) % len(self.SECTORS)
        return {"ticker": ticker, "sector": self.SECTORS[idx]}


class YFinanceFundamentalsProvider(FundamentalsProvider):
    """Optional yfinance fundamentals (best-effort, may be sparse)."""

    name = "yfinance"

    def get_fundamentals(self, ticker: str) -> dict:
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        return {
            "ticker": ticker,
            "sector": info.get("sector") or StubFundamentalsProvider().get_fundamentals(ticker)["sector"],
        }


def default_price_provider(offline: bool = False) -> PriceProvider:
    return SyntheticPriceProvider() if offline else YFinancePriceProvider()


def default_fundamentals_provider(offline: bool = False) -> FundamentalsProvider:
    return StubFundamentalsProvider() if offline else YFinanceFundamentalsProvider()
