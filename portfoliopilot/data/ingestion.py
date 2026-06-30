"""Ingestion orchestration.

Downloads/builds and caches everything the simulation needs, BEFORE any
simulation runs. Nothing here is called inside the simulation loop.

Outputs:
* data/raw/prices/<ticker>.parquet          (per-ticker daily OHLCV cache)
* data/processed/prices.parquet             (consolidated long table)
* data/processed/constituents_monthly.parquet
* data/processed/fundamentals.parquet
* data/processed/metadata.parquet
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .. import config
from ..utils import simulation_month_ends
from . import cache, universe
from .providers import (
    FundamentalsProvider,
    PriceProvider,
    default_fundamentals_provider,
    default_price_provider,
)

# A stock needs at least ~12 months of daily history to be eligible (for 12m
# features). We require a comfortable buffer.
MIN_VALID_DAYS = 252


def _date_range(years: int) -> tuple[dt.date, dt.date]:
    end = dt.date.today()
    # add ~14 months of lead-in so the first simulated month has 12m lookback
    start = end - dt.timedelta(days=int(365.25 * years) + 430)
    return start, end


# --------------------------------------------------------------------------- #
# Constituents
# --------------------------------------------------------------------------- #
def ingest_constituents(years: int = config.SIM_YEARS,
                        offline: bool = False,
                        mode: str = "auto") -> pd.DataFrame:
    """Build and cache the monthly S&P 500 membership table.

    mode:
      * "pit"       -> use point-in-time CSV at data/raw/constituents/membership.csv
      * "synthetic" -> deterministic additions/removals (good for tests/offline)
      * "current"   -> static current constituents (survivorship-biased fallback)
      * "auto"      -> pit if CSV present, else synthetic if offline, else current
    """
    config.ensure_dirs()
    month_ends = simulation_month_ends(total_months=years * config.MONTHS_PER_YEAR)

    csv = universe.load_membership_csv()
    if mode == "auto":
        if csv is not None:
            mode = "pit"
        elif offline:
            mode = "synthetic"
        else:
            mode = "current"

    if mode == "pit":
        if csv is None:
            raise FileNotFoundError("membership.csv not found for point-in-time mode")
        df = csv[csv["month_end"].isin(month_ends)].copy()
    elif mode == "synthetic":
        df = universe.build_synthetic_membership(month_ends)
    elif mode == "current":
        df = universe.build_membership_from_current(month_ends)
    else:
        raise ValueError(f"unknown constituents mode: {mode}")

    df["month_end"] = pd.to_datetime(df["month_end"]).dt.date
    cache.write_parquet(df, config.CONSTITUENTS_MONTHLY_PARQUET)
    return df


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
def ingest_prices(years: int = config.SIM_YEARS,
                  offline: bool = False,
                  provider: PriceProvider | None = None,
                  tickers: list[str] | None = None,
                  include_benchmark: bool = True) -> pd.DataFrame:
    """Download/cache daily OHLCV for all universe tickers + benchmark."""
    config.ensure_dirs()
    provider = provider or default_price_provider(offline=offline)
    start, end = _date_range(years)

    if tickers is None:
        uni_df = cache.read_parquet(config.CONSTITUENTS_MONTHLY_PARQUET)
        if uni_df is not None and not uni_df.empty:
            tickers = sorted(uni_df["ticker"].unique().tolist())
        else:
            tickers = universe.current_constituents()

    all_tickers = list(tickers)
    if include_benchmark and config.BENCHMARK_TICKER not in all_tickers:
        all_tickers.append(config.BENCHMARK_TICKER)

    meta_rows: list[dict] = []
    for t in all_tickers:
        df = cache.read_raw_prices(t)
        if df is None or df.empty:
            df = provider.get_daily_prices(t, start, end)
            if df is not None and not df.empty:
                cache.write_raw_prices(t, df)
        n = 0 if df is None else len(df)
        first = None if (df is None or df.empty) else pd.to_datetime(df.index.min()).date()
        last = None if (df is None or df.empty) else pd.to_datetime(df.index.max()).date()
        meta_rows.append(
            {
                "ticker": t,
                "n_days": n,
                "first_date": first,
                "last_date": last,
                "valid_history": bool(n >= MIN_VALID_DAYS) and t != config.BENCHMARK_TICKER,
                "is_benchmark": t == config.BENCHMARK_TICKER,
            }
        )

    prices = cache.build_prices_table(all_tickers)
    cache.write_parquet(prices, config.PRICES_PARQUET)

    meta = pd.DataFrame(meta_rows)
    # attach sector from fundamentals if already ingested
    fund = cache.read_parquet(config.FUNDAMENTALS_PARQUET)
    if fund is not None and "sector" in fund.columns:
        meta = meta.merge(fund[["ticker", "sector"]], on="ticker", how="left")
    cache.write_parquet(meta, config.METADATA_PARQUET)

    cache.register_processed_in_duckdb()
    return prices


# --------------------------------------------------------------------------- #
# Fundamentals
# --------------------------------------------------------------------------- #
def ingest_fundamentals(offline: bool = False,
                        provider: FundamentalsProvider | None = None,
                        tickers: list[str] | None = None) -> pd.DataFrame:
    """Cache fundamentals (sector at minimum) for all universe tickers."""
    config.ensure_dirs()
    provider = provider or default_fundamentals_provider(offline=offline)

    if tickers is None:
        uni_df = cache.read_parquet(config.CONSTITUENTS_MONTHLY_PARQUET)
        if uni_df is not None and not uni_df.empty:
            tickers = sorted(uni_df["ticker"].unique().tolist())
        else:
            tickers = universe.current_constituents()

    rows = [provider.get_fundamentals(t) for t in tickers]
    df = pd.DataFrame(rows)
    cache.write_parquet(df, config.FUNDAMENTALS_PARQUET)

    # refresh metadata sector column if metadata exists
    meta = cache.read_parquet(config.METADATA_PARQUET)
    if meta is not None and not meta.empty:
        meta = meta.drop(columns=[c for c in ["sector"] if c in meta.columns])
        meta = meta.merge(df[["ticker", "sector"]], on="ticker", how="left")
        cache.write_parquet(meta, config.METADATA_PARQUET)
    return df


def ingest_all(years: int = config.SIM_YEARS, offline: bool = False,
               constituents_mode: str = "auto") -> None:
    """Convenience: full ingestion pipeline in dependency order."""
    ingest_constituents(years=years, offline=offline, mode=constituents_mode)
    ingest_fundamentals(offline=offline)
    ingest_prices(years=years, offline=offline)
