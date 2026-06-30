"""Monthly feature engine.

Computes per-stock monthly features from local daily prices. Strictly avoids
look-ahead bias: features for month-end ``t`` use only price/volume data dated
on or before ``t``.

Stock features per (ticker, month_end):
    ret_1m, ret_3m, ret_6m, ret_12m,
    vol_3m, vol_6m, drawdown, ma_trend, volume_trend,
    current_price, valid_history, sector

Portfolio-level features (portfolio value, turnover, excess return, etc.) are
computed during simulation in ``monitoring.metrics`` since they depend on agent
decisions, not just market data.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .. import config
from ..data import cache

MIN_MONTHS_FOR_VALID = 12


def _prices_wide(prices: pd.DataFrame, field: str = "adj_close") -> pd.DataFrame:
    """Pivot the long prices table to a wide [date x ticker] frame for `field`."""
    if prices is None or prices.empty:
        return pd.DataFrame()
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    wide = p.pivot_table(index="date", columns="ticker", values=field, aggfunc="last")
    return wide.sort_index()


def _monthly_asof(wide: pd.DataFrame, month_ends: list[dt.date]) -> pd.DataFrame:
    """Value of each series as of each month-end (last obs on/before month_end).

    Reindex onto daily then forward-fill, then select the month-end rows. This
    naturally uses only past data per row.
    """
    if wide.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([d for d in month_ends]))
    me_index = pd.DatetimeIndex([pd.Timestamp(d) for d in month_ends])
    full_index = wide.index.union(me_index)
    monthly = wide.reindex(full_index).ffill().reindex(me_index)
    monthly.index = me_index
    return monthly


def _trend_label(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    return "positive" if value >= 0 else "negative"


def compute_features_asof(prices: pd.DataFrame,
                          month_end: dt.date,
                          sectors: dict[str, str] | None = None,
                          valid_flags: dict[str, bool] | None = None,
                          lookback_months: int = 18) -> pd.DataFrame:
    """Compute stock features for a single month-end using only data <= month_end.

    Returns a DataFrame indexed by ticker.
    """
    sectors = sectors or {}
    valid_flags = valid_flags or {}

    # Restrict to data up to month_end (the core no-look-ahead guarantee)
    p = prices[pd.to_datetime(prices["date"]) <= pd.Timestamp(month_end)]
    if p.empty:
        return pd.DataFrame()

    adj = _prices_wide(p, "adj_close")
    vol = _prices_wide(p, "volume")

    # build the trailing set of month-ends up to and including month_end
    me_list = [d.date() for d in pd.date_range(end=pd.Timestamp(month_end),
                                               periods=lookback_months, freq="ME")]
    monthly_px = _monthly_asof(adj, me_list)
    monthly_vol = _monthly_asof(vol, me_list)

    if monthly_px.empty:
        return pd.DataFrame()

    monthly_ret = monthly_px.pct_change()

    rows: list[dict] = []
    cur_ts = pd.Timestamp(month_end)
    for ticker in monthly_px.columns:
        series = monthly_px[ticker]
        if cur_ts not in series.index:
            continue
        current_price = series.loc[cur_ts]
        if pd.isna(current_price):
            continue

        def ret_n(n: int) -> float | None:
            if len(series) <= n:
                return None
            past = series.iloc[-(n + 1)]
            if pd.isna(past) or past == 0:
                return None
            return float(current_price / past - 1.0)

        rser = monthly_ret[ticker].dropna()
        vol_3m = float(rser.iloc[-3:].std()) if len(rser) >= 3 else None
        vol_6m = float(rser.iloc[-6:].std()) if len(rser) >= 6 else None

        # drawdown vs trailing 12m peak
        trailing = series.iloc[-12:].dropna()
        if len(trailing) >= 2:
            peak = trailing.max()
            drawdown = float(current_price / peak - 1.0) if peak > 0 else None
        else:
            drawdown = None

        # moving-average trend: price vs trailing 6m mean
        ma6 = series.iloc[-6:].mean() if series.iloc[-6:].notna().sum() >= 3 else np.nan
        ma_trend = _trend_label(float(current_price - ma6)) if not pd.isna(ma6) else "unknown"

        # volume trend: last month avg vs trailing 6m avg
        vseries = monthly_vol[ticker] if ticker in monthly_vol.columns else pd.Series(dtype=float)
        v_recent = vseries.iloc[-1] if len(vseries) else np.nan
        v_base = vseries.iloc[-6:].mean() if len(vseries) >= 3 else np.nan
        if not pd.isna(v_recent) and not pd.isna(v_base) and v_base > 0:
            volume_trend = float(v_recent / v_base - 1.0)
        else:
            volume_trend = None

        n_valid_months = int(series.notna().sum())
        valid_history = bool(
            valid_flags.get(ticker, True)
            and n_valid_months >= MIN_MONTHS_FOR_VALID
            and ret_n(12) is not None
        )

        rows.append(
            {
                "ticker": ticker,
                "month_end": month_end,
                "current_price": float(current_price),
                "ret_1m": ret_n(1),
                "ret_3m": ret_n(3),
                "ret_6m": ret_n(6),
                "ret_12m": ret_n(12),
                "vol_3m": vol_3m,
                "vol_6m": vol_6m,
                "drawdown": drawdown,
                "ma_trend": ma_trend,
                "volume_trend": volume_trend,
                "valid_history": valid_history,
                "sector": sectors.get(ticker),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("ticker")


def build_all_features(month_ends: list[dt.date] | None = None) -> pd.DataFrame:
    """Build and cache monthly_features.parquet for every simulation month-end.

    Reads local prices/metadata only.
    """
    prices = cache.read_parquet(config.PRICES_PARQUET)
    if prices is None or prices.empty:
        raise FileNotFoundError("prices.parquet not found. Run ingest_prices first.")

    meta = cache.read_parquet(config.METADATA_PARQUET)
    sectors: dict[str, str] = {}
    valid_flags: dict[str, bool] = {}
    if meta is not None and not meta.empty:
        if "sector" in meta.columns:
            sectors = {r.ticker: r.sector for r in meta.itertuples() if pd.notna(getattr(r, "sector", None))}
        valid_flags = {r.ticker: bool(getattr(r, "valid_history", True)) for r in meta.itertuples()}

    if month_ends is None:
        uni = cache.read_parquet(config.CONSTITUENTS_MONTHLY_PARQUET)
        if uni is None or uni.empty:
            raise FileNotFoundError("constituents_monthly.parquet not found.")
        month_ends = sorted(pd.to_datetime(uni["month_end"]).dt.date.unique().tolist())

    frames = []
    for me in month_ends:
        feats = compute_features_asof(prices, me, sectors=sectors, valid_flags=valid_flags)
        if not feats.empty:
            frames.append(feats.reset_index())

    if not frames:
        out = pd.DataFrame()
    else:
        out = pd.concat(frames, ignore_index=True)
    cache.write_parquet(out, config.MONTHLY_FEATURES_PARQUET)
    cache.register_processed_in_duckdb()
    return out


class FeatureStore:
    """Read-only accessor over cached monthly features (used by the simulation)."""

    def __init__(self, df: pd.DataFrame) -> None:
        d = df.copy()
        d["month_end"] = pd.to_datetime(d["month_end"]).dt.date
        self._df = d
        self._by_month = {me: g.set_index("ticker") for me, g in d.groupby("month_end")}

    @classmethod
    def load(cls) -> "FeatureStore":
        df = cache.read_parquet(config.MONTHLY_FEATURES_PARQUET)
        if df is None or df.empty:
            raise FileNotFoundError("monthly_features.parquet not found. Run build_features.")
        return cls(df)

    def features_on(self, month_end: dt.date) -> pd.DataFrame:
        g = self._by_month.get(month_end)
        return g.copy() if g is not None else pd.DataFrame()

    def price_on(self, month_end: dt.date, ticker: str) -> float | None:
        g = self._by_month.get(month_end)
        if g is None or ticker not in g.index:
            return None
        v = g.loc[ticker, "current_price"]
        return None if pd.isna(v) else float(v)
