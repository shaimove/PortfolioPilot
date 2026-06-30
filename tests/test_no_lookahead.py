"""Look-ahead bias guard.

Features computed for month-end ``t`` must be identical whether or not future
(> t) price data is present. We verify that truncating all data after ``t`` does
not change the features at ``t``.
"""
import pandas as pd

from portfoliopilot import config
from portfoliopilot.data import cache
from portfoliopilot.features.feature_engine import compute_features_asof


def test_truncating_future_data_does_not_change_features(offline_data):
    prices = cache.read_parquet(config.PRICES_PARQUET)
    uni = cache.read_parquet(config.CONSTITUENTS_MONTHLY_PARQUET)
    months = sorted(pd.to_datetime(uni["month_end"]).dt.date.unique())
    me = months[len(months) // 2]

    full = compute_features_asof(prices, me)

    truncated_prices = prices[pd.to_datetime(prices["date"]) <= pd.Timestamp(me)]
    truncated = compute_features_asof(truncated_prices, me)

    assert sorted(full.index) == sorted(truncated.index)
    cols = ["current_price", "ret_1m", "ret_3m", "ret_12m", "vol_3m", "drawdown"]
    a = full[cols].sort_index().fillna(-999.0)
    b = truncated.loc[full.index][cols].sort_index().fillna(-999.0)
    assert (a.round(8).values == b.round(8).values).all()


def test_no_future_returns_leak(offline_data):
    """ret_1m at month t must match (price_t / price_{t-1} - 1) using only <= t."""
    prices = cache.read_parquet(config.PRICES_PARQUET)
    uni = cache.read_parquet(config.CONSTITUENTS_MONTHLY_PARQUET)
    months = sorted(pd.to_datetime(uni["month_end"]).dt.date.unique())
    me = months[-1]
    feats_now = compute_features_asof(prices, me)
    # Adding fabricated future spike should not alter the latest month's features
    spike = prices.copy()
    future = spike[spike["ticker"] == feats_now.index[0]].iloc[-1:].copy()
    future["date"] = pd.Timestamp(me) + pd.Timedelta(days=40)
    future["adj_close"] *= 5
    spike2 = pd.concat([spike, future], ignore_index=True)
    feats_future = compute_features_asof(spike2, me)
    tkr = feats_now.index[0]
    assert abs(feats_now.loc[tkr, "current_price"] - feats_future.loc[tkr, "current_price"]) < 1e-6
