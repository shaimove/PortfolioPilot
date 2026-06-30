import pandas as pd

from portfoliopilot import config
from portfoliopilot.data import cache
from portfoliopilot.features.feature_engine import (
    FeatureStore,
    build_all_features,
    compute_features_asof,
)


def test_features_built_and_have_expected_columns(offline_data):
    df = cache.read_parquet(config.MONTHLY_FEATURES_PARQUET)
    assert df is not None and not df.empty
    expected = {
        "ticker", "month_end", "current_price", "ret_1m", "ret_3m", "ret_6m",
        "ret_12m", "vol_3m", "vol_6m", "drawdown", "ma_trend", "volume_trend",
        "valid_history", "sector",
    }
    assert expected.issubset(set(df.columns))


def test_feature_store_returns_month_slice(offline_data):
    store = FeatureStore.load()
    uni = cache.read_parquet(config.CONSTITUENTS_MONTHLY_PARQUET)
    me = sorted(pd.to_datetime(uni["month_end"]).dt.date.unique())[-1]
    feats = store.features_on(me)
    assert not feats.empty
    assert feats["current_price"].notna().all()


def test_valid_history_flag_present(offline_data):
    df = cache.read_parquet(config.MONTHLY_FEATURES_PARQUET)
    assert df["valid_history"].dtype == bool or set(df["valid_history"].unique()).issubset({True, False})


def test_asof_uses_only_past_prices(offline_data):
    prices = cache.read_parquet(config.PRICES_PARQUET)
    uni = cache.read_parquet(config.CONSTITUENTS_MONTHLY_PARQUET)
    months = sorted(pd.to_datetime(uni["month_end"]).dt.date.unique())
    me = months[len(months) // 2]
    feats = compute_features_asof(prices, me)
    # current price must equal last adj_close on/before me for some known ticker
    assert not feats.empty
    tkr = feats.index[0]
    sub = prices[(prices["ticker"] == tkr) &
                 (pd.to_datetime(prices["date"]) <= pd.Timestamp(me))]
    assert abs(feats.loc[tkr, "current_price"] - sub.iloc[-1]["adj_close"]) < 1e-6
