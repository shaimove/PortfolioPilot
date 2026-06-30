"""Build monthly per-stock features from local daily prices (no look-ahead).

    python scripts/build_features.py
"""
import _bootstrap  # noqa: F401

from portfoliopilot.features.feature_engine import build_all_features


def main() -> None:
    df = build_all_features()
    if df.empty:
        print("WARNING: no features built. Did you run ingest_prices / ingest_constituents?")
    else:
        print(f"Wrote monthly_features.parquet: {len(df)} rows, "
              f"{df['month_end'].nunique()} months, {df['ticker'].nunique()} tickers.")


if __name__ == "__main__":
    main()
