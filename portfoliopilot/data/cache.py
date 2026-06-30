"""Local cache helpers.

All persistence goes through here: raw per-ticker price CSV/Parquet caches,
processed Parquet tables, and a DuckDB registration helper. The simulation reads
only from these local files (never from a provider).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .. import config


# --------------------------------------------------------------------------- #
# Raw per-ticker price cache
# --------------------------------------------------------------------------- #
def raw_price_path(ticker: str) -> Path:
    safe = ticker.replace("^", "_").replace("/", "_")
    return config.RAW_PRICES / f"{safe}.parquet"


def write_raw_prices(ticker: str, df: pd.DataFrame) -> None:
    config.ensure_dirs()
    out = df.copy()
    out = out.reset_index()
    out.to_parquet(raw_price_path(ticker), index=False)


def read_raw_prices(ticker: str) -> pd.DataFrame | None:
    p = raw_price_path(ticker)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def has_raw_prices(ticker: str) -> bool:
    return raw_price_path(ticker).exists()


# --------------------------------------------------------------------------- #
# Processed parquet tables
# --------------------------------------------------------------------------- #
def write_parquet(df: pd.DataFrame, path: Path) -> None:
    config.ensure_dirs()
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# Consolidated prices table (long format)
# --------------------------------------------------------------------------- #
def build_prices_table(tickers: list[str]) -> pd.DataFrame:
    """Concatenate raw per-ticker caches into one long table.

    Columns: date, ticker, open, high, low, close, adj_close, volume.
    """
    frames = []
    for t in tickers:
        df = read_raw_prices(t)
        if df is None or df.empty:
            continue
        d = df.copy().reset_index()
        d["ticker"] = t
        frames.append(d)
    if not frames:
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        )
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# DuckDB registration (convenience for ad-hoc local querying)
# --------------------------------------------------------------------------- #
def register_processed_in_duckdb() -> None:
    """(Re)create DuckDB views over the processed parquet files.

    Purely a local convenience; the simulation uses pandas readers above.
    """
    config.ensure_dirs()
    con = duckdb.connect(str(config.DUCKDB_PATH))
    try:
        mapping = {
            "prices": config.PRICES_PARQUET,
            "monthly_features": config.MONTHLY_FEATURES_PARQUET,
            "constituents_monthly": config.CONSTITUENTS_MONTHLY_PARQUET,
            "fundamentals": config.FUNDAMENTALS_PARQUET,
            "metadata": config.METADATA_PARQUET,
        }
        for name, path in mapping.items():
            if Path(path).exists():
                con.execute(
                    f"CREATE OR REPLACE VIEW {name} AS "
                    f"SELECT * FROM read_parquet('{path.as_posix()}')"
                )
    finally:
        con.close()
