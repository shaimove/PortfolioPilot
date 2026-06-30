"""Download (or synthesize) and cache daily adjusted OHLCV for the universe.

Examples:
    python scripts/ingest_prices.py --years 10
    python scripts/ingest_prices.py --years 10 --offline   # deterministic synthetic prices
"""
import _bootstrap  # noqa: F401
import argparse

from portfoliopilot.data import ingestion


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest daily adjusted OHLCV prices.")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--offline", action="store_true", help="use deterministic synthetic prices")
    args = ap.parse_args()

    prices = ingestion.ingest_prices(years=args.years, offline=args.offline)
    if prices.empty:
        print("WARNING: no prices ingested. If online, check connectivity or use --offline.")
    else:
        print(f"Wrote prices.parquet: {len(prices)} rows, {prices['ticker'].nunique()} tickers.")


if __name__ == "__main__":
    main()
