"""Cache fundamentals (sector classification at minimum) for the universe.

Examples:
    python scripts/ingest_fundamentals.py
    python scripts/ingest_fundamentals.py --offline   # deterministic stub sectors
"""
import _bootstrap  # noqa: F401
import argparse

from portfoliopilot.data import ingestion


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest fundamentals (sector, etc.).")
    ap.add_argument("--offline", action="store_true", help="use deterministic stub provider")
    args = ap.parse_args()

    df = ingestion.ingest_fundamentals(offline=args.offline)
    print(f"Wrote fundamentals.parquet: {len(df)} tickers.")


if __name__ == "__main__":
    main()
