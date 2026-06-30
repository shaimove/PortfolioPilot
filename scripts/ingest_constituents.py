"""Build and cache the point-in-time S&P 500 monthly membership table.

Examples:
    python scripts/ingest_constituents.py --years 10
    python scripts/ingest_constituents.py --years 10 --offline   # synthetic, with changes
    python scripts/ingest_constituents.py --mode current         # survivorship-biased fallback
"""
import _bootstrap  # noqa: F401
import argparse

from portfoliopilot.data import ingestion


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest S&P 500 constituents (monthly membership).")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--offline", action="store_true", help="use deterministic synthetic membership")
    ap.add_argument("--mode", choices=["auto", "pit", "synthetic", "current"], default="auto")
    args = ap.parse_args()

    df = ingestion.ingest_constituents(years=args.years, offline=args.offline, mode=args.mode)
    months = df["month_end"].nunique()
    tickers = df["ticker"].nunique()
    print(f"Wrote constituents_monthly: {len(df)} rows across {months} months, {tickers} tickers.")
    if args.mode == "current" or (args.mode == "auto" and not args.offline):
        print("NOTE: using current-constituents fallback -> SURVIVORSHIP BIAS. "
              "Replace with point-in-time data for unbiased results.")


if __name__ == "__main__":
    main()
