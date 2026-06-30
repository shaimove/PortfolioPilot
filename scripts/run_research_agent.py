"""Demo the EXPERIMENTAL fundamental research agent (NOT part of the simulation).

Reads (synthetic) company reports and prints stock recommendations. Works fully
offline via the deterministic fallback; uses the LLM if OPENAI_API_KEY is set.

    python scripts/run_research_agent.py
    python scripts/run_research_agent.py --tickers AAPL MSFT JPM XOM NVDA
"""
import _bootstrap  # noqa: F401
import argparse
import datetime as dt
import json

from portfoliopilot.agent.research_agent import ACTIVATED, ResearchAgent
from portfoliopilot.data.universe import current_constituents


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the experimental research agent.")
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="tickers to analyze (default: a sample of the universe)")
    ap.add_argument("--top", type=int, default=10, help="how many recommendations to print")
    args = ap.parse_args()

    tickers = args.tickers or current_constituents()[:12]
    agent = ResearchAgent()
    reports = agent.reports_for(tickers)
    result = agent.recommend(reports, as_of=dt.date.today().isoformat())

    print(f"[research-agent ACTIVATED={ACTIVATED}] source={result.source}")
    print("NOTE: experimental, not wired into the simulation. Not financial advice.\n")
    print(result.output["summary"], "\n")
    for rec in result.output["recommendations"][: args.top]:
        print(f"  {rec['ticker']:<6} {rec['stance']:<5} conv={rec['conviction']:.2f}  "
              f"risks={len(rec['risks'])}")
    print("\nFull JSON of top recommendation:")
    print(json.dumps(result.output["recommendations"][0], indent=2))


if __name__ == "__main__":
    main()
