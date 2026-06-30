"""Company-report data sources for the (experimental) research agent.

Provides a provider abstraction over company fundamental reports plus a
deterministic stub so the research agent can run fully offline. The design is
intentionally identical in shape to a real filing source so SEC EDGAR
Companyfacts, FMP, Alpha Vantage, or Tiingo can be dropped in later.

NOTE: This is used only by the *not-yet-activated* research agent
(`agent/research_agent.py`). The live monthly simulation does not consume it,
which keeps the main decision agent's input free of fundamental fields (so the
LLM-as-judge can still flag unsupported fundamental claims).
"""
from __future__ import annotations

import abc
import hashlib

_SECTORS = [
    "Technology", "Financials", "Health Care", "Consumer Discretionary",
    "Consumer Staples", "Industrials", "Energy", "Utilities",
    "Materials", "Real Estate", "Communication Services",
]


def _seed(ticker: str) -> int:
    return int(hashlib.sha256(ticker.encode("utf-8")).hexdigest()[:8], 16)


class CompanyReportProvider(abc.ABC):
    """Returns a normalized fundamental report for a ticker.

    A report is a flat dict with (at least) the keys produced by
    ``StubCompanyReportProvider``. Subclass this to wire a real filings source.
    """

    name: str = "base"

    @abc.abstractmethod
    def get_report(self, ticker: str) -> dict:
        raise NotImplementedError

    def get_reports(self, tickers: list[str]) -> list[dict]:
        return [self.get_report(t) for t in tickers]


class StubCompanyReportProvider(CompanyReportProvider):
    """Deterministic synthetic fundamentals (reproducible per ticker).

    Generates plausible-but-fake figures so the research agent has something to
    reason over offline. Numbers are derived from the ticker hash, so a given
    ticker always yields the same report.
    """

    name = "stub"

    def get_report(self, ticker: str) -> dict:
        import numpy as np

        rng = np.random.default_rng(_seed(ticker))
        sector = _SECTORS[_seed(ticker) % len(_SECTORS)]

        revenue = float(rng.uniform(1e9, 4e11))
        revenue_growth_yoy = float(rng.normal(0.08, 0.12))
        net_margin = float(np.clip(rng.normal(0.12, 0.08), -0.15, 0.45))
        net_income = revenue * net_margin
        earnings_growth_yoy = float(rng.normal(revenue_growth_yoy, 0.15))
        gross_margin = float(np.clip(net_margin + rng.uniform(0.1, 0.4), 0.05, 0.9))
        operating_margin = float(np.clip(net_margin + rng.uniform(0.0, 0.12), -0.1, 0.6))
        free_cash_flow = net_income * float(rng.uniform(0.4, 1.4))
        debt_to_equity = float(np.clip(rng.normal(0.8, 0.6), 0.0, 4.0))
        roe = float(np.clip(rng.normal(0.15, 0.1), -0.2, 0.6))
        pe_ratio = float(np.clip(rng.normal(22, 12), 5, 90))
        ps_ratio = float(np.clip(rng.normal(4, 3), 0.4, 30))

        return {
            "ticker": ticker,
            "sector": sector,
            "fiscal_period": "FY (synthetic)",
            "revenue": round(revenue, 2),
            "revenue_growth_yoy": round(revenue_growth_yoy, 4),
            "net_income": round(net_income, 2),
            "earnings_growth_yoy": round(earnings_growth_yoy, 4),
            "gross_margin": round(gross_margin, 4),
            "operating_margin": round(operating_margin, 4),
            "net_margin": round(net_margin, 4),
            "free_cash_flow": round(free_cash_flow, 2),
            "debt_to_equity": round(debt_to_equity, 4),
            "roe": round(roe, 4),
            "pe_ratio": round(pe_ratio, 2),
            "ps_ratio": round(ps_ratio, 2),
            "source": self.name,
        }


def default_report_provider() -> CompanyReportProvider:
    return StubCompanyReportProvider()
