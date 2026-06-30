"""Central configuration: filesystem layout, simulation constants, and
risk/portfolio constraints. Everything reads from here so paths stay consistent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PRICES = RAW_DIR / "prices"
RAW_FUNDAMENTALS = RAW_DIR / "fundamentals"
RAW_CONSTITUENTS = RAW_DIR / "constituents"
RAW_CORPORATE_ACTIONS = RAW_DIR / "corporate_actions"

PROCESSED_DIR = DATA_DIR / "processed"
PRICES_PARQUET = PROCESSED_DIR / "prices.parquet"
MONTHLY_FEATURES_PARQUET = PROCESSED_DIR / "monthly_features.parquet"
CONSTITUENTS_MONTHLY_PARQUET = PROCESSED_DIR / "constituents_monthly.parquet"
FUNDAMENTALS_PARQUET = PROCESSED_DIR / "fundamentals.parquet"
METADATA_PARQUET = PROCESSED_DIR / "metadata.parquet"

DUCKDB_PATH = DATA_DIR / "portfoliopilot.duckdb"
MONITOR_DB_PATH = DATA_DIR / "monitor.sqlite"
MEMORY_DB_PATH = DATA_DIR / "memory.sqlite"
CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.sqlite"

DASHBOARD_DIR = ROOT / "dashboard"


def ensure_dirs() -> None:
    """Create the full data directory tree if it does not yet exist."""
    for p in (
        DATA_DIR,
        RAW_DIR,
        RAW_PRICES,
        RAW_FUNDAMENTALS,
        RAW_CONSTITUENTS,
        RAW_CORPORATE_ACTIONS,
        PROCESSED_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Simulation constants
# --------------------------------------------------------------------------- #
SIM_YEARS = 10
MONTHS_PER_YEAR = 12
TOTAL_MONTHS = SIM_YEARS * MONTHS_PER_YEAR  # 120 monthly steps

STARTING_CAPITAL = 100_000.0
TRANSACTION_COST_BPS = 10.0  # 10 basis points per traded dollar
BENCHMARK_TICKER = "^GSPC"   # benchmark only, never tradable

DEFAULT_SECONDS_PER_MONTH = 30.0
MIN_SECONDS_PER_MONTH = 1.0


@dataclass
class Constraints:
    """Portfolio constraints handed to the agent and enforced by the risk engine."""
    max_asset_weight: float = 0.08
    max_sector_weight: float = 0.30
    max_turnover_per_month: float = 0.30
    min_cash: float = 0.02
    max_cash: float = 0.30
    long_only: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# LLM / tracing configuration (all optional; system runs without them)
# --------------------------------------------------------------------------- #
@dataclass
class LLMConfig:
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    model: str = field(default_factory=lambda: os.getenv("PORTFOLIOPILOT_MODEL", "gpt-4o-mini"))
    agent_version: str = "decision-agent-v1"
    judge_version: str = "judge-v1"
    prompt_version: str = "prompt-v1"
    max_retries: int = 2

    @property
    def enabled(self) -> bool:
        return bool(self.openai_api_key)


@dataclass
class LangSmithConfig:
    api_key: str | None = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY"))
    project: str = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "portfoliopilot"))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


DEFAULT_CONSTRAINTS = Constraints()
LLM = LLMConfig()
LANGSMITH = LangSmithConfig()
