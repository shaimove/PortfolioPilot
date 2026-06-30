"""Shared test fixtures.

Ensures deterministic offline data exists for tests that need the full pipeline
(prices, features, constituents). Builds it once per session if missing.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfoliopilot import config  # noqa: E402
from portfoliopilot.data import cache, ingestion  # noqa: E402
from portfoliopilot.features.feature_engine import build_all_features  # noqa: E402


@pytest.fixture(scope="session")
def offline_data():
    """Build synthetic constituents/prices/fundamentals/features if absent."""
    need = (
        cache.read_parquet(config.MONTHLY_FEATURES_PARQUET) is None
        or cache.read_parquet(config.CONSTITUENTS_MONTHLY_PARQUET) is None
        or cache.read_parquet(config.PRICES_PARQUET) is None
    )
    if need:
        ingestion.ingest_constituents(offline=True, mode="synthetic")
        ingestion.ingest_fundamentals(offline=True)
        ingestion.ingest_prices(offline=True)
        build_all_features()
    return True
