"""Small shared utilities: calendar helpers and JSON-safe coercion."""
from __future__ import annotations

import datetime as dt
import math
from typing import Any

import numpy as np
import pandas as pd


def month_end_dates(start: dt.date, periods: int) -> list[dt.date]:
    """Return `periods` consecutive month-end calendar dates starting at the
    month-end on/after `start`."""
    idx = pd.date_range(start=pd.Timestamp(start), periods=periods, freq="ME")
    return [d.date() for d in idx]


def simulation_month_ends(end: dt.date | None = None, total_months: int = 120) -> list[dt.date]:
    """Month-end dates for the simulation window, anchored so the last month is
    the most recent completed month on/before `end` (defaults to today)."""
    if end is None:
        end = dt.date.today()
    last = (pd.Timestamp(end) + pd.offsets.MonthEnd(0)).normalize()
    if last.date() > end:
        last = (pd.Timestamp(end) - pd.offsets.MonthEnd(1)).normalize()
    idx = pd.date_range(end=last, periods=total_months, freq="ME")
    return [d.date() for d in idx]


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars and NaNs to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if math.isnan(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, (pd.Timestamp, dt.date, dt.datetime)):
        return obj.isoformat()[:10]
    return obj


def safe_round(x: float | None, ndigits: int = 4) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return round(float(x), ndigits)
