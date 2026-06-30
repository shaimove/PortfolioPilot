"""Simple memory retrieval.

Scores active memories by ticker overlap, type relevance, recency, and keyword
overlap, then returns the top-k. Deliberately NOT a vector store for the MVP.
"""
from __future__ import annotations

import datetime as dt
import re

from .memory_store import Memory, MemoryStore

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _recency_score(date_created: str, as_of: dt.date) -> float:
    try:
        created = dt.date.fromisoformat(date_created)
    except Exception:
        return 0.0
    days = max(0, (as_of - created).days)
    # 1.0 today, decaying with a ~2-year half-life
    return 0.5 ** (days / 730.0)


def retrieve(
    store: MemoryStore,
    as_of: dt.date,
    tickers: list[str] | None = None,
    keywords: str = "",
    types: set[str] | None = None,
    k: int = 5,
    include_stale: bool = False,
) -> list[Memory]:
    tickers = [t.upper() for t in (tickers or [])]
    kw = _tokens(keywords)
    scored: list[tuple[float, Memory]] = []

    for mem in store.all():
        if not include_stale and mem.status != "active":
            continue
        # respect validity window
        if mem.valid_until and mem.valid_until < as_of.isoformat():
            if not include_stale:
                continue

        score = 0.0
        rel = {t.upper() for t in mem.related_assets}
        if tickers and rel:
            overlap = len(rel & set(tickers))
            score += 2.0 * overlap
        if types and mem.type in types:
            score += 1.0
        score += 1.5 * _recency_score(mem.date_created, as_of)
        if kw:
            score += 0.5 * len(_tokens(mem.content) & kw)
        score += 0.5 * float(mem.confidence)

        if score > 0:
            scored.append((score, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:k]]
