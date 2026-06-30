"""Thin, optional OpenAI client wrapper.

Everything degrades gracefully: if no API key or the SDK is missing/erroring,
``available`` is False and callers fall back to deterministic logic. Returns
text plus token/latency usage for monitoring.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from . import config


@dataclass
class LLMResponse:
    text: str
    ok: bool
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    raw: dict = field(default_factory=dict)


# very rough price table (USD per 1K tokens); only used for monitoring estimates
_PRICE_PER_1K = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.005, 0.015),
}


def available() -> bool:
    return config.LLM.enabled


def _estimate_cost(model: str, pt: int, ct: int) -> float:
    pin, pout = _PRICE_PER_1K.get(model, (0.0, 0.0))
    return (pt / 1000.0) * pin + (ct / 1000.0) * pout


def chat_json(system_prompt: str, user_prompt: str,
              model: str | None = None, temperature: float = 0.1) -> LLMResponse:
    """Call the chat API requesting a JSON object. Returns LLMResponse.

    Never raises; on any failure returns ok=False so callers can fall back.
    """
    model = model or config.LLM.model
    if not config.LLM.enabled:
        return LLMResponse(text="", ok=False, error="LLM not configured")

    start = time.perf_counter()
    try:
        from openai import OpenAI

        client = OpenAI(api_key=config.LLM.openai_api_key)
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency = (time.perf_counter() - start) * 1000.0
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        tt = getattr(usage, "total_tokens", pt + ct) or (pt + ct)
        return LLMResponse(
            text=text, ok=True, latency_ms=latency,
            prompt_tokens=pt, completion_tokens=ct, total_tokens=tt,
            cost_usd=_estimate_cost(model, pt, ct),
        )
    except Exception as e:  # network, auth, rate limit, sdk missing, etc.
        latency = (time.perf_counter() - start) * 1000.0
        return LLMResponse(text="", ok=False, latency_ms=latency, error=str(e))


def extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction from a model response."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # try to find the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None
