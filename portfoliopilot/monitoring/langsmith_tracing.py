"""Optional LangSmith tracing.

If LANGSMITH_API_KEY is set and the langsmith SDK imports, monthly rebalance
runs (agent + judge) are logged. Otherwise every method is a safe no-op so the
app never crashes when LangSmith is missing.
"""
from __future__ import annotations

from typing import Any

from .. import config


class Tracer:
    def __init__(self) -> None:
        self._client = None
        self._enabled = False
        if config.LANGSMITH.enabled:
            try:
                from langsmith import Client

                self._client = Client(api_key=config.LANGSMITH.api_key)
                self._enabled = True
            except Exception:
                self._client = None
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def status(self) -> dict:
        if self._enabled:
            return {"enabled": True, "project": config.LANGSMITH.project}
        if config.LANGSMITH.api_key:
            return {"enabled": False, "reason": "langsmith sdk unavailable"}
        return {"enabled": False, "reason": "LANGSMITH_API_KEY not set"}

    def log_month(self, name: str, inputs: dict, outputs: dict,
                  metadata: dict[str, Any] | None = None,
                  error: str | None = None) -> None:
        if not self._enabled or self._client is None:
            return
        try:
            self._client.create_run(
                name=name,
                run_type="chain",
                inputs=inputs,
                outputs=outputs,
                error=error,
                project_name=config.LANGSMITH.project,
                extra={"metadata": metadata or {}},
            )
        except Exception:
            # never let tracing failures affect the simulation
            pass
