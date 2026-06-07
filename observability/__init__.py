"""
Per-call token / cost telemetry for Groq.

Wrap any `groq_client.chat.completions.create(...)` call with `track_call`
to capture model, prompt+completion tokens, latency, and an estimated USD
cost. The aggregator keeps a thread-safe running total per stage so a
single pipeline run can report end-to-end spend.

Pricing comes from `MODEL_PRICING` (USD per 1M tokens). Update there if
Groq adjusts rates — wrong constants are the most common silent telemetry
bug.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# Groq pricing as of the model snapshot used in this project.
# Source: https://groq.com/pricing — verify before relying on cost figures.
# Numbers are USD per 1M tokens.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant":    {"input": 0.05, "output": 0.08},
}


@dataclass
class CallRecord:
    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class StageTotals:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms_total: float = 0.0
    cost_usd: float = 0.0

    def add(self, rec: CallRecord) -> None:
        self.calls += 1
        self.prompt_tokens += rec.prompt_tokens
        self.completion_tokens += rec.completion_tokens
        self.latency_ms_total += rec.latency_ms
        self.cost_usd += rec.cost_usd


class CostTracker:
    """Thread-safe accumulator. ThreadPoolExecutor is used in scoring, so
    locking matters — without it concurrent calls drop records."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stages: dict[str, StageTotals] = {}
        self._records: list[CallRecord] = []

    def record(self, rec: CallRecord) -> None:
        with self._lock:
            self._stages.setdefault(rec.stage, StageTotals()).add(rec)
            self._records.append(rec)

    def reset(self) -> None:
        with self._lock:
            self._stages.clear()
            self._records.clear()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = {
                name: {
                    "calls": t.calls,
                    "prompt_tokens": t.prompt_tokens,
                    "completion_tokens": t.completion_tokens,
                    "total_tokens": t.prompt_tokens + t.completion_tokens,
                    "latency_ms_total": round(t.latency_ms_total, 1),
                    "cost_usd": round(t.cost_usd, 6),
                }
                for name, t in self._stages.items()
            }
            total_cost = sum(t.cost_usd for t in self._stages.values())
            total_tokens = sum(
                t.prompt_tokens + t.completion_tokens for t in self._stages.values()
            )
            return {
                "stages": stages,
                "total_cost_usd": round(total_cost, 6),
                "total_tokens": total_tokens,
                "total_calls": sum(t.calls for t in self._stages.values()),
            }

    def report(self) -> str:
        snap = self.snapshot()
        lines = [
            "─" * 60,
            f"{'Stage':<14} {'Calls':>5} {'In':>7} {'Out':>7} {'ms':>7} {'USD':>10}",
            "─" * 60,
        ]
        for name, s in snap["stages"].items():
            lines.append(
                f"{name:<14} {s['calls']:>5} {s['prompt_tokens']:>7} "
                f"{s['completion_tokens']:>7} {s['latency_ms_total']:>7.0f} "
                f"${s['cost_usd']:>9.4f}"
            )
        lines.append("─" * 60)
        lines.append(
            f"{'TOTAL':<14} {snap['total_calls']:>5} {'':>7} {'':>7} "
            f"{'':>7} ${snap['total_cost_usd']:>9.4f}"
        )
        return "\n".join(lines)


# Module-level singleton — one tracker per process. For a service, swap to
# a per-request tracker via contextvars; for this CLI/Streamlit use case
# the singleton is fine and `reset()` between runs covers the same need.
tracker = CostTracker()


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        logger.debug("cost.unknown_model model=%s — cost reported as 0", model)
        return 0.0
    return (
        prompt_tokens / 1_000_000 * pricing["input"]
        + completion_tokens / 1_000_000 * pricing["output"]
    )


@contextmanager
def track_call(stage: str, model: str) -> Iterator[dict[str, Any]]:
    """Context manager that times the call and captures usage from the
    response. Usage:

        with track_call("scoring", GROQ_MODEL) as ctx:
            resp = groq_client.chat.completions.create(...)
            ctx["response"] = resp

    The tracker reads `ctx["response"].usage` after exit. If the response
    is missing or has no usage, latency is still recorded and tokens are 0.
    """
    ctx: dict[str, Any] = {"response": None}
    started = time.perf_counter()
    try:
        yield ctx
    finally:
        latency_ms = (time.perf_counter() - started) * 1_000
        resp = ctx.get("response")
        prompt_t = completion_t = 0
        usage = getattr(resp, "usage", None) if resp is not None else None
        if usage is not None:
            prompt_t = getattr(usage, "prompt_tokens", 0) or 0
            completion_t = getattr(usage, "completion_tokens", 0) or 0
        cost = estimate_cost(model, prompt_t, completion_t)
        tracker.record(CallRecord(
            stage=stage, model=model,
            prompt_tokens=prompt_t, completion_tokens=completion_t,
            latency_ms=latency_ms, cost_usd=cost,
        ))
