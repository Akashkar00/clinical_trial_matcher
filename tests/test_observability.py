"""Tests for observability cost tracker."""
from unittest.mock import MagicMock

from observability import CostTracker, estimate_cost, track_call


def _resp(prompt_tokens: int, completion_tokens: int):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp = MagicMock()
    resp.usage = usage
    return resp


def test_estimate_cost_known_model():
    cost = estimate_cost("llama-3.3-70b-versatile", 1_000_000, 0)
    assert cost == 0.59
    cost = estimate_cost("llama-3.3-70b-versatile", 0, 1_000_000)
    assert cost == 0.79


def test_estimate_cost_unknown_model_zero():
    assert estimate_cost("gpt-fictional", 1_000, 1_000) == 0.0


def test_track_call_records_usage():
    from observability import tracker
    tracker.reset()
    with track_call("scoring", "llama-3.3-70b-versatile") as ctx:
        ctx["response"] = _resp(prompt_tokens=500, completion_tokens=200)
    snap = tracker.snapshot()
    assert snap["total_calls"] == 1
    assert snap["stages"]["scoring"]["prompt_tokens"] == 500
    assert snap["stages"]["scoring"]["completion_tokens"] == 200
    expected = 500 / 1e6 * 0.59 + 200 / 1e6 * 0.79
    assert snap["total_cost_usd"] == round(expected, 6)


def test_track_call_handles_missing_usage():
    t = CostTracker()
    # CostTracker does not register itself — patch the singleton path
    from observability import tracker as global_tracker
    global_tracker.reset()
    with track_call("extraction", "llama-3.3-70b-versatile") as ctx:
        ctx["response"] = MagicMock(usage=None)
    snap = global_tracker.snapshot()
    # Latency recorded even though tokens are zero
    assert snap["total_calls"] == 1
    assert snap["total_tokens"] == 0


def test_tracker_aggregates_multiple_stages():
    from observability import tracker
    tracker.reset()
    with track_call("extraction", "llama-3.3-70b-versatile") as ctx:
        ctx["response"] = _resp(100, 50)
    with track_call("scoring", "llama-3.3-70b-versatile") as ctx:
        ctx["response"] = _resp(200, 100)
    with track_call("scoring", "llama-3.3-70b-versatile") as ctx:
        ctx["response"] = _resp(300, 150)
    snap = tracker.snapshot()
    assert snap["total_calls"] == 3
    assert snap["stages"]["extraction"]["calls"] == 1
    assert snap["stages"]["scoring"]["calls"] == 2
    assert snap["stages"]["scoring"]["prompt_tokens"] == 500


def test_tracker_report_formats_table():
    from observability import tracker
    tracker.reset()
    with track_call("scoring", "llama-3.3-70b-versatile") as ctx:
        ctx["response"] = _resp(100, 100)
    report = tracker.report()
    assert "scoring" in report
    assert "TOTAL" in report
    assert "USD" in report
