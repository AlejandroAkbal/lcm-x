"""#608: a spent threshold-sweep time budget is a stop condition, not an error.

Engine-level: a host list in, the returned list, the public status, the sweep telemetry and the log
lines out. Summaries are stubbed and counted; the sweep clock is a real monotonic clock plus an offset
that a wrapped engine step advances."""

from __future__ import annotations

import logging
import time

import pytest

import hermes_lcm.compaction as lcm_compaction
import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 30
RETRY_LINE = "retrying with smaller oldest chunk"
BUDGET_LINE = "spent its time budget before the first leaf"
CONDENSATION_LINE = "condensation stopped"


class _Clock:
    """time.monotonic() plus an offset the test advances inside an engine step."""

    def __init__(self):
        self._real = time.monotonic
        self.offset = 0.0

    def __call__(self) -> float:
        return self._real() + self.offset


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(lcm_compaction.time, "monotonic", fake)
    return fake


@pytest.fixture
def summaries(monkeypatch):
    calls: list[str] = []

    def summarize(**kwargs):
        calls.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return calls


def _engine(tmp_path, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "threshold_full_sweep_enabled": True, "max_assembly_tokens": 100_000,
                "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float) -> list[dict]:
    return [{"role": "user", "content": f"[{tag}] user turn{PAD}", "timestamp": ts},
            {"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _view(turns: int = 6) -> list[dict]:
    return [{"role": "system", "content": "system prompt"},
            *[row for i in range(turns) for row in _turn(f"T{i}", 10.0 * (i + 1))]]


def _advance_on_call(monkeypatch, engine, name: str, clock: _Clock, seconds: float, *, call: int = 1):
    """Wrap engine step ``name``: its ``call``-th invocation moves the sweep clock by ``seconds``."""
    original = getattr(engine, name)
    count = 0

    def wrapped(*args, **kwargs):
        nonlocal count
        result = original(*args, **kwargs)
        count += 1
        if count == call:
            clock.offset += seconds
        return result

    monkeypatch.setattr(engine, name, wrapped)


def _count(caplog, text: str) -> int:
    return sum(text in record.getMessage() for record in caplog.records)


def _spend_budget_before_first_leaf(engine, view, clock, monkeypatch, caplog):
    """Test 1's state: the first store-id map of pass 0 takes the whole budget."""
    _advance_on_call(monkeypatch, engine, "_get_store_id_map_for_messages", clock, 121.0)
    engine.ingest(view)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        return engine.compress(view, current_tokens=engine.threshold_tokens + 1)


# -- 2. five seconds left at the pre-call check ------------------------------------------------------------

def test_five_seconds_left_at_the_pre_call_check_makes_no_summariser_call(
        tmp_path, summaries, clock, monkeypatch, caplog):
    engine = _engine(tmp_path)
    view = _view()
    _advance_on_call(monkeypatch, engine, "_identity_anchor_summary_input", clock, 115.0)
    try:
        engine.ingest(view)
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert summaries == []
        assert result is view and telemetry["stop_reason"] == "time_budget_exhausted"
        assert _count(caplog, RETRY_LINE) == 0
        with pytest.raises(lcm_engine.SweepBudgetExhausted, match="threshold full sweep time budget exhausted"):
            engine._summarize_leaf_chunk_with_rescue(view[1:5], deadline=clock() + 5.0)
        assert summaries == []
    finally:
        engine.shutdown()


# -- 3. a provider timeout with budget left keeps the smaller-chunk retry -----------------------------------

def test_provider_timeout_with_budget_left_still_retries_a_smaller_chunk(tmp_path, clock, monkeypatch, caplog):
    engine = _engine(tmp_path)
    calls = []

    def summarize(**kwargs):
        calls.append(kwargs["timeout"])
        if len(calls) == 1:
            raise TimeoutError("provider timed out")
        return "Earlier turns.", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    chunk = [row for i in range(3) for row in _turn(f"R{i}", 10.0 * i)]
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            used, _tokens, _text, _level, attempts = engine._summarize_leaf_chunk_with_rescue(
                chunk, deadline=clock() + 100.0)
        assert len(calls) == 2 and attempts == 2 and len(used) < len(chunk)
        assert _count(caplog, RETRY_LINE) == 1
    finally:
        engine.shutdown()


# -- 4. leaves stored, then the budget ends at a pre-call check --------------------------------------------

@pytest.mark.parametrize("left_at_call", [5.0, -5.0], ids=["five-seconds-left", "five-seconds-over"])
def test_budget_ending_after_a_stored_leaf_keeps_the_partial_result(
        tmp_path, clock, monkeypatch, caplog, left_at_call):
    engine = _engine(tmp_path, leaf_chunk_tokens=200)
    view = _view(8)
    calls = []

    def summarize(**kwargs):
        calls.append(1)
        clock.offset += 110.0  # the first leaf takes most of the budget
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    _advance_on_call(monkeypatch, engine, "_identity_anchor_summary_input", clock, 10.0 - left_at_call, call=2)
    try:
        engine.ingest(view)
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert calls == [1] and result is not view and len(engine._dag.get_session_nodes("S")) == 1
        assert engine._last_compression_status == "compacted"
        assert telemetry["leaf_passes"] == 1 and telemetry["status"] == "partial"
        assert telemetry["stop_reason"] == "time_budget_exhausted" and telemetry["budget_exhausted"] is True
        assert "leaf_summary_error" not in caplog.text and _count(caplog, "sweep stopped after") == 0
        assert _count(caplog, RETRY_LINE) == 0 and _count(caplog, BUDGET_LINE) == 0
    finally:
        engine.shutdown()


# -- 5. condensation: the budget ends at its pre-call check -------------------------------------------------

@pytest.mark.parametrize("advance", [0.0, 10.0], ids=["five-seconds-left", "deadline-passes-in-group-selection"])
def test_condensation_budget_end_is_a_stop_reason_without_a_warning(
        tmp_path, clock, monkeypatch, caplog, advance):
    engine = _engine(tmp_path, condensation_fanin=2, summary_prefix_target_tokens=100)
    calls = []
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **k: calls.append(1) or ("merged", 1))
    for index in range(2):
        engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary=f"group {index}", token_count=1000,
                                         source_token_count=2000, source_ids=[], source_type="messages",
                                         created_at=index))
    _advance_on_call(monkeypatch, engine, "_select_threshold_sweep_condensation_group", clock, advance)
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            passes, reason = engine._run_threshold_sweep_condensation(
                target_tokens=100, pass_budget=5, deadline=clock() + 5.0)
        assert (passes, reason) == (0, "time_budget_exhausted")
        assert calls == [] and _count(caplog, CONDENSATION_LINE) == 0
    finally:
        engine.shutdown()
