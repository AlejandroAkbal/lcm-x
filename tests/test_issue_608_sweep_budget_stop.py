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


# -- 1. the budget is spent before the first leaf ------------------------------------------------------

def test_budget_spent_in_the_first_map_returns_the_input_unchanged(tmp_path, summaries, clock, monkeypatch, caplog):
    engine = _engine(tmp_path, leaf_chunk_tokens=2000)  # a multi-row chunk: the base logs two retry lines
    view = _view()
    try:
        result = _spend_budget_before_first_leaf(engine, view, clock, monkeypatch, caplog)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert result is view
        assert engine._last_compression_status == "noop"
        assert engine._last_compression_noop_reason == "threshold sweep time budget spent before the first leaf"
        assert telemetry["status"] == "noop" and telemetry["leaf_passes"] == 0
        assert telemetry["stop_reason"] == "time_budget_exhausted" and telemetry["budget_exhausted"] is True
        assert summaries == []
        assert _count(caplog, RETRY_LINE) == 0
        assert _count(caplog, BUDGET_LINE) == 1
        line = next(r.getMessage() for r in caplog.records if BUDGET_LINE in r.getMessage())
        assert "(budget 120s); steps: " in line and "anchor_ids=121.0s" in line  # numbers and step names only
        assert "user turn" not in line and "alpha" not in line
    finally:
        engine.shutdown()


def test_budget_spent_in_the_store_complete_step_stops_before_the_identity_anchor(
        tmp_path, summaries, clock, monkeypatch, caplog):
    """The view is all fresh tail and the owned backlog is hidden: the store-complete step is the one
    that spends the budget, and the pass leaves before the identity-anchor step."""
    engine = _engine(tmp_path)
    old = [*_turn("H1", 100.0), *_turn("H2", 110.0)]
    tail = _turn("T9", 900.0)
    anchor_calls = []
    original_anchor = engine._identity_anchor_summary_input
    monkeypatch.setattr(engine, "_identity_anchor_summary_input",
                        lambda *a, **k: anchor_calls.append(1) or original_anchor(*a, **k))
    _advance_on_call(monkeypatch, engine, "_store_complete_backlog", clock, 121.0)
    try:
        engine.ingest([*old, *tail])
        view = list(tail)
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert result is view and engine._last_compression_status == "noop"
        assert telemetry["stop_reason"] == "time_budget_exhausted"
        assert anchor_calls == [] and summaries == []
        assert _count(caplog, BUDGET_LINE) == 1
    finally:
        engine.shutdown()


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


# -- 6. the hold after a no-leaf stop -----------------------------------------------------------------------

def test_no_leaf_budget_stop_holds_the_threshold_answer_only(tmp_path, summaries, clock, monkeypatch, caplog):
    engine = _engine(tmp_path)
    view = _view()
    try:
        _spend_budget_before_first_leaf(engine, view, clock, monkeypatch, caplog)
        assert engine._sweep_budget_hold_until > time.time()
        assert engine.should_compress(engine.threshold_tokens + 1) is False
        assert engine.should_compress_preflight(view) is False
        assert engine.should_compress(150_000) is True  # over the 100k assembly cap: overflow recovery
        real_time = time.time
        with monkeypatch.context() as later:
            later.setattr(lcm_engine.time, "time", lambda: real_time() + 601.0)
            assert engine.should_compress(engine.threshold_tokens + 1) is True
        engine._sweep_budget_hold_until = time.time() + 600.0
        assert engine.should_compress(engine.threshold_tokens + 1) is False
        clock.offset = 0.0  # a normal budget: compress() is not gated, and a stored leaf clears the hold
        monkeypatch.setattr(engine, "_get_store_id_map_for_messages",
                            LCMEngine._get_store_id_map_for_messages.__get__(engine))
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert engine._last_compression_status == "compacted" and summaries
        assert engine._sweep_budget_hold_until == 0.0
        assert engine.should_compress(engine.threshold_tokens + 1) is True
    finally:
        engine.shutdown()


# -- 7. the empty anchor slice is not mapped -----------------------------------------------------------------

def test_empty_anchor_slice_is_not_mapped_and_the_pass_result_is_the_same(tmp_path, summaries, monkeypatch):
    """No system prompt: the anchor slice is empty. Skipping its map uses the value the map returns for
    an empty slice, so the pass publishes the same leaf and returns the same list as without the skip."""
    engine = _engine(tmp_path)
    view = _view()[1:]
    mapped = []
    original = engine._get_store_ids_for_messages
    monkeypatch.setattr(engine, "_get_store_ids_for_messages",
                        lambda messages, *a, **k: mapped.append(len(messages)) or original(messages, *a, **k))
    try:
        assert original([]) == []
        engine.ingest(view)
        result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        nodes = engine._dag.get_session_nodes("S")
        assert engine._last_compression_status == "compacted"
        assert [node.source_ids for node in nodes] == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        assert result[-2:] == view[-2:] and len(result) == 3
        assert 0 not in mapped
    finally:
        engine.shutdown()
