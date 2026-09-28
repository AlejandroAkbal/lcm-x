"""#608: a spent threshold-sweep time budget is a stop condition, not an error.

Engine-level: a host list in, the returned list, the public status, the sweep telemetry and the log
lines out. Summaries are stubbed and counted; the sweep clock is a real monotonic clock plus an offset
that a wrapped engine step advances."""

from __future__ import annotations

import inspect
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


def _engine(tmp_path, context_length: int = 200_000, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "threshold_full_sweep_enabled": True, "max_assembly_tokens": 100_000,
                "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=context_length, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float) -> list[dict]:
    return [{"role": "user", "content": f"[{tag}] user turn{PAD}", "timestamp": ts},
            {"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _view(turns: int = 6) -> list[dict]:
    return [{"role": "system", "content": "system prompt"},
            *[row for i in range(turns) for row in _turn(f"T{i}", 10.0 * (i + 1))]]


def _advance_on_call(monkeypatch, engine, name: str, clock: _Clock, seconds: float, *, call: int | None = 1):
    """Wrap engine step ``name``: its ``call``-th invocation (every one when None) moves the sweep clock."""
    original = getattr(engine, name)
    count = 0

    def wrapped(*args, **kwargs):
        nonlocal count
        result = original(*args, **kwargs)
        count += 1
        if call is None or count == call:
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


@pytest.mark.parametrize(("method", "call", "step"), [
    ("_get_store_id_map_for_messages", 1, "anchor_ids"),
    ("_committed_replay_drops", 1, "replay_drops"),
    ("_get_store_id_map_for_messages", 2, "store_id_map"),
    ("_identity_anchor_summary_input", 1, "identity_anchor"),
])
def test_pass_leaves_right_after_the_step_that_spends_the_budget(
        tmp_path, summaries, clock, monkeypatch, caplog, method, call, step):
    engine = _engine(tmp_path)
    view = _view()
    _advance_on_call(monkeypatch, engine, method, clock, 121.0, call=call)
    try:
        engine.ingest(view)
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert result is view and summaries == [] and engine._last_compression_status == "noop"
        line = next(r.getMessage() for r in caplog.records if BUDGET_LINE in r.getMessage())
        assert line.endswith(f"{step}=121.0s")  # the last timed step: nothing ran after it
    finally:
        engine.shutdown()


def test_budget_spent_in_a_provider_timeout_names_the_summariser_step(tmp_path, clock, monkeypatch, caplog):
    """The first summariser call times out after 110 s: the smaller-chunk retry finds 10 s left and the
    sweep stops before the first leaf; the WARNING shows where the time went."""
    engine = _engine(tmp_path, leaf_chunk_tokens=2000)
    view = _view()
    calls = []

    def summarize(**kwargs):
        calls.append(1)
        clock.offset += 110.0
        raise TimeoutError("provider timed out")

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    try:
        engine.ingest(view)
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert calls == [1] and result is view and telemetry["stop_reason"] == "time_budget_exhausted"
        assert _count(caplog, RETRY_LINE) == 1 and _count(caplog, BUDGET_LINE) == 1
        line = next(r.getMessage() for r in caplog.records if BUDGET_LINE in r.getMessage())
        assert "summariser=110.0s" in line
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


# -- 8. the hold never applies at or over the survival ceiling ------------------------------------------------

def _hold(engine) -> None:
    engine._sweep_budget_hold_until = time.time() + 600.0


def test_hold_ends_at_the_survival_ceiling_in_should_compress(tmp_path):
    """A1: window 100,000 and reserve 0.15 make the ceiling 85,000."""
    engine = _engine(tmp_path, context_length=100_000, max_assembly_tokens=0)
    try:
        _hold(engine)
        assert engine.should_compress(84_999) is False
        assert engine.should_compress(85_000) is True
    finally:
        engine.shutdown()


def test_hold_ends_at_the_ceiling_by_the_last_prompt_size_in_preflight_without_a_replay_diff(tmp_path):
    """A2: the listed messages are far below the ceiling; the host's last prompt decides."""
    engine = _engine(tmp_path, context_length=100_000, max_assembly_tokens=0)
    view = _view()
    try:
        engine.ingest(view)
        _hold(engine)
        engine.last_prompt_tokens = 1_000
        assert engine.should_compress_preflight(view) is False
        engine.last_prompt_tokens = 85_000
        assert engine.should_compress_preflight(view) is True
    finally:
        engine.shutdown()


def test_hold_ends_at_the_ceiling_by_the_last_prompt_size_in_preflight_with_a_replay_diff(tmp_path, monkeypatch):
    """A3: the replay differs from the host list (not a cleanup the host must adopt)."""
    engine = _engine(tmp_path, context_length=100_000, max_assembly_tokens=0)
    view = _view()
    replay_diffs = []
    original = engine._ingest_messages

    def ingest_with_a_diff(messages):
        replay = [dict(message) for message in original(messages)]
        replay[1]["content"] += " (replayed)"
        replay_diffs.append(1)
        return replay

    monkeypatch.setattr(engine, "_ingest_messages", ingest_with_a_diff)
    try:
        _hold(engine)
        engine.last_prompt_tokens = 1_000
        assert engine.should_compress_preflight(view) is False
        engine.last_prompt_tokens = 85_000
        assert engine.should_compress_preflight(view) is True
        assert replay_diffs == [1, 1]
    finally:
        engine.shutdown()


def test_hold_applies_at_any_size_when_no_window_is_known(tmp_path):
    """A4: context_length 0 has no ceiling: today's hold."""
    engine = _engine(tmp_path, context_length=0, max_assembly_tokens=0)
    try:
        engine.context_length = 0
        engine.threshold_tokens = 200
        _hold(engine)
        assert engine.should_compress(10_000_000) is False
    finally:
        engine.shutdown()


def test_a_list_over_the_survival_budget_is_fitted_during_a_hold(tmp_path, summaries, clock, monkeypatch, caplog):
    """A5: every sweep spends its budget in its first map. After the first no-leaf stop, a list over the
    survival budget is still asked for and fitted: status noop, no raise, the result at or under budget."""
    engine = _engine(tmp_path, context_length=6_000)
    _advance_on_call(monkeypatch, engine, "_get_store_id_map_for_messages", clock, 121.0, call=None)
    small, large = _view(4), _view(40)
    budget = int(6_000 * (1 - 0.15))
    try:
        engine.ingest(small)
        engine.compress(small, current_tokens=engine.threshold_tokens + 1)
        assert engine._sweep_budget_hold_active() and engine._last_compression_noop_reason.startswith(
            "threshold sweep time budget spent")
        engine.ingest(large)
        tokens = engine._survival_measure(large)
        assert tokens > budget
        assert engine.should_compress(tokens) is True
        result = engine.compress(large, current_tokens=tokens)
        assert engine._last_compression_status == "noop"
        assert engine.get_status()["threshold_full_sweep"]["stop_reason"] == "time_budget_exhausted"
        assert result is not large and engine._survival_measure(result) <= budget
        assert summaries == []
    finally:
        engine.shutdown()


# -- 9. a request the provider rejected comes back shorter ----------------------------------------------------

def _rejected_state(tmp_path, monkeypatch, clock, *, spend=True):
    """Window 6,000 (ceiling 5,100); every sweep spends its budget in its first map when ``spend``."""
    engine = _engine(tmp_path, context_length=6_000)
    if spend:
        _advance_on_call(monkeypatch, engine, "_get_store_id_map_for_messages", clock, 121.0, call=None)
    view = _view(8)
    engine.ingest(view)
    return engine, view


def _shorter_by_host_score(engine, result, messages) -> bool:
    return len(result) < len(messages) or engine._survival_measure(result) < 0.95 * engine._survival_measure(messages)


def _fit_spy(monkeypatch, engine) -> list:
    caps = []
    original = engine._survival_fit

    def spy(*args, window_cap=None, **kwargs):
        caps.append(window_cap)
        return original(*args, window_cap=window_cap, **kwargs)

    monkeypatch.setattr(engine, "_survival_fit", spy)
    return caps


def test_rejected_request_after_a_no_leaf_stop_comes_back_shorter(tmp_path, summaries, clock, monkeypatch):
    """C1: below the ceiling, recovery attempt, the request estimate as current_tokens: a shorter list whose
    dropped rows are all stored, the fit named provider_overflow, no error."""
    engine, view = _rejected_state(tmp_path, monkeypatch, clock)
    request = engine._survival_measure(view) + 1_000  # the host adds its system prompt and tools
    assert request < int(6_000 * 0.85)
    try:
        result = engine.compress(view, current_tokens=request, bypass_cooldown=True)
        assert _shorter_by_host_score(engine, result, view) and summaries == []
        stored = {(r["role"], r["content"]) for r in engine._store.get_session_messages("S", limit=100_000)}
        kept = {(m["role"], m["content"]) for m in result}
        assert all((m["role"], m["content"]) in stored for m in view[1:] if (m["role"], m["content"]) not in kept)
        assert engine._last_survival_fit["reason"].startswith("provider_overflow:")
        assert engine._last_compression_status == "noop"
    finally:
        engine.shutdown()


def test_same_state_without_a_recovery_attempt_returns_the_identical_list(tmp_path, summaries, clock, monkeypatch):
    """C2: bypass_cooldown False (turn start, threshold): the identical list, no fit."""
    engine, view = _rejected_state(tmp_path, monkeypatch, clock)
    try:
        result = engine.compress(view, current_tokens=engine._survival_measure(view) + 1_000, bypass_cooldown=False)
        assert result is view and engine._last_survival_fit is None
    finally:
        engine.shutdown()


def test_a_recovery_attempt_that_shortens_the_list_runs_no_capped_fit(tmp_path, summaries, clock, monkeypatch):
    """C3: a sweep stores leaves and the list is shorter: the fit is not called with a window cap."""
    engine, view = _rejected_state(tmp_path, monkeypatch, clock, spend=False)
    caps = _fit_spy(monkeypatch, engine)
    try:
        result = engine.compress(view, current_tokens=engine._survival_measure(view) + 1_000, bypass_cooldown=True)
        assert engine._last_compression_status == "compacted" and summaries
        assert _shorter_by_host_score(engine, result, view)
        assert caps and all(cap is None for cap in caps)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("current_tokens", [None, 0])
def test_without_a_request_size_the_cap_is_the_measure_of_the_list(tmp_path, summaries, clock, monkeypatch,
                                                                   current_tokens):
    """C4: no positive current_tokens: the cap is the measure of the list."""
    engine, view = _rejected_state(tmp_path, monkeypatch, clock)
    caps = _fit_spy(monkeypatch, engine)
    try:
        result = engine.compress(view, current_tokens=current_tokens, bypass_cooldown=True)
        assert caps == [None, engine._survival_measure(view)]
        assert _shorter_by_host_score(engine, result, view)
    finally:
        engine.shutdown()


def test_the_next_ingest_after_a_capped_fit_stores_only_the_new_turn(tmp_path, summaries, clock, monkeypatch):
    """C5: the fitted list plus one new turn: the new turn is stored once, no old row again."""
    engine, view = _rejected_state(tmp_path, monkeypatch, clock)

    def rows():
        return sorted((r["role"], r["content"]) for r in engine._store.get_session_messages("S", limit=100_000))
    try:
        result = engine.compress(view, current_tokens=engine._survival_measure(view) + 1_000, bypass_cooldown=True)
        assert len(result) < len(view)
        before = rows()
        new = _turn("N1", 999.0)
        engine.ingest(result + new)
        after = rows()
        added = list(after)
        for row in before:
            added.remove(row)
        assert sorted(added) == sorted((m["role"], m["content"]) for m in new)
    finally:
        engine.shutdown()


def test_the_except_path_of_a_recovery_attempt_returns_the_capped_fit(tmp_path, clock, monkeypatch):
    """C6: _compress_impl raises; recovery attempt; list below the ceiling: the fitted list, no raise."""
    engine, view = _rejected_state(tmp_path, monkeypatch, clock, spend=False)

    def boom(*args, **kwargs):
        raise RuntimeError("injected (ids only)")

    monkeypatch.setattr(engine, "_compress_impl", boom)
    try:
        result = engine.compress(view, current_tokens=engine._survival_measure(view) + 1_000, bypass_cooldown=True)
        assert _shorter_by_host_score(engine, result, view)
        assert engine._last_survival_fit["reason"] == "provider_overflow:exception:RuntimeError"
        assert engine._last_compression_status == "error"
    finally:
        engine.shutdown()


def test_compress_accepts_the_host_recovery_keyword(tmp_path):
    """C7 (in-process half): the host passes bypass_cooldown only when compress() names it."""
    engine = _engine(tmp_path)
    try:
        assert "bypass_cooldown" in inspect.signature(engine.compress).parameters
    finally:
        engine.shutdown()


@pytest.mark.parametrize(("observed", "expected"), [(None, 5100), (1432 + 500, 4600), (100_000, 2100)])
def test_survival_budget_without_a_cap_is_unchanged(tmp_path, observed, expected):
    """C8: window_cap=None gives the numbers pinned at c1312f3c (window 6,000, reserve 0.15, 1,432 tokens)."""
    engine = _engine(tmp_path, context_length=6_000)
    view = [{"role": "system", "content": "system prompt"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"[{i}]{PAD}"} for i in range(8)]
    try:
        assert lcm_engine.count_messages_tokens(view) == 1432
        assert engine._survival_fit_budget(view, observed, window_cap=None) == expected
        assert engine._survival_fit_budget(view, observed) == expected
    finally:
        engine.shutdown()
