"""#581 store-complete leaves and #582 survival fit (v0.24.5 SPEC tests (a)-(h)).

Engine-level: a host list in; stored rows, summary coverage, the lifecycle frontier and the returned
list out. Summaries are stubbed; they record every summarizer input."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import types
from collections import Counter

import pytest

import hermes_lcm.compaction as lcm_compaction
import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError

PAD = " alpha beta gamma delta" * 30


@pytest.fixture
def summaries(monkeypatch):
    captured: list[str] = []

    def summarize(**kwargs):
        captured.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return captured


def _engine(tmp_path, session="S", context_length=200_000, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start(session, platform="telegram", context_length=context_length, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float, *, tool: bool = False, stamp_user: bool = True) -> list[dict]:
    """A user row (host-stamped unless ``stamp_user`` is False); unstamped assistant/tool rows as eva's view."""
    user = {"role": "user", "content": f"[{tag}] user turn{PAD}"}
    if stamp_user:
        user["timestamp"] = ts
    rows = [user]
    if tool:
        call = {"id": f"call_{tag}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        rows += [{"role": "assistant", "content": "", "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": f"call_{tag}", "content": f"result of {tag}{PAD}"}]
    return rows + [{"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _rows(engine) -> list[dict]:
    return engine._store.get_session_messages(engine._session_id, limit=100_000)


def _frontier(engine) -> int:
    return int(engine._lifecycle.get_by_conversation("conv").current_frontier_store_id or 0)


def _covered(engine) -> list[int]:
    return [store_id for node in engine._dag.get_session_nodes(engine._session_id)
            if node.source_type == "messages" for store_id in node.source_ids]


def _assert_contiguous(engine) -> None:
    """Every owned row at or below the frontier is covered by a summary exactly once, or excluded by an
    existing rule (none are here): the publication proof, re-checked from the outside."""
    covered = Counter(_covered(engine))
    frontier = _frontier(engine)
    below = [int(row["store_id"]) for row in _rows(engine) if int(row["store_id"]) <= frontier]
    assert below and all(covered[store_id] == 1 for store_id in below), (frontier, below, covered)


def _eva_store(tmp_path):
    """The eva shape: owned rows the host view does not show, below its first mapped row.
    - natively compacted history: stamped users, unstamped assistant and tool rows, one unstamped user;
    - older duplicate copies (from a past ingest bug) of assistant/tool rows the view shows.
    The view is the host's native summary followed by the live rows."""
    engine = _engine(tmp_path)
    old = [*_turn("H1", 100.0, tool=True), *_turn("H2", 110.0), *_turn("H3", 120.0, tool=True),
           *_turn("H4", 0.0, stamp_user=False)]
    live = [row for i in range(10, 16) for row in _turn(f"T{i}", 100.0 * i, tool=i % 2 == 0)]
    native = {"role": "user", "content": "[CONTEXT COMPACTION] earlier turns were summarized by the host" + PAD,
              "timestamp": 999.0}
    try:
        engine.ingest(old)
        for message in live[1:4]:  # older copies of rows the view shows (unstamped assistant/tool rows)
            engine._store.append("S", dict(message), conversation_id="conv")
        engine.ingest([*old, native, *live])
    finally:
        engine.shutdown()
    return [native, *live]


# -- (a) the eva shape --------------------------------------------------------------------------------

def test_a_eva_shape_publishes_and_the_frontier_advances_over_passes(tmp_path, summaries):
    view = _eva_store(tmp_path)
    # restart: a fresh bind on the stored session, as the gateway does; bounded leaves of ~3 rows
    engine = _engine(tmp_path, dynamic_leaf_chunk_enabled=True, dynamic_leaf_chunk_max=400)
    try:
        before = [(int(r["store_id"]), r["role"], r["content"]) for r in _rows(engine)]
        frontiers, statuses, live = [_frontier(engine)], [], view
        for _ in range(4):
            live = engine.compress(live)
            statuses.append(engine._last_compression_status)
            frontiers.append(_frontier(engine))
            engine.on_session_start("S", boundary_reason="compression", old_session_id="S",
                                    platform="telegram", conversation_id="conv")
        after = [(int(r["store_id"]), r["role"], r["content"]) for r in _rows(engine)]
        assert statuses[:3] == ["compacted"] * 3, (statuses, engine._last_compression_noop_reason)
        assert frontiers[1] > frontiers[0] and frontiers[2] > frontiers[1] and frontiers[3] > frontiers[2], frontiers
        assert after == before  # 0 row changes: nothing lost, nothing re-stored
        _assert_contiguous(engine)
        hidden_tool = next(r for r in _rows(engine) if r["role"] == "tool" and "result of H1" in r["content"])
        assert int(hidden_tool["store_id"]) in _covered(engine)
        assert any("result of H1" in text for text in summaries)  # read from the store into the leaf
    finally:
        engine.shutdown()


# -- (b) a hidden-only leaf ---------------------------------------------------------------------------

def test_b_hidden_only_leaf_consumes_no_host_row(tmp_path, summaries):
    """No host raw chunk (the view is all fresh tail), owned hidden backlog below it: scheduled, not a no-op."""
    engine = _engine(tmp_path)
    old = [*_turn("H1", 100.0, tool=True), *_turn("H2", 0.0, stamp_user=False)]
    tail = _turn("T9", 900.0)
    try:
        engine.ingest([*old, *tail])
        hidden = [int(r["store_id"]) for r in _rows(engine)][: len(old)]
        result = engine.compress(list(tail))
        assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
        assert [m for m in result if m.get("content") in {t["content"] for t in tail}] == tail  # 0 host rows consumed
        assert set(hidden) & set(_covered(engine)) and _frontier(engine) < min(
            int(r["store_id"]) for r in _rows(engine) if r["content"] == tail[0]["content"])
        _assert_contiguous(engine)
    finally:
        engine.shutdown()


# -- (c) the endpoint stops before an unresolved retained occurrence -------------------------------------

def test_c_leaf_ends_before_a_retained_occurrence(tmp_path, summaries):
    """R (an unstamped user row) is stored early; the host re-orders it into the fresh tail. The leaf
    never covers R while the view retains it, and publication never conflicts."""
    engine = _engine(tmp_path, fresh_tail_count=1, leaf_chunk_tokens=1)
    r = {"role": "user", "content": "[R] retained occurrence" + PAD}
    stored = [*_turn("T1", 100.0), r, *[row for i in range(2, 6) for row in _turn(f"T{i}", 100.0 * i)]]
    try:
        engine.ingest(stored)
        r_id = next(int(row["store_id"]) for row in _rows(engine) if row["content"] == r["content"])
        view = [m for m in stored if m is not r] + [dict(r)]
        engine._ingest_cursor = len(view)  # the host holds the same rows, R moved to the tail
        statuses = []
        for _ in range(3):
            view = engine.compress(view)
            statuses.append(engine._last_compression_status)
        assert "error" not in statuses, engine._last_compression_noop_reason
        assert r_id not in _covered(engine) and _frontier(engine) < r_id
        assert _frontier(engine) > 0
        _assert_contiguous(engine)
    finally:
        engine.shutdown()


def test_b_native_on_off_host_summary_chunk_leaves_cover_the_stored_rows(tmp_path, summaries):
    """The native-on-off shape: a native-ON ref left a host summary (a row the store does not map) as the
    only raw chunk row, with the rows it summarised stored above frontier 0. The leaf covers stored rows
    (the host summary never takes the whole budget) instead of publishing no coverage."""
    engine = _engine(tmp_path, fresh_tail_count=6, leaf_chunk_tokens=300)
    old = [row for i in range(1, 9) for row in _turn(f"T{i}", 10.0 * i)]
    host_summary = {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
                                               "into the summary below." + PAD * 3, "timestamp": 95.0}
    tail = [row for i in range(9, 12) for row in _turn(f"T{i}", 10.0 * i)]
    try:
        engine.ingest([*old, *tail])
        view = [host_summary, *tail]
        engine._ingest_cursor = len(view)
        statuses = []
        for _ in range(3):
            view = engine.compress(view)
            statuses.append(engine._last_compression_status)
        assert statuses[0] == "compacted" and "error" not in statuses, (statuses, engine._last_compression_noop_reason)
        assert _frontier(engine) > 0
        _assert_contiguous(engine)
    finally:
        engine.shutdown()


def test_c_bound_leaf_passes_rows_of_another_conversation_under_the_session(tmp_path, summaries):
    """Eva thread 1: the bound session also holds rows of ANOTHER conversation, interleaved in store
    order. The bound conversation's obligation is its own rows (and blank ones) only: its leaves pass
    the other conversation's rows without covering them, and publication never conflicts."""
    engine = _engine(tmp_path, fresh_tail_count=2, leaf_chunk_tokens=300)
    first = [row for i in range(1, 4) for row in _turn(f"T{i}", 10.0 * i)]
    second = [row for i in range(4, 7) for row in _turn(f"T{i}", 10.0 * i)]
    tail = [row for i in range(8, 10) for row in _turn(f"T{i}", 10.0 * i)]
    try:
        engine.ingest(first)
        foreign = [engine._store.append("S", {"role": "assistant", "content": f"[X{i}] other conversation" + PAD},
                                        conversation_id="other") for i in range(3)]
        view = [*first, *second, *tail]
        engine.ingest(view)
        statuses = []
        for _ in range(6):
            view = engine.compress(view)
            statuses.append(engine._last_compression_status)
        assert "error" not in statuses, (statuses, engine._last_compression_noop_reason)
        assert _frontier(engine) > max(foreign), (_frontier(engine), foreign, statuses)
        assert set(foreign).isdisjoint(_covered(engine))
        assert all("other conversation" not in text for text in summaries)
        _assert_contiguous_for_conversation(engine)
    finally:
        engine.shutdown()


def _assert_contiguous_for_conversation(engine) -> None:
    """#5 per conversation: every row of the bound (or blank) conversation at or below the frontier
    is covered exactly once; rows of other conversations are neither required nor covered."""
    covered = Counter(_covered(engine))
    frontier = _frontier(engine)
    own = [int(r["store_id"]) for r in _rows(engine)
           if int(r["store_id"]) <= frontier and str(r.get("conversation_id") or "").strip() in ("", "conv")]
    assert own and all(covered[store_id] == 1 for store_id in own), (frontier, own, covered)


def _stage(engine, expected, covered, node_source=None):
    node = SummaryNode(session_id="S", summary="s", token_count=1, source_token_count=1,
                       source_ids=list(node_source or covered))

    def stage(conn, node_id) -> None:
        engine._lifecycle.stage_compaction_publication(conn, "conv", "S", node_id, expected, list(covered))

    engine._dag.add_node(node, before_commit=stage)


def _mixed_rows(engine):
    """Own A, other X, own B, other Y, own C: two conversations interleaved under session S."""
    store, ids = engine._store, {}
    for tag, conversation in (("A", "conv"), ("X", "other"), ("B", "conv"), ("Y", "other"), ("C", "conv")):
        ids[tag] = store.append("S", {"role": "assistant", "content": f"[{tag}] row" + PAD}, conversation_id=conversation)
    return ids


def test_5_bound_frontier_passes_other_conversation_rows_but_not_an_unproven_bound_row(tmp_path):
    """#5 per conversation: A..C (own rows A, B, C) publishes past the other conversation's X and Y;
    skipping the unproven own row B, which sits between X and Y, is refused."""
    engine = _engine(tmp_path)
    try:
        ids = _mixed_rows(engine)
        with pytest.raises(LifecyclePublicationConflictError, match="not contiguous"):
            _stage(engine, 0, [ids["A"], ids["C"]])
        assert _frontier(engine) == 0
        _stage(engine, 0, [ids["A"], ids["B"], ids["C"]])
        assert _frontier(engine) == ids["C"]
    finally:
        engine.shutdown()


def test_proof_rejects_covering_another_conversations_row(tmp_path):
    """A leaf that claims another conversation's row as coverage is still refused."""
    engine = _engine(tmp_path)
    try:
        ids = _mixed_rows(engine)
        with pytest.raises(LifecyclePublicationConflictError, match="ownership"):
            _stage(engine, 0, [ids["A"], ids["X"], ids["B"]])
        assert _frontier(engine) == 0 and _covered(engine) == []
    finally:
        engine.shutdown()


# -- (d) the carry set ----------------------------------------------------------------------------------

def _state_db(tmp_path, sessions):
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
    conn.executemany("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)", sessions)
    conn.commit()
    conn.close()


def test_d_owned_rows_are_the_session_plus_the_publication_carry_set(tmp_path, summaries):
    """A compression child C carries parent P's rows (P, 1, 4]; P's hidden carried row 3 is read into the
    leaf. P's row 6, outside the carry, and a sibling session's rows are never read or covered."""
    _state_db(tmp_path, [("P", None, "compression"), ("C", "P", None), ("X", None, None)])
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=400, context_threshold=0.001,
                                        database_path=str(tmp_path / "lcm.db")), hermes_home=str(tmp_path))
    try:
        engine.on_session_start("C", platform="telegram", context_length=200_000, conversation_id="conv")
        store = engine._store
        parent = [*_turn("P1", 10.0), *_turn("P2", 20.0)]
        ids = [store.append("P", dict(m), conversation_id="conv") for m in parent]  # 1..4
        sibling = store.append("X", {"role": "user", "content": "[X] sibling row" + PAD}, conversation_id="conv")
        outside = store.append("P", {"role": "assistant", "content": "[P] outside the carry" + PAD}, conversation_id="conv")
        store.write_metadata_json([engine._identity_anchor_carry_key()], json.dumps([["P", 0, ids[-1]]]))
        live = [row for i in range(3, 7) for row in _turn(f"C{i}", 100.0 * i)]
        view = [parent[0], parent[1], parent[3], *live]  # P's row 3 is hidden (the host dropped it)
        engine.ingest(view)
        engine.compress(view)
        covered = set(_covered(engine))
        assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
        assert ids[2] in covered and {sibling, outside}.isdisjoint(covered)
        assert all("sibling row" not in text and "outside the carry" not in text for text in summaries)
    finally:
        engine.shutdown()


# -- (e)-(g) the survival fit ------------------------------------------------------------------------------

WINDOW = 6000
TARGET = int(WINDOW * 0.85)
NOTICE = "[LCM survival fit:"


def _host_rough(messages) -> int:
    """A stand-in for the host's rough request estimator (chars / 4 over content and tool calls)."""
    return sum(4 + (len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or ""))) // 4
               for m in messages)


@pytest.fixture
def host_estimator(monkeypatch):
    module = types.ModuleType("agent.model_metadata")
    module.estimate_messages_tokens_rough = _host_rough
    monkeypatch.setitem(sys.modules, "agent.model_metadata", module)
    return _host_rough


def _long_view(turns=24) -> list[dict]:
    """A system prompt and ``turns`` stored turns (every third with a tool call): ~2x the window."""
    rows = [row for i in range(turns) for row in _turn(f"L{i}", 10.0 * i, tool=i % 3 == 0)]
    return [{"role": "system", "content": "system prompt"}, *rows]


def _assert_fitted(engine, view, result, host) -> None:
    """(e): the fitted list is under target by the host estimator, starts at a user turn after the
    system slot, carries the notice there (never as a row) and raised exactly one warning."""
    assert host(view) > TARGET and host(result) <= TARGET, (host(view), host(result))
    assert result[0]["role"] == "system" and NOTICE in result[0]["content"]
    assert result[1]["role"] == "user" and not any(NOTICE in str(m.get("content")) for m in result[1:])
    assert [m for m in result[1:]] == view[-len(result) + 1:]  # the newest whole turns, unchanged
    first = engine.get_automatic_compaction_status_message(phase="compress", default_message="Compacting")
    second = engine.get_automatic_compaction_status_message(phase="compress", default_message="Compacting")
    assert first and "LCM" in first and second is None and engine.emit_automatic_compaction_status is False


def test_e_survival_fit_after_an_injected_publication_conflict(tmp_path, summaries, host_estimator, caplog):
    engine = _engine(tmp_path, context_length=WINDOW)
    view = _long_view()
    try:
        engine.ingest(view)

        def conflict(*args, **kwargs):
            raise LifecyclePublicationConflictError("injected conflict (ids only)")

        engine._lifecycle.stage_compaction_publication = conflict
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=host_estimator(view))
        assert engine._last_compression_status == "error"
        _assert_fitted(engine, view, result, host_estimator)
        assert sum("LCM survival fit applied" in r.getMessage() for r in caplog.records) == 1
        assert "publication_invariant_conflict" in caplog.text
        counter = engine._store.read_metadata_json("survival_fit:counter")
        assert counter["count"] == 1 and counter["last_reason"] == "publication_invariant_conflict"
    finally:
        engine.shutdown()


def test_e_survival_fit_after_a_sweep_deadline(tmp_path, summaries, host_estimator, monkeypatch):
    monkeypatch.setattr(lcm_compaction, "_THRESHOLD_FULL_SWEEP_MAX_SECONDS", 0.0)
    engine = _engine(tmp_path, context_length=WINDOW, context_threshold=0.5, threshold_full_sweep_enabled=True)
    view = _long_view()
    try:
        engine.ingest(view)
        result = engine.compress(view, current_tokens=host_estimator(view))
        _assert_fitted(engine, view, result, host_estimator)
    finally:
        engine.shutdown()


def test_e_survival_fit_after_a_lock_after_commit(tmp_path, summaries, host_estimator, monkeypatch):
    """One leaf publishes, condensation then hits a SQLite lock: the committed list is fitted."""
    engine = _engine(tmp_path, context_length=WINDOW, fresh_tail_count=60)  # a long retained tail
    view = _long_view()
    try:
        engine.ingest(view)

        def locked(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(engine, "_maybe_condense", locked)
        result = engine.compress(view, current_tokens=100)  # below the list: one bounded leaf, no overflow
        assert _frontier(engine) > 0 and engine._last_compression_noop_reason == \
            "summary publication blocked by SQLite lock"
        assert host_estimator(result) <= TARGET and NOTICE in result[0]["content"]
        assert result[1]["role"] == "user" and not any(NOTICE in str(m.get("content")) for m in result[1:])
        assert engine.get_automatic_compaction_status_message(phase="compress", default_message="x")
    finally:
        engine.shutdown()


def test_f_newest_turn_over_budget_is_projected(tmp_path, summaries, host_estimator):
    """The newest user turn alone is over the window: a bounded projection, never empty, raw rows intact."""
    engine = _engine(tmp_path, context_length=WINDOW)
    call = {"id": "call_big", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    big = {"role": "tool", "tool_call_id": "call_big", "content": "row " * 12_000}
    view = [*_long_view(4), {"role": "user", "content": "[N] newest", "timestamp": 99.0},
            {"role": "assistant", "content": "", "tool_calls": [call]}, big]
    try:
        engine.ingest(view)
        engine._lifecycle.stage_compaction_publication = lambda *a, **k: (_ for _ in ()).throw(
            LifecyclePublicationConflictError("injected"))
        result = engine.compress(view, current_tokens=host_estimator(view))
        assert result and host_estimator(result) <= TARGET, host_estimator(result)
        assert result[1]["content"] == "[N] newest" and result[-1]["role"] == "tool"
        assert result[-1]["content"] != big["content"] and result[-1]["tool_call_id"] == "call_big"
        stored = [r for r in _rows(engine) if r["role"] == "tool" and r["content"] == big["content"]]
        assert len(stored) == 1  # the raw row stays stored verbatim
    finally:
        engine.shutdown()


def test_g_fitted_list_re_ingests_without_new_rows(tmp_path, summaries, host_estimator):
    """The host adopts the fitted list, archives the session, and a cold process resumes it: only a
    genuinely new turn adds rows."""
    engine = _engine(tmp_path, context_length=WINDOW)
    view = _long_view()
    try:
        engine.ingest(view)
        engine._lifecycle.stage_compaction_publication = lambda *a, **k: (_ for _ in ()).throw(
            LifecyclePublicationConflictError("injected"))
        fitted = engine.compress(view, current_tokens=host_estimator(view))
        assert len(fitted) < len(view)
        before = len(_rows(engine))
        engine.on_session_end("S", fitted)
        assert len(_rows(engine)) == before
    finally:
        engine.shutdown()
    cold = _engine(tmp_path, context_length=WINDOW)
    try:
        new = _turn("NEW", 500.0)
        cold.ingest([*fitted, *new])
        added = _rows(cold)[before:]
        assert len(added) == len(new), [(r["role"], r["store_id"]) for r in added]
    finally:
        cold.shutdown()


def test_h_survival_fit_off_leaves_the_result_unchanged(tmp_path, summaries, host_estimator):
    engine = _engine(tmp_path, context_length=WINDOW, survival_fit=False)
    view = _long_view()
    try:
        engine.ingest(view)
        engine._lifecycle.stage_compaction_publication = lambda *a, **k: (_ for _ in ()).throw(
            LifecyclePublicationConflictError("injected"))
        result = engine.compress(view, current_tokens=host_estimator(view))
        assert result == view and engine.emit_automatic_compaction_status is False
        assert engine._store.read_metadata_json("survival_fit:counter") is None
    finally:
        engine.shutdown()


# -- (h) flag off: unchanged --------------------------------------------------------------------------

def test_h_identity_anchor_off_leaves_the_eva_shape_unchanged(tmp_path, summaries, monkeypatch):
    """With LCM_IDENTITY_ANCHOR=false no row is read from the store: the eva shape fails open as before."""
    view = _eva_store(tmp_path)
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "false")
    engine = _engine(tmp_path, survival_fit=False)
    try:
        result = engine.compress(view)
        assert engine._last_compression_status == "error"
        assert engine._last_compression_noop_reason == "summary publication could not prove contiguous source coverage"
        assert len(result) == len(view) and _covered(engine) == []
    finally:
        engine.shutdown()
