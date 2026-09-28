"""#581 store-complete leaves and #582 survival fit (v0.24.5 SPEC tests (a)-(h)).

Engine-level: a host list in; stored rows, summary coverage, the lifecycle frontier and the returned
list out. Summaries are stubbed; they record every summarizer input."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
from hermes_lcm.tokens import count_messages_tokens

PAD = " alpha beta gamma delta" * 30


@pytest.fixture
def summaries(monkeypatch):
    captured: list[str] = []

    def summarize(**kwargs):
        captured.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return captured


def _engine(tmp_path, session="S", **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start(session, platform="telegram", context_length=200_000, conversation_id="conv")
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


# -- (h) flag off: unchanged --------------------------------------------------------------------------

def test_h_identity_anchor_off_leaves_the_eva_shape_unchanged(tmp_path, summaries, monkeypatch):
    """With LCM_IDENTITY_ANCHOR=false no row is read from the store: the eva shape fails open as before."""
    view = _eva_store(tmp_path)
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "false")
    monkeypatch.setenv("LCM_SURVIVAL_FIT", "false")
    engine = _engine(tmp_path)
    try:
        result = engine.compress(view)
        assert engine._last_compression_status == "error"
        assert engine._last_compression_noop_reason == "summary publication could not prove contiguous source coverage"
        assert len(result) == len(view) and _covered(engine) == []
    finally:
        engine.shutdown()
