"""#436: host-timestamp-anchored, occurrence-bound message identity (REVISION 1).

(a)-(e) are the REVISION 1 acceptance counterexamples; the rest pin one rule each. Engine-level only:
a host list in, stored rows / relations / summary coverage out."""

from __future__ import annotations

import sqlite3

import pytest

import hermes_lcm.engine as lcm_engine

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SYSTEM = {"role": "system", "content": "stable system prompt"}
PAD = " alpha beta gamma delta" * 40


def _u(text: str, ts: float | None) -> dict:
    return {"role": "user", "content": text} if ts is None else {"role": "user", "content": text, "timestamp": ts}


def _a(text: str, ts: float | None) -> dict:
    return {"role": "assistant", "content": text} if ts is None else {"role": "assistant", "content": text, "timestamp": ts}


def _turns(start: int, count: int, base_ts: float) -> list[dict]:
    rows = []
    for index in range(start, start + count):
        rows += [_u(f"[T{index}] user turn {index}:{PAD}", base_ts + index * 10),
                 _a(f"reply to T{index}", base_ts + index * 10 + 1)]
    return rows


@pytest.fixture
def summaries(monkeypatch):
    """Every summarizer input, in call order."""
    captured: list[str] = []

    def summarize(**kwargs):
        captured.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return captured


def _engine(tmp_path, session: str = "S", conversation: str = "conv") -> LCMEngine:
    config = LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1, database_path=str(tmp_path / "lcm.db"))
    engine = LCMEngine(config=config)
    engine.on_session_start(session, platform="cli", context_length=200_000, conversation_id=conversation)
    return engine


def _rows(engine: LCMEngine, session: str | None = None) -> list[dict]:
    return [row for row in engine._store.get_session_messages(session or engine._session_id) if row.get("role") != "system"]


def _relations(engine: LCMEngine) -> list[tuple]:
    conn = engine._store._conn
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='message_relations'").fetchone():
        return []
    return conn.execute("SELECT store_id, kind, related_store_id, ordinal FROM message_relations ORDER BY relation_id").fetchall()


def _state_db(tmp_path, sessions: list[tuple[str, str | None, str | None]]) -> None:
    """The host's state.db next to lcm.db: (id, parent_session_id, end_reason)."""
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
    conn.executemany("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)", sessions)
    conn.commit()
    conn.close()


def _assert_claims_are_in_the_input(engine: LCMEngine, captured: list[str]) -> list[int]:
    """Every leaf source a summary claims had its text in a summarizer input."""
    text = "\n".join(captured)
    claimed = [sid for node in engine._dag.get_session_nodes(engine._session_id)
               if node.source_type == "messages" for sid in node.source_ids]
    rows = engine._store.get_batch(claimed)
    for store_id in claimed:
        content = str(rows[store_id].get("content") or "")
        assert content.strip()[:60] in text, f"store_id {store_id} claimed without its text: {content[:60]!r}"
    return claimed


# -- (a)-(e) REVISION 1 acceptance -------------------------------------------------------------------

def test_a_unrelated_same_timestamp_row_is_not_absorbed(tmp_path, summaries):
    """Astra's counterexample: R and U share one host stamp and nothing else. U never absorbs R:
    no relation joins them, and R is claimed only with its own bytes in the summarizer input."""
    engine = _engine(tmp_path)
    r = _u("unrelated retained text" + PAD, 500.0)
    u = _u("new user text" + PAD, 500.0)
    head = _turns(1, 3, 0.0)  # Hermes hands the engine no system row
    try:
        engine.ingest([*head, r])
    finally:
        engine.shutdown()
    engine = _engine(tmp_path)  # restart; the host no longer shows R, U carries the same (batch) stamp
    try:
        live = [*head, u, _a("reply to U", 502.0), *_turns(10, 4, 600.0)]
        engine.compress(live)
        by_text = {str(row["content"]): int(row["store_id"]) for row in _rows(engine)}
        r_id, u_id = by_text[r["content"]], by_text[u["content"]]
        assert not [rel for rel in _relations(engine) if {r_id, u_id} <= {rel[0], rel[2]}]
        claimed = _assert_claims_are_in_the_input(engine, summaries)
        assert u_id in claimed
    finally:
        engine.shutdown()


@pytest.mark.parametrize("restart", [False, True], ids=["steady", "restart"])
def test_b_summary_never_claims_a_row_whose_text_is_not_in_its_input(tmp_path, summaries, restart):
    """H1 persist override: R's survivor is rewritten to U under R's stamp, R leaves the host list.
    U is stored once, and a summary may claim R only with R's own bytes in the summarizer input."""
    engine = _engine(tmp_path)
    r = _u("interrupted prompt R" + PAD, 500.0)
    u = _u("follow-up U" + PAD, 500.0)
    head = _turns(1, 3, 0.0)  # Hermes hands the engine no system row
    try:
        engine.ingest([*head, r])
        if restart:
            engine.shutdown()
            engine = _engine(tmp_path)
        live = [*head, u, _a("reply to U", 502.0), *_turns(10, 4, 600.0)]
        engine.ingest(live)
        engine.compress(live)
        texts = [str(row["content"]) for row in _rows(engine)]
        assert texts.count(r["content"]) == 1 and texts.count(u["content"]) == 1
        assert engine._last_compression_status == "compacted"  # R pending forever would stall publication
        u_id = next(int(row["store_id"]) for row in _rows(engine) if row["content"] == u["content"])
        assert u_id in _assert_claims_are_in_the_input(engine, summaries)
    finally:
        engine.shutdown()


def test_c_same_conversation_sibling_session_gets_no_carry(tmp_path, summaries):
    """A session that merely shares the conversation id is not a compression child: its replay of
    the other session's rows gets no carry, and its publication claims only its own rows."""
    _state_db(tmp_path, [("A", None, "user_exit"), ("B", None, None)])
    history = [SYSTEM, *_turns(1, 4, 0.0)]
    engine = _engine(tmp_path, "A")
    try:
        engine.ingest(history)
        a_ids = {int(row["store_id"]) for row in _rows(engine, "A")}
        engine.on_session_start("B", platform="cli", context_length=200_000, conversation_id="conv")
        live = [*history, *_turns(10, 4, 600.0)]
        engine.ingest(live)
        assert engine._load_compression_carry_ranges() == []
        engine.compress(live)
        claimed = _assert_claims_are_in_the_input(engine, summaries)
        assert not a_ids & set(claimed)
        assert len(_rows(engine, "B")) == len(live) - 1  # B keeps its own copy of every host row
    finally:
        engine.shutdown()


def test_c_positive_control_a_verified_compression_child_inherits_its_parents_rows(tmp_path):
    _state_db(tmp_path, [("P", None, "compression"), ("C", "P", None)])
    history = [SYSTEM, *_turns(1, 4, 0.0)]
    engine = _engine(tmp_path, "P")
    try:
        engine.ingest(history)
        engine.on_session_start("C", platform="cli", context_length=200_000, conversation_id="conv")
        engine.ingest([*history, *_turns(10, 1, 600.0)])
        assert [str(row["content"]) for row in _rows(engine, "C")] == [
            m["content"] for m in _turns(10, 1, 600.0)
        ]
        assert {source for source, _a, _b in engine._load_compression_carry_ranges()} == {"P"}
    finally:
        engine.shutdown()


def test_d_two_identical_gateway_messages_at_one_timestamp_are_both_stored(tmp_path):
    engine = _engine(tmp_path)
    ok = _u("ok", 700.0)
    try:
        head = [SYSTEM, *_turns(1, 2, 0.0)]
        engine.ingest([*head, dict(ok), _a("reply one", 701.0)])
        live = [*head, dict(ok), _a("reply one", 701.0), dict(ok), _a("reply two", 702.0)]
        engine.ingest(live)
        assert [str(row["content"]) for row in _rows(engine)].count("ok") == 2
    finally:
        engine.shutdown()
    restarted = _engine(tmp_path)  # a restart replays the same host list: still exactly two
    try:
        restarted.ingest(live)
        assert [str(row["content"]) for row in _rows(restarted)].count("ok") == 2
    finally:
        restarted.shutdown()


@pytest.mark.parametrize("flag", ["true", "false"])
def test_e_legacy_null_observed_at_store_behaves_unchanged(tmp_path, monkeypatch, summaries, flag):
    """Rows without a host stamp (a legacy store, a host that sends none) take today's path."""
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", flag)
    history = [SYSTEM, *[{k: v for k, v in m.items() if k != "timestamp"} for m in _turns(1, 4, 0.0)]]
    engine = _engine(tmp_path)
    try:
        engine.ingest(history)
    finally:
        engine.shutdown()
    engine = _engine(tmp_path)
    try:
        live = [*history, _u("[T9] later" + PAD, None), _a("reply to T9", None), *_turns(10, 3, 600.0)]
        out = engine.compress(live)
        rows = _rows(engine)
        assert all(row.get("observed_at") is None for row in rows[: len(history) - 1])
        assert [str(row["content"]) for row in rows] == [m["content"] for m in live[1:]]
        assert engine._last_compression_status == "compacted"
        assert _relations(engine) == []
        _assert_claims_are_in_the_input(engine, summaries)
        assert out
    finally:
        engine.shutdown()


# -- per rule ----------------------------------------------------------------------------------------

def test_r1_replay_after_restart_is_matched_per_occurrence(tmp_path):
    history = [SYSTEM, *_turns(1, 4, 0.0)]
    engine = _engine(tmp_path)
    try:
        engine.ingest(history)
    finally:
        engine.shutdown()
    engine = _engine(tmp_path)
    try:  # the host re-issues its list with a row it dropped: the rest are the same occurrences
        live = [SYSTEM, *_turns(1, 1, 0.0), *_turns(3, 2, 0.0), *_turns(10, 1, 600.0)]
        engine.ingest(live)
        texts = [str(row["content"]) for row in _rows(engine)]
        assert len(texts) == len(history) - 1 + 2 and len(set(texts)) == len(texts)
    finally:
        engine.shutdown()


def test_r2_a_live_composite_of_stored_rows_is_recognised_with_a_witness(tmp_path, summaries):
    """H3 merge: C = R + "\\n\\n" + U, both stored, carries R's stamp. Nothing new is stored; the
    decomposition is recorded; a summary of C claims R and U with their bytes in its input."""
    engine = _engine(tmp_path)
    r, u = _u("R prompt" + PAD, 500.0), _u("U prompt" + PAD, 510.0)
    try:
        head = [SYSTEM, *_turns(1, 3, 0.0)]
        engine.ingest([*head, r, u])  # a failed turn: two consecutive user rows
        composite = _u(r["content"] + "\n\n" + u["content"], 500.0)
        live = [*head, composite, _a("reply to U", 511.0), *_turns(10, 4, 600.0)]
        engine.ingest(live)
        texts = [str(row["content"]) for row in _rows(engine)]
        assert composite["content"] not in texts and texts.count(r["content"]) == 1
        ids = {str(row["content"]): int(row["store_id"]) for row in _rows(engine)}
        kinds = {(rel[1], rel[2]) for rel in _relations(engine)}
        assert {("composite", ids[r["content"]]), ("composite", ids[u["content"]])} <= kinds
        engine.compress(live)
        claimed = _assert_claims_are_in_the_input(engine, summaries)
        assert {ids[r["content"]], ids[u["content"]]} <= set(claimed)
    finally:
        engine.shutdown()


def test_r3_remainder_is_stored_once_byte_exact(tmp_path):
    """A held head plus a new remainder: U is stored once, with its exact bytes and separators, and
    the recorded constituents rebuild the host composite byte for byte."""
    engine = _engine(tmp_path)
    r = _u("head R\n\n  indented line \n\n\nthree newlines\t", 500.0)
    remainder = "  remainder U \n\nsecond  paragraph\n\n\n  tail \t"
    try:
        head = [SYSTEM, *_turns(1, 2, 0.0)]
        engine.ingest([*head, r])
        composite = _u(r["content"] + "\n\n" + remainder, 500.0)
        engine.ingest([*head, composite, _a("reply", 501.0)])
        rows = _rows(engine)
        texts = [str(row["content"]) for row in rows]
        assert texts.count(r["content"]) == 1 and texts.count(remainder) == 1
        assert composite["content"] not in texts
        stored_u = next(row for row in rows if row["content"] == remainder)
        assert stored_u.get("observed_at") is None  # its own host stamp is unknown
        group = sorted((rel for rel in _relations(engine) if rel[1] == "composite"), key=lambda rel: rel[3])
        by_id = {int(row["store_id"]): str(row["content"]) for row in rows}
        assert "\n\n".join(by_id[rel[2]] for rel in group) == composite["content"]
        engine.ingest([*head, composite, _a("reply", 501.0)])  # re-reading it stores nothing
        assert len(_rows(engine)) == len(rows)
    finally:
        engine.shutdown()


def test_r5_null_stamp_is_backfilled_only_for_the_proven_occurrence(tmp_path):
    engine = _engine(tmp_path)
    try:
        head = [SYSTEM, *[{k: v for k, v in m.items() if k != "timestamp"} for m in _turns(1, 2, 0.0)]]
        engine.ingest(head)
    finally:
        engine.shutdown()
    engine = _engine(tmp_path)
    try:  # a restart: the host now stamps the same rows
        engine.ingest([SYSTEM, *_turns(1, 2, 0.0), *_turns(10, 1, 600.0)])
        rows = _rows(engine)
        assert len(rows) == 6
        assert [row.get("observed_at") for row in rows[:4]] == [m["timestamp"] for m in _turns(1, 2, 0.0)]
    finally:
        engine.shutdown()


def test_flag_off_writes_no_identity_state(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "false")
    engine = _engine(tmp_path)
    try:
        r = _u("R" + PAD, 500.0)
        engine.ingest([SYSTEM, r])
        engine.ingest([SYSTEM, _u(r["content"] + "\n\nU", 500.0)])
        assert _relations(engine) == []
        assert not engine._store._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='message_relations'"
        ).fetchone()
    finally:
        engine.shutdown()
