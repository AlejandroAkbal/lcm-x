"""#436: host-timestamp-anchored, occurrence-bound message identity (REVISION 1).

(a)-(e) are the REVISION 1 acceptance counterexamples; the rest pin one rule each. Engine-level only:
a host list in, stored rows / relations / summary coverage out."""

from __future__ import annotations

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
