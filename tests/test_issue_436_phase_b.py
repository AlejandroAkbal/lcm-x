"""#436 rc1 Phase B: B-ID-1 (a repeated user text merged into a dangling R) and B-ID-2 (a profile
rebind reusing the identity-anchor caches against another database)."""

from __future__ import annotations

from collections import Counter

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_issue_436_identity_anchor import (
    PAD,
    _a,
    _assert_claims_are_in_the_input,
    _engine,
    _relations,
    _rows,
    _turns,
    _u,
    summaries,  # noqa: F401 (fixture)
)


def _stored(rows) -> Counter:
    return Counter((row["role"], str(row["content"])) for row in rows)


def _merge_turn(engine, tmp_path, new_u, *, host_object, restart):
    """[U@1, A@2, R@3], then Hermes' _merge_consecutive_users folds the new U into the dangling R
    (R's dict keeps its stamp): [U@1, A@2, "R\\n\\nU"@3, reply@5]."""
    u, a, r = _u(new_u[0], 1.0), _a("reply to U" + PAD, 2.0), _u("[R] failed turn" + PAD, 3.0)
    r_text = r["content"]
    engine.ingest([u, a, r])
    if restart:
        engine.shutdown()
        engine = _engine(tmp_path)
    if host_object == "in-place":
        composite = r
        composite["content"] = r_text + "\n\n" + new_u[1]
        live = [u, a, composite, _a("reply to R+U", 5.0)]
    else:
        live = [dict(u), dict(a), _u(r_text + "\n\n" + new_u[1], 3.0), _a("reply to R+U", 5.0)]
    engine.ingest(live)
    return engine, live, r_text


def _assert_no_relation_older_than_its_donor(engine):
    for head, kind, member, _ordinal in _relations(engine):
        assert kind != "composite" or member is None or int(member) >= int(head), (head, member)


LONG = "[U] please go on with the report" + PAD


@pytest.mark.parametrize("restart", [False, True], ids=["steady", "restart"])
@pytest.mark.parametrize("host_object", ["in-place", "new-dict"])
@pytest.mark.parametrize("text, second", [
    ("continue", "continue"), (LONG, LONG),  # B-ID-1: the new U repeats an earlier user row
    ("continue", "continue (again)"), (LONG, LONG + " again"),  # control: distinct text
], ids=["repeat-continue", "repeat-long", "control-continue", "control-long"])
def test_bid1_merged_user_turn_is_stored_even_when_its_text_repeats_an_earlier_row(tmp_path, text, second, host_object, restart):
    engine = _engine(tmp_path)
    try:
        engine, _live, r_text = _merge_turn(engine, tmp_path, (text, second), host_object=host_object, restart=restart)
        expected = Counter([("user", text), ("assistant", "reply to U" + PAD), ("user", r_text),
                            ("user", second), ("assistant", "reply to R+U")])
        assert _stored(_rows(engine)) == expected
        _assert_no_relation_older_than_its_donor(engine)
    finally:
        engine.shutdown()


def test_bid1_merge_turn_compacts_with_contiguous_coverage(tmp_path, summaries):  # noqa: F811
    """After the B-ID-1 merge turn a forced compaction commits, covers a contiguous prefix that
    includes the repeated U, and never claims a source whose text is not in its input."""
    engine = _engine(tmp_path)
    try:
        engine, live, _r_text = _merge_turn(engine, tmp_path, ("continue", "continue"), host_object="in-place", restart=False)
        live = [*live, *_turns(10, 4, 600.0)]
        engine.ingest(live)
        statuses = []
        for _ in range(3):
            live = engine.compress(live, force=True)
            statuses.append(engine._last_compression_status)
        assert "error" not in statuses and "compacted" in statuses, engine._last_compression_noop_reason
        claimed = set(_assert_claims_are_in_the_input(engine, summaries))
        frontier = int(engine._last_compacted_store_id or 0)
        owned = [int(row["store_id"]) for row in _rows(engine)]
        assert [sid for sid in owned if sid <= frontier] == sorted(claimed & set(owned))
        repeats = [int(row["store_id"]) for row in _rows(engine) if row["content"] == "continue"]
        assert len(repeats) == 2 and all(sid <= frontier for sid in repeats)
    finally:
        engine.shutdown()


def _home_engine(home) -> LCMEngine:
    home.mkdir(parents=True, exist_ok=True)
    return LCMEngine(config=LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1), hermes_home=str(home))


def _start(engine, home) -> None:
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv", hermes_home=str(home))


@pytest.mark.parametrize("engine_kind", ["reused-from-profile-A", "fresh-engine"])
def test_bid2_profile_rebind_never_aliases_another_databases_row(tmp_path, engine_kind):
    """A / S stores "R\\n\\nU"@20 as row 1; B / S holds X@5 (row 1) and R@10 (row 2). A's engine
    rebinds to B, replays B, then sees [X@5, "R\\n\\nU"@10, reply@30]: U is stored in B, and no
    alt_stamp points at B's unrelated row 1."""
    home_a, home_b = tmp_path / "profile-a", tmp_path / "profile-b"
    r_text, u_text = "[R] prompt in profile B" + PAD, "[U] follow-up"
    x, r = _a("[X] assistant row in B" + PAD, 5.0), _u(r_text, 10.0)
    seed = _home_engine(home_b)
    try:
        _start(seed, home_b)
        seed.ingest([x, r])
    finally:
        seed.shutdown()
    engine = _home_engine(home_a if engine_kind == "reused-from-profile-A" else home_b)
    try:
        if engine_kind == "reused-from-profile-A":
            _start(engine, home_a)
            engine.ingest([_u(r_text + "\n\n" + u_text, 20.0)])
        _start(engine, home_b)
        engine.ingest([dict(x), dict(r)])
        engine.ingest([dict(x), _u(r_text + "\n\n" + u_text, 10.0), _a("reply to R+U", 30.0)])
        rows = _rows(engine)
        assert _stored(rows) == Counter([("assistant", x["content"]), ("user", r_text), ("user", u_text),
                                         ("assistant", "reply to R+U")])
        assert not [rel for rel in _relations(engine) if rel[1] == "alt_stamp"]
    finally:
        engine.shutdown()
