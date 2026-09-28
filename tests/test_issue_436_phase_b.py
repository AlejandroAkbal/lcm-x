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


def _merge_turn(engine, tmp_path, new_u, *, host_object, restart, u_stamp=1.0, reply=True):
    """[U@1, A@2, R@3], then Hermes' _merge_consecutive_users folds the new U into the dangling R
    (R's dict keeps its stamp): [U@1, A@2, "R\\n\\nU"@3, reply@5]."""
    u, a, r = _u(new_u[0], u_stamp), _a("reply to U" + PAD, 2.0), _u("[R] failed turn" + PAD, 3.0)
    r_text = r["content"]
    engine.ingest([u, a, r] if reply else [u, r])
    if restart:
        engine.shutdown()
        engine = _engine(tmp_path)
    if host_object == "in-place":
        composite = r
        composite["content"] = r_text + "\n\n" + new_u[1]
        live = [u, a, composite, _a("reply to R+U", 5.0)]
    else:
        live = [dict(u), dict(a), _u(r_text + "\n\n" + new_u[1], 3.0), _a("reply to R+U", 5.0)]
    if not reply:
        live.remove(a)
    engine.ingest(live)
    return engine, live, r_text


def _assert_shown_row_is_no_constituent(engine, text):
    """The host view shows the earlier U as its own occurrence: it is never a composite constituent."""
    shown = min(int(row["store_id"]) for row in _rows(engine) if row["content"] == text)
    assert not [rel for rel in _relations(engine) if rel[1] == "composite" and rel[2] == shown]


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
        _assert_shown_row_is_no_constituent(engine, text)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("restart", [False, True], ids=["steady", "restart"])
@pytest.mark.parametrize("reply", [True, False], ids=["replied", "unreplied"])
@pytest.mark.parametrize("text", ["continue", LONG], ids=["continue", "long"])
def test_bid1_an_unstamped_earlier_row_is_reserved_too(tmp_path, text, reply, restart):
    """Mixed legacy/current history: the earlier U has no host stamp (stored NULL, shown unstamped).
    It is still the view's own occurrence, so the new repeated U is stored, not absorbed."""
    engine = _engine(tmp_path)
    try:
        engine, _live, r_text = _merge_turn(engine, tmp_path, (text, text), host_object="in-place", restart=restart,
                                            u_stamp=None, reply=reply)
        expected = Counter([("user", text), ("user", r_text), ("user", text), ("assistant", "reply to R+U")]
                           + [("assistant", "reply to U" + PAD)] * reply
                           # A restart whose list no longer proves the stored tail re-stores an unstamped row
                           # (pre-existing: nothing keys it; flag off re-stores the whole list): duplication, never loss.
                           + [("user", text)] * restart)
        assert _stored(_rows(engine)) == expected
        _assert_shown_row_is_no_constituent(engine, text)
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


# -- round 3: a row answers any of its forms (exact, host-rewrite): the reservation, R1 and gap-fill match as a
# whole (maximum matching), never greedily per row. The adversarial form order is forced, not hash-seed luck.

def _forms_first(engine, monkeypatch, first: dict) -> None:
    """``_stored_row_forms`` of a row whose content is a key of ``first`` lists those texts, in that order."""
    original = engine._stored_row_forms

    def forms(row):
        exact = next(iter(original(row)))
        texts = first.get(str(row.get("content")))
        return original(row) if texts is None else [(exact[0], text, *exact[2:]) for text in texts]

    monkeypatch.setattr(engine, "_stored_row_forms", forms)


def _ids(engine, text):
    return [int(row["store_id"]) for row in _rows(engine) if row["content"] == text]


@pytest.mark.parametrize("stamp, a_text, b_text, restart", [
    (1.0, "alpha", "beta", False),  # one stamp: A answers {alpha, beta}, B only beta; the view shows both
    (None, "foo ", "foo", True),  # NULL stamps after a restart: A "foo " (override "foo"), B "foo"
], ids=["stamped-steady", "null-restart"])
def test_r3_a_multi_form_row_never_frees_a_shown_row_into_a_composite(tmp_path, monkeypatch, stamp, a_text, b_text, restart):
    engine = _engine(tmp_path)
    try:
        a, b, r = _u(a_text, stamp), _u(b_text, stamp), _u("[R] failed turn" + PAD, 3.0)
        head = [a, b, _a("reply" + PAD, 2.0)]
        engine.ingest([*head, r])
        if restart:
            engine.shutdown()
            engine = _engine(tmp_path)
        _forms_first(engine, monkeypatch, {a_text: [b_text, a_text]})
        shown = set(_ids(engine, a_text) + _ids(engine, b_text))
        r["content"] += "\n\n" + b_text  # the new U repeats B's text, merged into the dangling R
        engine.ingest([*head, r, _a("reply to R+U", 5.0)])
        members = [rel[2] for rel in _relations(engine) if rel[1] == "composite"]
        assert not shown & set(members), (shown, members)
        assert set(_ids(engine, b_text)) & set(members)  # the new U: stored, its own occurrence
    finally:
        engine.shutdown()


def test_r3_gap_fill_never_rehydrates_a_row_the_view_shows(tmp_path, monkeypatch):
    """R4 gap-fill: A answers {alpha, beta}, B only beta, the view shows alpha and beta: both rows are shown,
    none is rehydrated into the summarizer input."""
    engine = _engine(tmp_path)
    try:
        x = _u("[X] later user row" + PAD, 2.0)
        engine.ingest([_u("alpha", 1.0), _u("beta", 1.0), x])
        _forms_first(engine, monkeypatch, {"alpha": ["beta", "alpha"]})
        [b_id], [x_id] = _ids(engine, "beta"), _ids(engine, x["content"])
        out = engine._identity_anchor_summary_input([x], {id(x): x_id}, view=[_u("alpha", 1.0), _u("beta", 1.0), x])
        assert not [ids for _row, ids in out or () if b_id in ids], out
    finally:
        engine.shutdown()


def test_r3_r1_a_multi_form_row_leaves_the_single_form_row_its_occurrence(tmp_path, monkeypatch):
    """R1: A answers {xval, yval} (listed first in store order), B only xval; the host re-issues both. A takes
    yval and B xval: nothing is stored again."""
    engine = _engine(tmp_path)
    try:
        a, b = _u("yval", 1.0), _u("xval", 1.0)
        engine.ingest([_u("[P] dropped prefix" + PAD, 0.5), a, b, _a("reply" + PAD, 2.0)])
        _forms_first(engine, monkeypatch, {"yval": ["xval", "yval"]})
        engine.ingest([dict(b), dict(a), _a("reply" + PAD, 2.0), _u("[N] new" + PAD, 5.0)])  # the prefix row left
        assert [row["content"] for row in _rows(engine)].count("yval") == 1
        assert [row["content"] for row in _rows(engine)].count("xval") == 1
    finally:
        engine.shutdown()
