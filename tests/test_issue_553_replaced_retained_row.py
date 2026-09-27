"""#553: a RETAINED last user row R of an in-place compaction that the host replaced or merged.

The process dies after a preflight compaction commits and before the reply, so the restored list ends
on R. The next turn glues its prompt U into R and Hermes' persist step rewrites that row to U alone.
A (S2): the in-place durable walk admits U at R's position (#519 R1's predicate); R is recorded and
the publication passes it. B (S1): a stored #535 composite ``R + "\\n\\n" + U`` also maps U alone.
D': a gateway reload re-merges R and U from state.db; ``R + "\\n\\n" + U`` maps the stored U when R is
the recorded row. E: the durable walk after a restart accepts that re-merged row over the stored U.
Every control stores what the base stores (duplicates or today's error, never a skipped row).
"""
from __future__ import annotations

import json
import re
from copy import deepcopy

import pytest

import hermes_lcm.reconcile as reconcile_module
from hermes_lcm.engine import LCMEngine
from tests.test_compression_boundary import _turn
from tests.test_issue_535_retained_row_merge import NEW, _all_rows, _dups, _persisted, _raw_count, _Session

KEY = "host_replaced_rows:S0"
REPLY = {"role": "assistant", "content": "reply to NEW-0"}


def _no_a(monkeypatch):
    monkeypatch.setattr(
        LCMEngine, "_in_place_proof_cursor", lambda self, messages, floor: self._cursor_from_durable_commit_proof(messages),
        raising=False,
    )


def _no_d(monkeypatch):
    monkeypatch.setattr(LCMEngine, "_restored_merge_form", lambda self, base, row, recorded: None, raising=False)


def _no_de(monkeypatch):
    _no_d(monkeypatch)
    monkeypatch.setattr(reconcile_module, "_merge_append_cuts", lambda content, limit=64: [], raising=False)


def _base_behaviour(monkeypatch):
    """The base: A, B, D' and E never hold."""
    _no_a(monkeypatch)
    _no_de(monkeypatch)
    monkeypatch.setattr(LCMEngine, "_merge_append_remainder", lambda self, base, row: None, raising=False)


def _last_user_id(engine):
    return engine._store._conn.execute("SELECT max(store_id) FROM messages WHERE role = 'user'").fetchone()[0]


def _nodes(engine):
    return [json.loads(ids) for (ids,) in engine._dag._conn.execute(
        "SELECT source_ids FROM summary_nodes WHERE source_type = 'messages' ORDER BY node_id")]


def _compact_three(s):
    out = []
    for first in (40, 50, 60):
        s.turns(first)
        out.append((s.compact()[0], s.engine._last_compression_noop_reason))
    return out


def _replaced(tmp_path, monkeypatch, *, mutate=None, reply=False):
    """S2 in place: the host list [S, kept tail, U, reply] after the persist step replaced R by U."""
    s = _Session(tmp_path, monkeypatch, reply=reply)
    r_id, r_text, before = _last_user_id(s.engine), s.host[-1]["content"], len(_all_rows(s.engine))
    if mutate is None:
        s.host[-1] = {"role": "user", "content": NEW}
    else:
        mutate(s)
    s.add(dict(REPLY))
    return s, r_id, r_text, before


def test_in_place_replaced_row_is_stored_once_and_compacts(tmp_path, monkeypatch):
    """A: U and its reply are stored once; three compactions commit; R ends below the frontier and
    in no node's sources (passed, never claimed: its bytes stay stored)."""
    s, r_id, _r, before = _replaced(tmp_path, monkeypatch)
    try:
        assert len(_all_rows(s.engine)) == before + 2 and _dups(s.engine) == 0 and _raw_count(s.engine, NEW) == 1
        assert s.engine._store.read_metadata_json(KEY) == {"version": 1, "store_ids": [r_id]}
        assert [status for status, _ in _compact_three(s)] == ["compacted"] * 3
        assert s.engine._last_compacted_store_id > r_id and r_id not in sum(_nodes(s.engine), [])
        assert _dups(s.engine) == 0 and _raw_count(s.engine, NEW) == 1
    finally:
        s.engine.shutdown()


def test_positive_control_without_a_re_stores_and_errors(tmp_path, monkeypatch):
    _no_a(monkeypatch)
    s, _r_id, _r, before = _replaced(tmp_path, monkeypatch)
    try:
        assert len(_all_rows(s.engine)) == before + 7 and _dups(s.engine) == 5
        assert [status for status, _ in _compact_three(s)] == ["error"] * 3
        assert s.engine._store.read_metadata_json(KEY) is None
    finally:
        s.engine.shutdown()


def _failing_record_write(s, monkeypatch):
    original = s.engine._store.write_metadata_json

    def failing(keys, serialized, **kwargs):
        if any(str(key).startswith("host_replaced_rows") for key in keys):
            raise RuntimeError("database is locked")
        return original(keys, serialized, **kwargs)

    monkeypatch.setattr(s.engine._store, "write_metadata_json", failing)
    s.host[-1] = {"role": "user", "content": NEW}


def _run(tmp_path, monkeypatch, disable, *, mutate=None, reply=False):
    disable(monkeypatch)
    s, _r_id, _r, _before = _replaced(tmp_path, monkeypatch, mutate=mutate, reply=reply)
    try:
        stored = _all_rows(s.engine)
        return stored, [status for status, _ in _compact_three(s)], _all_rows(s.engine), s.engine._store.read_metadata_json(KEY)
    finally:
        s.engine.shutdown()


def _assistant_last(s):
    s.host[-1] = {"role": "user", "content": NEW}  # the output ended on an assistant row


def _tool_row(s):
    s.host[-1] = {"role": "user", "content": NEW, "tool_call_id": "call_553"}


def _non_last_edit(s):
    s.host[-3] = {**s.host[-3], "content": s.host[-3]["content"] + " (edited by host)"}
    s.host[-1] = {"role": "user", "content": NEW}


def _own_row_after_proof(s):
    s.engine.ingest(list(s.host) + [{"role": "assistant", "content": "a reply stored before the restart"}])
    s.engine.shutdown()
    s.engine = s.make("S0")
    s.host[-1] = {"role": "user", "content": NEW}


def _native(s):
    key = "compaction_commit_proof:S0"
    payload = s.engine._store.read_metadata_json(key)
    n = len(payload["effective_sha256"])
    payload.update(native=True, native_summary_index=0, droppable=[False] * n, skip_landing=[False] * n)
    s.engine._store.write_metadata_json([key], json.dumps(payload, sort_keys=True))
    s.engine.shutdown()
    s.engine = s.make("S0")
    s.host[-1] = {"role": "user", "content": NEW}


A_NEGATIVE = {
    "assistant_last_row": (_assistant_last, True),
    "tool_bearing_last_row": (_tool_row, False),
    "edit_of_a_non_last_row": (_non_last_edit, False),
    "own_row_after_the_proof": (_own_row_after_proof, False),
    "native_on": (_native, False),
}


@pytest.mark.parametrize("case", sorted(A_NEGATIVE))
def test_a_negative_controls_store_what_the_base_stores(tmp_path, monkeypatch, case):
    mutate, reply = A_NEGATIVE[case]
    fixed = _run(tmp_path / "fixed", monkeypatch, lambda m: None, mutate=mutate, reply=reply)
    no_a = _run(tmp_path / "no_a", monkeypatch, _no_a, mutate=mutate, reply=reply)
    monkeypatch.undo()
    base = _run(tmp_path / "base", monkeypatch, _base_behaviour, mutate=mutate, reply=reply)
    assert fixed == no_a == base and fixed[3] is None


def test_a_failed_record_write_is_todays_behaviour(tmp_path, monkeypatch):
    """Fail-soft: a failed metadata write gives today's walk (today's rows, today's error)."""
    fixed = _run(tmp_path / "fixed", monkeypatch, lambda m: None, mutate=lambda s: _failing_record_write(s, monkeypatch))
    monkeypatch.undo()
    base = _run(tmp_path / "base", monkeypatch, _base_behaviour)
    assert fixed == base and fixed[1] == ["error"] * 3 and fixed[3] is None


# -- B (S1): the composite was stored whole, then the persist step replaced it by U alone -----------


@pytest.mark.parametrize("mode", ["inplace", "rotation"])
def test_persisted_composite_maps_the_replaced_row(tmp_path, monkeypatch, mode):
    s, before = _persisted(tmp_path, monkeypatch, mode)
    try:
        assert len(_all_rows(s.engine)) == before + 2
        assert [status for status, _ in _compact_three(s)] == ["compacted"] * 3
        assert _dups(s.engine) == 0 and _raw_count(s.engine, NEW) == 1
        composite = s.engine._store._conn.execute(
            "SELECT store_id FROM messages WHERE content LIKE ? ORDER BY store_id", (f"%\n\n{NEW}",)).fetchone()[0]
        assert {composite - 1, composite} <= set(sum(_nodes(s.engine), []))  # N2 consumed B with C
    finally:
        s.engine.shutdown()


def _carrier_run(tmp_path, monkeypatch, *, fixed):
    """Tail 1: the composite is #499's carrier extension (a summary carrier plus the merged row)."""
    if not fixed:
        monkeypatch.setattr(LCMEngine, "_merge_append_remainder", lambda self, base, row: None, raising=False)
    s = _Session(tmp_path, monkeypatch, tail=1)
    try:
        s.engine.ingest([*map(dict, s.host[:-1]), {"role": "user", "content": s.host[-1]["content"] + "\n\n" + NEW}])
        s.host[-1] = {"role": "user", "content": NEW}
        s.add(dict(REPLY))
        stored = _all_rows(s.engine)
        return stored, [status for status, _ in _compact_three(s)], _all_rows(s.engine)
    finally:
        s.engine.shutdown()


def test_b_negative_control_carrier_composite(tmp_path, monkeypatch):
    fixed = _carrier_run(tmp_path / "fixed", monkeypatch, fixed=True)
    monkeypatch.undo()
    base = _carrier_run(tmp_path / "base", monkeypatch, fixed=False)
    assert fixed == base


def test_b_remainder_is_only_a_plain_merge_append(tmp_path, monkeypatch):
    s = _Session(tmp_path, monkeypatch)
    try:
        base = {"store_id": 1, "role": "user", "content": "B row"}
        remainder = s.engine._merge_append_remainder
        assert remainder(base, {"store_id": 2, "role": "user", "content": "B row\n\nX row"}) == "X row"
        assert remainder(base, {"store_id": 2, "role": "user", "content": "B row\nX row"}) is None  # single newline
        assert remainder(base, {"store_id": 2, "role": "user", "content": "B row\n\n  \n"}) is None  # blank remainder
        assert remainder(base, {"store_id": 2, "role": "assistant", "content": "B row\n\nX row"}) is None
        assert remainder({**base, "role": "assistant"}, {"store_id": 2, "role": "user", "content": "B row\n\nX row"}) is None
        assert remainder(base, {"store_id": 2, "role": "user", "content": "other\n\nB row\n\nX row"}) is None
    finally:
        s.engine.shutdown()


# -- D' + E: the gateway reload restores R and U from state.db and re-merges them ----------------


def _reloaded(s, r_text, joiner="\n\n", tail=NEW):
    host = deepcopy(s.host)
    assert host[-2] == {"role": "user", "content": NEW}
    host[-2] = {"role": "user", "content": r_text + joiner + tail}
    return host


@pytest.mark.parametrize("restart", [False, True], ids=["in-memory", "restart"])
def test_reload_re_merged_row_maps_and_n2_claims_the_recorded_row(tmp_path, monkeypatch, restart):
    s, r_id, r_text, _before = _replaced(tmp_path, monkeypatch)
    try:
        rows = _all_rows(s.engine)
        host = _reloaded(s, r_text)
        if restart:
            s.engine.shutdown()
            s.engine = s.make("S0")
        s.engine.ingest(host)
        assert _all_rows(s.engine) == rows  # nothing re-stored
        s.host[:] = host
        assert [status for status, _ in _compact_three(s)] == ["compacted"] * 3
        assert r_id in sum(_nodes(s.engine), []) and _dups(s.engine) == 0 and _raw_count(s.engine, NEW) == 1
    finally:
        s.engine.shutdown()


def _record_other(s, r_id):
    s.engine._store.write_metadata_json([KEY], json.dumps({"version": 1, "store_ids": [r_id - 2]}))


DE_NEGATIVE = {  # E is proof-anchored (R's digest), not record-anchored: that control disables D' only
    "single_newline_joiner": {"joiner": "\n"},
    "whitespace_remainder": {"tail": " \n "},
    "base_is_not_the_recorded_row": {"record": _record_other, "disable": _no_d},
}


def _de_run(tmp_path, monkeypatch, case, *, restart, fixed):
    spec = DE_NEGATIVE[case]
    if not fixed:
        spec.get("disable", _no_de)(monkeypatch)
    s, r_id, r_text, _before = _replaced(tmp_path, monkeypatch)
    try:
        if "record" in spec:
            spec["record"](s, r_id)
        host = _reloaded(s, r_text, spec.get("joiner", "\n\n"), spec.get("tail", NEW))
        if restart:
            s.engine.shutdown()
            s.engine = s.make("S0")
        s.engine.ingest(host)
        stored = _all_rows(s.engine)
        s.host[:] = host
        return stored, [status for status, _ in _compact_three(s)], _all_rows(s.engine)
    finally:
        s.engine.shutdown()


@pytest.mark.parametrize("restart", [False, True], ids=["in-memory", "restart"])
@pytest.mark.parametrize("case", sorted(DE_NEGATIVE))
def test_de_negative_controls_store_what_the_base_stores(tmp_path, monkeypatch, case, restart):
    """A holds in both runs (its record is the premise); D' and E must not change anything."""
    fixed = _de_run(tmp_path / "fixed", monkeypatch, case, restart=restart, fixed=True)
    monkeypatch.undo()
    base = _de_run(tmp_path / "base", monkeypatch, case, restart=restart, fixed=False)
    assert fixed == base


def test_ingest_ignore_pattern_pre_map_never_sees_the_553_forms(tmp_path, monkeypatch):
    """The ingest pre-map decides storage: it maps without the B / D' alternate forms."""
    s, _r_id, r_text, _before = _replaced(tmp_path, monkeypatch)
    try:
        host = _reloaded(s, r_text)
        full = s.engine._get_store_id_map_for_messages(host)
        gated = s.engine._get_store_id_map_for_messages(host, merge_forms=False)
        assert id(host[-2]) in full and id(host[-2]) not in gated
        seen = []
        original = LCMEngine._get_store_id_map_for_messages

        def spy(self, messages, occurrences=None, **kwargs):
            seen.append(kwargs.get("merge_forms", True))
            return original(self, messages, occurrences, **kwargs)

        monkeypatch.setattr(LCMEngine, "_get_store_id_map_for_messages", spy)
        s.engine._compiled_ignore_message_patterns = [re.compile("never-matches-553")]  # any pattern turns the pre-map on
        s.engine._ingest_messages(host + [dict(_turn(39)[0])])
        assert seen and seen[0] is False
    finally:
        s.engine.shutdown()
