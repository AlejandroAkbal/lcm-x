"""#559: a leaf chunk must not end inside a parallel tool-call group.

One assistant row calls ``write_file`` (A) and ``lcm_expand`` (B). ``lcm_expand`` ingests the
assistant row and result A mid-turn; the large result B is stored at the end of the turn. When
the token-greedy leaf selector stops between A and B (or right after the assistant row), the
summary covers the assistant, the result left raw after it is an orphan, the host's sequence
repair drops it, and every later compaction fails with a non-contiguous publication. The driver
replays Hermes' in-place commit sequence (native recovery off) against the real engine.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
import importlib.util
from collections import Counter
from copy import deepcopy

import pytest

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SID = "S559"
FILLER = "alpha beta gamma delta " * 30  # ~120 tokens


def _stub_summarizer():
    calls = {"n": 0}

    def summarize(*args, **kwargs):
        calls["n"] += 1
        return f"Stub summary #{calls['n']}.\nExpand for details about: stub", 1

    return summarize


def _hermes_python():
    python = os.environ.get("LCM_REAL_HERMES_PYTHON")
    if python:
        return python, os.environ.get("LCM_REAL_HERMES_SRC", "")
    if importlib.util.find_spec("hermes_cli") is not None:
        return sys.executable, ""
    return None


def _emulated_repair(messages):
    """Hermes' stray-tool-result pass: a result whose call no assistant row declares is dropped."""
    declared = {
        str(call.get("id") or "")
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    }
    return [m for m in messages if m.get("role") != "tool" or str(m.get("tool_call_id") or "") in declared]


def _real_repair(messages):
    """Hermes' own repair_message_sequence (pinned host), in its interpreter."""
    python, src = _hermes_python()
    done = subprocess.run(
        [python, "-c", "import json, sys\nfrom agent.agent_runtime_helpers import repair_message_sequence\n"
         "m = json.load(sys.stdin)\nrepair_message_sequence(None, m)\nprint(json.dumps(m))"],
        input=json.dumps(messages), cwd=src or "/", capture_output=True, text=True, timeout=300, check=False,
        env={"HOME": "/nonexistent", "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


REPAIRS = {"emulated": _emulated_repair, "real-hermes": _real_repair}
_REAL = pytest.mark.skipif(_hermes_python() is None, reason="no real Hermes runtime available")
REPAIR_PARAMS = [pytest.param("emulated"), pytest.param("real-hermes", marks=_REAL)]


def _call(call_id, name):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _group(shape):
    """The parallel-call turn. ``between``: the budget ends between results A and B;
    ``after-assistant``: it ends right after the assistant row; ``control``: the group fits."""
    result_a = "wrote notes.txt" if shape != "after-assistant" else "wrote notes.txt " + FILLER * 3
    result_b = "expanded node 1: " + ("zeta eta theta iota " * 700 if shape != "control" else "short")
    return [
        {"role": "user", "content": "write the notes file and expand node 1"},
        {"role": "assistant", "content": "", "tool_calls": [_call("call_A", "write_file"),
                                                            _call("call_B", "lcm_expand")]},
        {"role": "tool", "tool_call_id": "call_A", "content": result_a},
        {"role": "tool", "tool_call_id": "call_B", "content": result_b},
        {"role": "assistant", "content": "done: notes written and node 1 expanded"},
    ]


def _turn(i):
    return [{"role": "user", "content": f"U{i:02d} " + FILLER}, {"role": "assistant", "content": f"R{i:02d} ok"}]


def _key(message):
    if message.get("role") == "tool":
        return ("tool", message.get("tool_call_id"))
    if message.get("tool_calls"):
        return ("assistant", tuple(call["id"] for call in message["tool_calls"]))
    return (message.get("role"), message.get("content"))


def _run(tmp_path, monkeypatch, caplog, shape, repair="emulated", mode="dynamic"):
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=4,
        leaf_chunk_tokens=400,
        large_output_externalization_enabled=True,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    if mode == "dynamic":
        config.dynamic_leaf_chunk_enabled = True
        config.dynamic_leaf_chunk_max = 400
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SID, platform="acp", context_length=200_000)
    if mode == "threshold-sweep":
        config.threshold_full_sweep_enabled = True
    caplog.set_level(logging.WARNING)
    host = [{"role": "system", "content": "You are the #559 probe agent."}]
    produced = []

    def add(rows, ingest=True):
        for row in rows:
            host.append(dict(row))
            produced.append(_key(row))
            if ingest:
                engine.ingest(host)

    out = {}
    try:
        add(_turn(1))
        group = _group(shape)
        add(group[:1])
        add(group[1:3], ingest=False)
        engine.ingest(host)  # lcm_expand ingests the assistant row and result A mid-turn
        add(group[3:4], ingest=False)
        engine.ingest(host)  # result B at the end of the turn
        add(group[4:])
        for i in (3, 4):
            add(_turn(i))
        add(_turn(5)[:1])
        if mode == "threshold-sweep":
            engine.threshold_tokens = 1
        compressed = engine.compress(list(host))
        out["status_1"] = engine._last_compression_status
        engine.on_session_end(SID, list(host))
        engine.on_session_start(SID, boundary_reason="compression", old_session_id=SID, platform="acp")
        host[:] = REPAIRS[repair](deepcopy(compressed))
        add(_turn(5)[1:])
        for i in range(6, 12):
            add(_turn(i))
        add(_turn(12)[:1])
        if mode == "threshold-sweep":
            engine.threshold_tokens = 1
        engine.compress(list(host))
        out["status_2"] = engine._last_compression_status
        out["noop_2"] = engine._last_compression_noop_reason
        nodes = [json.loads(ids) for (ids,) in engine._dag._conn.execute(
            "SELECT source_ids FROM summary_nodes WHERE source_type = 'messages' ORDER BY node_id")]
    finally:
        engine.shutdown()
    conn = sqlite3.connect(str(tmp_path / "lcm.db"))
    try:
        rows = conn.execute(
            "SELECT store_id, role, content, tool_call_id, tool_calls FROM messages ORDER BY store_id").fetchall()
    finally:
        conn.close()
    stored = Counter(
        ("tool", tcid) if role == "tool" else
        ("assistant", tuple(c["id"] for c in json.loads(calls))) if calls and json.loads(calls) else (role, content)
        for _sid, role, content, tcid, calls in rows
        if role != "system"  # the system prompt is an anchor, not a transcript row
    )
    expected = Counter(produced)
    group = {sid for sid, _r, _c, tcid, calls in rows if tcid in ("call_A", "call_B") or "call_A" in (calls or "")}
    out["conflicts"] = sum("publication_invariant_conflict" in r.getMessage() for r in caplog.records)
    out["missing"] = sorted(map(str, (expected - stored).elements()))
    out["duplicates"] = sorted(map(str, (stored - expected).elements()))
    out["split_leaves"] = sum(0 < len(group & set(ids)) < len(group) for ids in nodes)
    out["leaves"] = len(nodes)
    return out


@pytest.mark.parametrize("repair", REPAIR_PARAMS)
@pytest.mark.parametrize("mode", ["dynamic", "threshold-sweep"])
@pytest.mark.parametrize("shape", ["between", "after-assistant", "control"])
def test_leaf_chunk_never_splits_a_parallel_tool_group(tmp_path, monkeypatch, caplog, shape, mode, repair):
    out = _run(tmp_path, monkeypatch, caplog, shape, repair=repair, mode=mode)
    print("ISSUE559=" + json.dumps({"shape": shape, "mode": mode, "repair": repair, **out}, sort_keys=True))
    assert out["status_1"] == "compacted", out
    assert out["status_2"] == "compacted" and out["conflicts"] == 0, out
    assert out["missing"] == [] and out["duplicates"] == [], out
    assert out["split_leaves"] == 0, out


def _rows(*roles_and_ids):
    rows = []
    for item in roles_and_ids:
        if item == "u":
            rows.append({"role": "user", "content": "u"})
        elif item == "r":
            rows.append({"role": "assistant", "content": "r"})
        elif item.startswith("a:"):
            rows.append({"role": "assistant", "content": "", "tool_calls": [_call(c, "t") for c in item[2:].split(",")]})
        else:
            rows.append({"role": "tool", "tool_call_id": item[2:] or None, "content": "x"})
    return rows


@pytest.mark.parametrize("rows, end, expected", [
    (("u", "a:A,B", "t:A", "t:B", "r"), 3, 1),  # between sibling results: trim before the group
    (("u", "a:A,B", "t:A", "t:B", "r"), 2, 1),  # right after the assistant row
    (("u", "a:A,B", "t:A", "t:B", "r"), 4, 4),  # group closed: unchanged
    (("u", "a:A,B", "t:A", "t:B", "r"), 1, 1),  # before the group: unchanged
    (("a:A,B", "t:A", "t:B", "r"), 1, 3),  # trimming would empty the chunk: take the whole group
    (("a:A,B", "t:A", "t:B"), 2, 3),  # the group ends the list
    (("u", "a:A", "t:"), 2, 2),  # a result without an id cannot be classified: unchanged
    (("u", "a:A", "t:Z"), 2, 2),  # a result the assistant did not declare: unchanged
    (("u", "r"), 2, 2),
])
def test_tool_group_safe_end(rows, end, expected):
    from hermes_lcm.fresh_tail import tool_group_safe_end

    assert tool_group_safe_end(_rows(*rows), end) == expected


def test_leaf_rescue_fallback_does_not_split_a_group(tmp_path):
    """The rescue's last-row fallback trims to a group boundary, too."""
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), leaf_chunk_tokens=100_000),
                       hermes_home=str(tmp_path / "home"))
    try:
        chunk = _rows("u", "a:A,B", "t:A", "t:B")
        assert engine._next_leaf_rescue_chunk(chunk, 50) == chunk[:1]
        assert engine._next_leaf_rescue_chunk(chunk[1:], 50) == chunk[1:]  # only the group: no smaller chunk
    finally:
        engine.shutdown()
