"""Reliability harness R1 (bench/instruments/reliability): scorers, registry and host loader, no Hermes.

Every bar gets a PASS and a FAIL path over a tiny synthetic cell dir (transcript.jsonl, phase-A.json,
db/lcm.db) built here.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.instruments.reliability import cells, hosts, plugin_tree, probe  # noqa: E402
from bench.instruments.reliability.scorers import bars, dupes, multiset  # noqa: E402

U = "[T{:02d}] user turn {}: alpha beta end."
R = "reply to T{:02d}: noted item {}."


def turn_events(t, *, reply=True, held=None, tags=None):
    text = U.format(t, t)
    return [{"phase": "A", "turn": t, "event": "user_sent", "tag": f"T{t:02d}", "content": text, "persist": text},
            {"phase": "A", "turn": t, "event": "turn_end", "tag": f"T{t:02d}", "held": held or text,
             "reply": R.format(t, t) if reply else None, "user_tags": tags or {f"T{t:02d}": 1}}]


def make(tmp_path, *, rows, events, nodes=(), phase=None, sids=None, parents=None, **cell_kw):
    d = tmp_path / "cell"
    (d / "db").mkdir(parents=True)
    if parents is not None:
        con = sqlite3.connect(d / "db" / "state.db")
        con.execute("create table sessions (id text primary key, parent_session_id text)")
        con.executemany("insert into sessions values (?,?)", parents.items())
        con.commit()
        con.close()
    con = sqlite3.connect(d / "db" / "lcm.db")
    con.execute("create table messages (store_id integer primary key, session_id text, role text, content text,"
                " tool_calls text, tool_call_id text, ingested_at real)")
    con.execute("create table summary_nodes (node_id integer primary key, session_id text, depth integer,"
                " source_ids text, source_type text, created_at real)")
    for i, (role, content, *rest) in enumerate(rows, 1):
        sid = sids[i - 1] if sids else "S0"
        con.execute("insert into messages values (?,?,?,?,?,?,?)", (i, sid, role, content, *(rest + [None, None])[:2], i * 10.0))
    for i, ids in enumerate(nodes, 1):
        con.execute("insert into summary_nodes values (?,?,?,?,?,?)", (i, "S0", 0, json.dumps(ids), "messages", 1.0))
    con.commit()
    con.close()
    comp = [{"phase": "A", "turn": i + 1, "event": "compaction", "compression_status": "compacted", "depth0_nodes": i + 1}
            for i in range(cell_kw.pop("passes", 2))] + cell_kw.pop("extra_events", [])
    (d / "transcript.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events + comp))
    base = {"phase": "A", "log_counts": {}, "counters": {"failed": [], "orphan_drops": 0, "native_max": 1},
            "compactions_logged": sum(e.get("compression_status") == "compacted" for e in comp), "final_check": {"published": True}}
    (d / "phase-A.json").write_text(json.dumps({**base, **(phase or {})}))
    cell = {"id": "t", "tool_plan": [], "native_recovery": False, "min_compactions": 2, "final_compaction_check": True,
            "bars": list(cells.BARS), **cell_kw}
    return bars.score(cell, d)


def clean_rows(n=3):
    return [r for t in range(1, n + 1) for r in (("user", U.format(t, t)), ("assistant", R.format(t, t)))]


def clean_events(n=3):
    return [e for t in range(1, n + 1) for e in turn_events(t)]


def test_clean_cell_passes_every_bar(tmp_path):
    out = make(tmp_path, rows=clean_rows(), events=clean_events())
    assert out["verdict"] == "PASS", out["failed_bars"]
    assert out["applicable_bars"] == ["B1", "B2", "B3", "B4", "B5"]


def test_b1_b2_duplicate_user_row_fails(tmp_path):
    out = make(tmp_path, rows=clean_rows() + [("user", U.format(2, 2) + " ")], events=clean_events())
    assert out["failed_bars"]["B1"] == {"T02": {"expected": 1, "stored": 2}}
    assert out["numbers"]["B2"]["surplus_rows"] == 1


def test_b2_catches_identical_reply_surplus_that_tags_cannot(tmp_path):
    events = clean_events()
    for e in events:
        if e["event"] == "turn_end":
            e["reply"] = "same reply."
    rows = [r if r[0] == "user" else ("assistant", "same reply.") for r in clean_rows()] + [("assistant", "same reply.")]
    out = make(tmp_path, rows=rows, events=events)
    assert set(out["failed_bars"]) == {"B2"}
    assert out["numbers"]["B2"]["duplicated_keys"] == 1 and out["numbers"]["B2"]["surplus_rows"] == 1


def test_b2_loss_and_crash_merge_composite(tmp_path):
    lost = make(tmp_path, rows=clean_rows()[:-1], events=clean_events())
    assert lost["numbers"]["B2"]["deficit_rows"] == 1 and "B1" in lost["failed_bars"]
    # Turn 2 crashed before its reply; turn 3's prompt merged into the dangling row (host composite).
    composite = U.format(2, 2) + "\n\n" + U.format(3, 3)
    events = turn_events(1) + turn_events(2)[:1] + turn_events(3, held=composite, tags={"T02": 1, "T03": 1})
    rows = clean_rows(1) + [("user", composite), ("assistant", R.format(3, 3))]
    assert make(tmp_path / "m", rows=rows, events=events)["verdict"] == "PASS"


def test_b3_b4(tmp_path):
    out = make(tmp_path, rows=clean_rows(), events=clean_events(),
               phase={"log_counts": {"publication_invariant_conflict": 2}, "counters": {"failed": ["T03"]}})
    assert out["failed_bars"]["B3"] == {"publication_invariant_conflict": 2}
    assert out["failed_bars"]["B4"]["failed_turns"] == ["T03"]
    unpublished = make(tmp_path / "f", rows=clean_rows(), events=clean_events(), phase={"final_check": {"published": False}})
    assert set(unpublished["failed_bars"]) == {"B4"}
    cleanup_only = {"outcome": "inconclusive", "engine_status": "sanitized", "attempts": [{}, {}], "entry": "x"}
    unsure = make(tmp_path / "i", rows=clean_rows(), events=clean_events(), phase={"final_check": cleanup_only})
    assert unsure["verdict"] == "INCONCLUSIVE" and unsure["failed_bars"] == {} and set(unsure["inconclusive_bars"]) == {"B4"}


def test_compact_transcript_fields_mean_held_equals_persist(tmp_path):
    events = clean_events()
    for e in events:
        if e["event"] == "turn_end":
            del e["held"]
            e["held_same"] = True
        else:
            e["persist"] = None
    assert make(tmp_path, rows=clean_rows(), events=events)["verdict"] == "PASS"


def test_b5_growth_minimum_and_log_parity(tmp_path):
    assert "B5" in make(tmp_path, rows=clean_rows(), events=clean_events(), passes=1)["failed_bars"]
    stall = clean_events() + [{"phase": "A", "turn": 3, "event": "compaction", "compression_status": "compacted",
                               "depth0_nodes": 1}]
    out = make(tmp_path / "s", rows=clean_rows(), events=stall, phase={"compactions_logged": 3})
    assert out["failed_bars"]["B5"]["non_growing_passes"] == [1]  # depth-0 sequence 1, 1, 2
    assert "B5" in make(tmp_path / "l", rows=clean_rows(), events=clean_events(), phase={"compactions_logged": 5})["failed_bars"]


def test_b6_split_group_and_orphans(tmp_path):
    calls = json.dumps([{"id": "c1"}, {"id": "c2"}])
    rows = clean_rows(1) + [("assistant", "", calls), ("tool", "a", None, "c1"), ("tool", "b", None, "c2")] + clean_rows(3)[2:]
    kw = {"tool_plan": [{"turns": [1], "calls": []}]}
    whole = make(tmp_path, rows=rows, events=clean_events(), nodes=[[1, 2, 3, 4, 5]], **kw)
    assert whole["verdict"] == "PASS", whole["failed_bars"]
    split = make(tmp_path / "s", rows=rows, events=clean_events(), nodes=[[1, 2, 3, 4]], **kw)
    assert split["failed_bars"]["B6"]["split_groups"] == 1
    orphan = make(tmp_path / "o", rows=rows, events=clean_events(), phase={"counters": {"failed": [], "orphan_drops": 1}}, **kw)
    assert orphan["failed_bars"]["B6"]["host_orphan_drops"] == 1


def native_events(n=3):
    events = clean_events(n)
    for e in events:
        if e["event"] == "turn_end":
            e["native_attempts"] = 1
    return events


def test_b7_native_health(tmp_path):
    ok = make(tmp_path, rows=clean_rows(), events=native_events(), native_recovery=True)
    assert ok["verdict"] == "PASS" and "B7" in ok["applicable_bars"]
    bad = make(tmp_path / "b", rows=clean_rows(), events=native_events(), native_recovery=True,
               phase={"log_counts": {"summary_generation_aborted": 1}, "counters": {"failed": [], "native_max": 2}})
    assert bad["failed_bars"]["B7"]["summary_generation_aborted"] == 1
    assert bad["failed_bars"]["B7"]["max_native_attempts_per_turn"] == 2


def test_multiset_normalisation_passes_and_a_split_reply_fails():  # R1.2 F1: a split is surplus, never PASS
    assert multiset.score([("user", "a  b\n")], [(1, "S", "user", "a b")])["verdict"] == "PASS"
    out = multiset.score([("user", "a  b\n"), ("assistant", "one two")],
                         [(1, "S", "user", "a b"), (2, "S", "assistant", "one"), (3, "S", "assistant", "two")])
    assert out["verdict"] == "FAIL" and out["split_keys"] == 1 and out["surplus_rows"] == 2


def test_f1_stored_only_key_fails_b2(tmp_path):
    out = make(tmp_path, rows=clean_rows() + [("user", "synthetic row nobody sent")], events=clean_events())
    assert out["numbers"]["B2"]["stored_rows_not_expected"] == 1 and out["numbers"]["B2"]["surplus_rows"] == 1
    assert "B2" in out["failed_bars"] and "B1" not in out["failed_bars"]


def test_f2_native_pass_needs_its_own_host_commit(tmp_path):
    native = [{"phase": "A", "turn": 3, "event": "compaction", "compression_status": "host_native"}]
    final = [{"phase": "A", "turn": 3, "event": "compaction", "compression_status": "host_native", "final": True}]
    unproven = make(tmp_path, rows=clean_rows(), events=clean_events(), extra_events=native + final)
    assert unproven["failed_bars"]["B5"]["native_passes_without_host_commit"] == ["A:T3"]
    events = clean_events()
    events[-1]["host_commits"] = 1  # turn 3's turn_end saw one host "committed" telemetry line
    proven = make(tmp_path / "p", rows=clean_rows(), events=events, extra_events=native + final, passes=1, min_compactions=2)
    assert proven["verdict"] == "PASS", proven["failed_bars"]
    assert proven["numbers"]["B5"]["published"] == 2  # the final forced pass is B4 evidence, not counted here
    crash = [{"phase": "A", "turn": 3, "event": "crash", "fault": "crash_after_rotation_before_child_row", "host_commits": 0}]
    killed = make(tmp_path / "k", rows=clean_rows(), events=clean_events(), extra_events=native + crash)
    assert killed["verdict"] == "PASS" and killed["numbers"]["B5"]["native_passes_interrupted_by_crash"] == ["A:T3"]
    assert killed["numbers"]["B5"]["published"] == 2  # the interrupted native pass is not counted
    final_lcm = [{"phase": "A", "turn": 3, "event": "compaction", "compression_status": "compacted", "final": True, "depth0_nodes": 9}]
    logged = make(tmp_path / "l", rows=clean_rows(), events=clean_events(), extra_events=final_lcm, phase={"compactions_logged": 3})
    assert logged["verdict"] == "PASS" and logged["numbers"]["B5"]["published"] == 2  # final LCM pass logs, is not counted


def test_f3_unproven_scenarios_are_unsupported(tmp_path):
    events = clean_events()
    events[1].update(tools_planned=2, tools_answered=1)
    calls = json.dumps([{"id": "c1"}, {"id": "c2"}])
    rows = clean_rows(1) + [("assistant", "", calls), ("tool", "a", None, "c1"), ("tool", "b", None, "c2")] + clean_rows(3)[2:]
    plan = {"tool_plan": [{"turns": [1], "calls": [{"name": "x"}, {"name": "y"}]}]}
    short = make(tmp_path, rows=rows, events=events, nodes=[[1, 2, 3, 4, 5]], **plan)
    assert short["verdict"] == "UNSUPPORTED" and "results seen 1" in short["reason"]
    events[1].update(tools_answered=2)
    assert make(tmp_path / "ok", rows=rows, events=events, nodes=[[1, 2, 3, 4, 5]], **plan)["verdict"] == "PASS"
    no_groups = make(tmp_path / "g", rows=clean_rows(), events=events, **plan)
    assert no_groups["verdict"] == "UNSUPPORTED" and "no tool-call group" in no_groups["reason"]
    no_native = make(tmp_path / "n", rows=clean_rows(), events=clean_events(), native_recovery=True)
    assert no_native["verdict"] == "UNSUPPORTED" and "zero native" in no_native["reason"]


def test_host_written_assistant_row_is_held_not_surplus(tmp_path):
    notice = "Your request was not processed. Send it again if you still want me to carry it out."
    rows = clean_rows() + [("assistant", notice)]
    assert "B2" in make(tmp_path, rows=rows, events=clean_events())["failed_bars"]
    events = clean_events()
    events[-1]["host_replies"] = [notice]  # the host appended its own interrupted-turn row after T03
    assert make(tmp_path / "h", rows=rows, events=events)["verdict"] == "PASS"


def test_f3_continue_rows_are_position_bound(tmp_path):
    events = clean_events(3)
    events[2]["content"] = events[2]["persist"] = events[3]["held"] = "continue"
    rows = [("user", U.format(1, 1)), ("assistant", R.format(1, 1)), ("user", "continue"), ("assistant", R.format(2, 2)),
            ("user", U.format(3, 3)), ("assistant", R.format(3, 3))]
    assert make(tmp_path, rows=rows, events=events)["verdict"] == "PASS"
    moved = [rows[0], rows[1], rows[3], rows[2], rows[4], rows[5]]  # same multiset, continue after the wrong reply
    out = make(tmp_path / "m", rows=moved, events=events)
    assert "continue" in out["failed_bars"]["B1"] and "B2" not in out["failed_bars"]


def test_f4_rows_are_scored_per_session_lineage(tmp_path):
    k = "[K01] user turn 1: alpha beta end."
    events = clean_events(2) + [
        {"phase": "A", "turn": 1, "event": "user_sent", "tag": "K01", "session_prefix": "K", "content": k, "persist": k},
        {"phase": "A", "turn": 1, "event": "turn_end", "tag": "K01", "session_prefix": "K", "held": k,
         "reply": "reply to K01: noted item 1.", "user_tags": {"K01": 1}, "session": "cron_job_01"}]
    rows = clean_rows(2) + [("user", k), ("assistant", "reply to K01: noted item 1.")]
    right = ["S0", "S0", "child", "child", "cron_job_01", "cron_job_01"]  # turn 2 after a rotation to "child"
    ok = make(tmp_path, rows=rows, events=events, sids=right, parents={"S0": None, "child": "S0", "cron_job_01": None})
    assert ok["verdict"] == "PASS", ok["failed_bars"]
    swapped = ["S0", "S0", "cron_job_01", "cron_job_01", "child", "child"]  # same global multiset, lineages swapped
    out = make(tmp_path / "s", rows=rows, events=events, sids=swapped, parents={"S0": None, "child": "S0", "cron_job_01": None})
    assert {"B1", "B2"} <= set(out["failed_bars"])
    assert out["numbers"]["B2"]["per_session"] == {"chat": "FAIL", "cron_job_01": "FAIL"}


def test_dupes_counts_a_multi_origin_burst_after_compaction(tmp_path):
    db = tmp_path / "d.db"
    con = sqlite3.connect(db)
    con.execute("create table messages (store_id integer primary key, session_id text, role text, content text,"
                " tool_calls text, tool_call_id text, ingested_at real)")
    con.execute("create table summary_nodes (created_at real)")
    rows = [("user", "a", 1.0), ("assistant", "b", 5.0), ("user", "a", 20.0), ("assistant", "b", 20.1)]
    for i, (role, content, ts) in enumerate(rows, 1):
        con.execute("insert into messages values (?,?,?,?,?,?,?)", (i, "S", role, content, None, None, ts))
    con.execute("insert into summary_nodes values (10.0)")
    con.commit()
    con.close()
    assert dupes.count(db)["replayed_rows_after_compaction"] == 2


def test_registry_is_valid_and_selectable():
    everything = cells.select("all")
    assert len({c["id"] for c in everything}) == len(everything)
    assert [c["id"] for c in cells.select("baseline/*")] == ["baseline/in-place/acp", "baseline/rotation/acp"]
    json.dumps(everything)
    with pytest.raises(ValueError):
        cells.select("no-such-cell/*")
    targeted = {t for c in everything for t in c["targets"]}
    assert all(issue in targeted or cap for issue, (_b, cap) in cells.ISSUES.items())


def test_hosts_loader_refuses_the_live_hermes_dir(tmp_path):
    live, src = tmp_path / ".hermes", tmp_path / "src"
    (live / "hermes-agent").mkdir(parents=True)
    src.mkdir()
    good = {"python": str(tmp_path / "py"), "src": str(src), "sha": "abc"}
    path = tmp_path / "hosts.json"
    path.write_text(json.dumps({"hosts": {"ok": good, "live": {**good, "hermes_home": str(live / "profile")}}}))
    assert list(hosts.load(path, ["ok"], hermes_dir=live)) == ["ok"]
    with pytest.raises(ValueError, match="under the live"):
        hosts.load(path, ["live"], hermes_dir=live)
    path.write_text(json.dumps({"hosts": {"x": {**good, "src": str(live / "hermes-agent")}}}))
    with pytest.raises(ValueError, match="under the live"):
        hosts.load(path, hermes_dir=live)


def test_f5_symlinked_live_path_and_tree_manifest(tmp_path):
    live, outside = tmp_path / ".hermes", tmp_path / "outside"
    (live / "hermes-agent").mkdir(parents=True)
    outside.mkdir()
    (live / "link").symlink_to(outside)  # lexically under the live dir, resolves outside it
    (tmp_path / "back").symlink_to(live / "hermes-agent")  # lexically outside, resolves into it
    assert hosts.under_real_hermes(live / "link", hermes_dir=live)
    assert hosts.under_real_hermes(tmp_path / "back", hermes_dir=live)
    assert not hosts.under_real_hermes(outside, hermes_dir=live)
    repo, src = tmp_path / "repo", tmp_path / "export"
    repo.mkdir()
    (repo / "a.py").write_text("A = 1\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "x"], check=True)
    sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    (src / "tree").mkdir(parents=True)
    (src / "tree" / "a.py").write_text("A = 1\n")
    hosts.write_manifest(src / "tree", sha, repo)
    assert hosts.verify("x", {"src": str(src / "tree"), "sha": sha})["method"] == "tree-manifest"
    (src / "tree" / "a.py").write_text("A = 2\n")
    with pytest.raises(ValueError, match="tree hash"):
        hosts.verify("x", {"src": str(src / "tree"), "sha": sha})


def test_f6_final_check_never_falls_back_from_the_selected_api(monkeypatch):
    fallback = []

    def compress_now(*_a, **_k):
        raise ImportError("failure inside the selected path")
    for name, attrs in {"agent": {}, "agent.conversation_compression_manual": {"compress_now": compress_now,
                                                                            "parse_compress_args": lambda s: s},
                        "agent.conversation_compression": {"finalize_context_engine_compression_notification": print},
                        "acp_adapter": {}, "acp_adapter.commands": {"_estimate_tokens": lambda *a: 1}}.items():
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))
    agent = types.SimpleNamespace(context_compressor=types.SimpleNamespace(_last_compression_status="compacted"),
                                  _compress_context=lambda *a, **k: fallback.append(1) or (a[0], None))
    out = probe.final_check(agent, [], probe.io.StringIO())
    assert out["outcome"] == "failed" and "failure inside" in out["exception"]
    assert out["entry"] == "compress_now" and not fallback and len(out["attempts"]) == 1


def test_plugin_identity_is_read_from_the_tree(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "plugin.yaml").write_text("name: hermes-lcm\n")
    (old / "engine.py").write_text("class E:\n    @property\n    def name(self) -> str:\n        return \"lcm\"\n")
    (new / "plugin.yaml").write_text("name: hermes-lcm-x\n")
    (new / "plugin_identity.py").write_text('ENGINE_NAME = "lcm-x"\n')
    assert plugin_tree.identity(old) == {"dir": "hermes-lcm", "enabled": "hermes-lcm", "engine": "lcm",
                                         "module": "hermes_plugins.hermes_lcm"}
    assert plugin_tree.identity(new)["engine"] == "lcm-x"
