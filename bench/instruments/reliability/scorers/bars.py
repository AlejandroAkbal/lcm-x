"""Bars B1-B7 over one finished cell: its DB copy (db/lcm.db), transcript.jsonl and phase-*.json.

The expected transcript is what the host HELD for each attempt (the probe records the user row the host
kept after its persist override and consecutive-user merge), so the bars compare LCM's store with the
host, never with LCM itself. An attempt with no stored reply whose prompt the next attempt of the same
session folded into a composite row (and no other held row still carries its tag) is superseded by that
composite: that is the host merge (agent/agent_runtime_helpers.py ``_merge_consecutive_users``).
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from . import multiset, summary, tool_groups

ALL_BARS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7")


def load(cell_dir: Path):
    events = [json.loads(x) for x in (cell_dir / "transcript.jsonl").read_text().splitlines() if x.strip()]
    phases = [json.loads(p.read_text()) for p in sorted(cell_dir.glob("phase-*.json"), key=lambda p: (len(p.name), p.name))]
    return events, phases


def attempts(events: list[dict]) -> list[dict]:
    out, open_ = [], {}
    for e in events:
        if e["event"] in ("user_sent", "retry"):
            a = {"tag": e["tag"], "prefix": e.get("session_prefix", "T"), "content": e["content"], "persist": e["persist"],
                 "held": None, "reply": None, "user_tags": {}, "ended": False}
            out.append(a)
            open_[e["tag"]] = a
        elif e["event"] == "turn_end" and e["tag"] in open_:
            a = open_.pop(e["tag"])
            a.update(held=e.get("held"), reply=e.get("reply"), user_tags=e.get("user_tags") or {}, ended=True)
    return out


def expected_items(atts: list[dict]) -> list[tuple[str, str]]:
    items = []
    for i, a in enumerate(atts):
        text = a["held"] if a["held"] is not None else a["persist"]
        nxt = next((b for b in atts[i + 1:] if b["prefix"] == a["prefix"]), None)
        cand = multiset.norm(text)
        folded = (nxt is not None and not a["reply"] and nxt["held"] is not None and cand
                  and cand in multiset.norm(nxt["held"]) and cand != multiset.norm(nxt["held"])
                  and nxt["user_tags"].get(a["tag"], 1) <= 1)
        if not folded:
            items.append(("user", text))
        if a["reply"]:
            items.append(("assistant", a["reply"]))
    return items


def tag_counts(texts, pattern):
    counts = Counter()
    for text in texts:
        for tag in set(re.findall(pattern, text or "")):
            counts[tag] += 1
    return counts


def score(cell: dict, cell_dir: Path) -> dict:
    events, phases = load(cell_dir)
    db = cell_dir / "db" / "lcm.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        stored = con.execute("select store_id, session_id, role, content from messages order by store_id").fetchall()
    finally:
        con.close()
    atts = attempts(events)
    items = expected_items(atts)
    applicable = [b for b in cell.get("bars") or ALL_BARS
                  if (b != "B6" or cell.get("tool_plan")) and (b != "B7" or cell.get("native_recovery"))
                  and (b != "B5" or cell.get("min_compactions", 5) > 0)]
    failed, numbers = {}, {}

    user_pat, reply_pat = r"\[([A-Z]\d\d)\] user turn", r"reply to ([A-Z]\d\d)\b"
    want_u = tag_counts([t for r, t in items if r == "user"], user_pat)
    want_a = tag_counts([t for r, t in items if r == "assistant"], reply_pat)
    have_u = tag_counts([c for _s, _sid, r, c in stored if r == "user"], user_pat)
    have_a = tag_counts([c for _s, _sid, r, c in stored if r == "assistant"], reply_pat)
    b1 = {k: {"expected": want_u[k], "stored": have_u[k]} for k in set(want_u) | set(have_u) if want_u[k] != have_u[k]}
    b1.update({f"reply {k}": {"expected": want_a[k], "stored": have_a[k]}
               for k in set(want_a) | set(have_a) if want_a[k] != have_a[k]})
    numbers["B1"] = {"user_tags": len(want_u), "reply_tags": len(want_a), "mismatched": len(b1),
                     "per_session": {p: sum(1 for k in b1 if k.split()[-1].startswith(p)) for p in
                                     sorted({a["prefix"] for a in atts})}}
    if b1:
        failed["B1"] = dict(sorted(b1.items())[:30])
    ms = multiset.score(items, stored)
    numbers["B2"] = {k: ms[k] for k in ("expected_items", "missing_keys", "deficit_rows", "duplicated_keys",
                                        "surplus_rows", "stored_rows_not_expected")}
    if ms["verdict"] != "PASS":
        failed["B2"] = {**numbers["B2"], "missing": ms["missing"][:5], "duplicated": ms["duplicated"][:5]}
    conflicts = sum(p.get("log_counts", {}).get("publication_invariant_conflict", 0) for p in phases)
    numbers["B3"] = {"publication_invariant_conflict": conflicts}
    if conflicts:
        failed["B3"] = numbers["B3"]
    failed_turns = [t for p in phases for t in p.get("counters", {}).get("failed", [])]
    final = next((p["final_check"] for p in reversed(phases) if "final_check" in p), None)
    numbers["B4"] = {"failed_turns": failed_turns, "final_check": final}
    if failed_turns or (cell.get("final_compaction_check", True) and not (final or {}).get("published")):
        failed["B4"] = numbers["B4"]
    grow = summary.growth(events, sum(p.get("compactions_logged", 0) for p in phases), cell.get("min_compactions", 5))
    numbers["B5"] = grow
    if not grow["ok"]:
        failed["B5"] = {k: v for k, v in grow.items() if k != "depth0_sequence"} | {"depth0_tail": grow["depth0_sequence"][-8:]}
    tg = tool_groups.split_groups(db)
    hooked = all("orphan_hook" not in p for p in phases)
    orphans = sum(p.get("counters", {}).get("orphan_drops", 0) if hooked else p.get("log_counts", {}).get("orphan_log", 0)
                  for p in phases)
    numbers["B6"] = {"groups": tg["groups"], "split_groups": tg["split_groups"], "host_orphan_drops": orphans,
                     "tool_results": sum(1 for e in events if e["event"] == "tool_result")}
    if tg["split_groups"] or orphans:
        failed["B6"] = {**numbers["B6"], "splits": tg["splits"][:3]}
    native = {"native_unusable": sum(p.get("log_counts", {}).get("native_unusable", 0) for p in phases),
              "summary_generation_aborted": sum(p.get("log_counts", {}).get("summary_generation_aborted", 0) for p in phases),
              "max_native_attempts_per_turn": max((p.get("counters", {}).get("native_max", 0) for p in phases), default=0)}
    numbers["B7"] = native
    if native["native_unusable"] or native["summary_generation_aborted"] or native["max_native_attempts_per_turn"] > 1:
        failed["B7"] = native
    failed = {b: v for b, v in failed.items() if b in applicable}
    numbers["diagnostic"] = {
        "log_counts": {k: sum(p.get("log_counts", {}).get(k, 0) for p in phases)
                       for k in ("resident_engine_conflict", "skipped_ingest_resident_conflict", "recorded_replaced")},
        "phases": len(phases), "session_count": phases[-1].get("session_count") if phases else None,
        "lcm_tool_calls": sum(p.get("counters", {}).get("lcm_tool_calls", 0) for p in phases),
        "summary_nodes": summary.nodes_report(db)}
    return {"verdict": "FAIL" if failed else "PASS", "applicable_bars": applicable, "failed_bars": failed, "numbers": numbers}
