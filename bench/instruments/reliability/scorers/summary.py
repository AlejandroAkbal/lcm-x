"""Summary-node report and the compaction growth bar (from ``summary_nodes_report.py`` + ``compaction_ledger.py``).

The gauntlet versions attribute nodes to turns by wall clock over a live run's raw.log; here the probe
records every compress() call as a transcript ``compaction`` event with the depth-0, message-sourced node
count read right after a published pass, so the ledger needs no clock.
"""
from __future__ import annotations

import sqlite3


def nodes_report(db_path) -> dict:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = db.execute("select depth, source_type, count(*) from summary_nodes group by depth, source_type").fetchall()
    finally:
        db.close()
    return {"by_depth_source": {f"d{d}/{s}": n for d, s, n in rows}, "total": sum(n for *_x, n in rows)}


def growth(events: list[dict], logged_publications: int, min_compactions: int) -> dict:
    """The per-published-pass ledger. An LCM pass proves itself by depth-0 message-node growth; a host-native
    pass (LCM writes no node) proves itself only by a host ``commit_status: committed`` telemetry line in the
    same turn (``host_commits`` on that turn's turn_end/crash events). Published passes >= min; logged
    ``LCM compaction #`` lines == LCM passes (the final forced one included: it logs too). The final forced
    compaction is B4 evidence, not a published pass here."""
    passes = [e for e in events if e.get("event") == "compaction" and not e.get("final")]
    published = [e for e in passes if e.get("compression_status") in ("compacted", "host_native")]
    lcm = [e for e in passes if e.get("compression_status") == "compacted"]  # native passes add no LCM nodes
    lcm_all = [e for e in events if e.get("event") == "compaction" and e.get("compression_status") == "compacted"]
    counts = [e.get("depth0_nodes") for e in lcm]
    stalls = [i for i in range(1, len(counts)) if counts[i] is None or counts[i - 1] is None or counts[i] <= counts[i - 1]]
    if counts and (counts[0] is None or counts[0] < 1):
        stalls.insert(0, 0)
    commits, natives = {}, {}
    for e in events:
        key = (e.get("phase"), e.get("session_prefix", "T"), e.get("turn"))
        if e.get("event") in ("turn_end", "crash") and e.get("host_commits") is not None:
            commits[key] = commits.get(key, 0) + e["host_commits"]
        if e.get("event") == "compaction" and e.get("compression_status") == "host_native" and not e.get("final"):
            natives[key] = natives.get(key, 0) + 1
    unproven = sorted(f"{k[0]}:{k[1]}{k[2]}" for k, n in natives.items() if commits.get(k, 0) < n)
    return {"published": len(published), "lcm_published": len(lcm), "logged": logged_publications,
            "min_compactions": min_compactions, "depth0_sequence": counts, "non_growing_passes": stalls,
            "native_passes_without_host_commit": unproven,
            "ok": not stalls and not unproven and len(published) >= min_compactions and logged_publications == len(lcm_all)}
