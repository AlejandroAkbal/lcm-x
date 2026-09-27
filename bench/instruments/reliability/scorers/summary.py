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
    """Depth-0 message nodes grow after every LCM pass; published passes (LCM or host-native) >= min;
    logged ``LCM compaction #`` lines == LCM passes in the ledger."""
    passes = [e for e in events if e.get("event") == "compaction"]
    published = [e for e in passes if e.get("compression_status") in ("compacted", "host_native")]
    lcm = [e for e in passes if e.get("compression_status") == "compacted"]  # native passes add no LCM nodes
    counts = [e.get("depth0_nodes") for e in lcm]
    stalls = [i for i in range(1, len(counts)) if counts[i] is None or counts[i - 1] is None or counts[i] <= counts[i - 1]]
    if counts and (counts[0] is None or counts[0] < 1):
        stalls.insert(0, 0)
    return {"published": len(published), "lcm_published": len(lcm), "logged": logged_publications,
            "min_compactions": min_compactions, "depth0_sequence": counts, "non_growing_passes": stalls,
            "ok": not stalls and len(published) >= min_compactions and logged_publications == len(lcm)}
