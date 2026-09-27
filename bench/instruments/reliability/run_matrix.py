"""Run a named cell set against named Hermes hosts and lcm-x refs; write results.jsonl, MATRIX.md, ISSUE-MAP.md.

    python bench/instruments/reliability/run_matrix.py --hosts eva-0.21.5,customer-0.21.2 \\
        --plugin-ref origin/main --cells 'baseline/*' --jobs 8 --out <dir>

Stdlib only. Claim class: advisory / code_green_local (see README.md).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from bench.instruments.reliability import cells as C, hosts as H, plugin_tree, report  # noqa: E402
from bench.instruments.reliability.scorers import bars  # noqa: E402

PROBE = Path(__file__).with_name("probe.py")
PHASES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def slug(cid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", cid)


def config_yaml(cell: dict, plugin: dict) -> str:
    return (f"context:\n  engine: {plugin['engine']}\n"
            f"compression:\n  enabled: true\n  threshold: 0.8\n  in_place: {'true' if cell['in_place'] else 'false'}\n"
            "  target_ratio: 0.3\n"
            + ("lcm:\n  context_threshold: 0.5\n" if cell["lcm_env"] else "")
            + f"plugins:\n  enabled: [{plugin['enabled']}]\n  disabled: []\n")


def backup(src: Path, dst: Path) -> None:
    """WAL-safe copy through the sqlite3 backup API."""
    if not src.exists():
        return
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(dst)
    try:
        s.backup(d)
        d.execute("PRAGMA journal_mode=DELETE")  # a standalone copy: readable with mode=ro, no -wal/-shm
    finally:
        d.close()
        s.close()


def unfired_reason(cell: dict, fired: set, citations: dict) -> str | None:
    missing = [f["kind"] for f in cell["faults"] if f["kind"] not in fired]
    if not missing:
        return None
    if "crash_between_session_end_and_start" in missing:
        return ("the host rotation path never called on_session_end before on_session_start(boundary_reason="
                f"'compression') (rotation notify at {citations.get('rotation_start')}; on_session_end only in "
                f"{citations.get('session_transition')})")
    return f"fault trigger(s) {missing} never fired at this host sha"


def run_cell(cell: dict, host_name: str, host: dict, plugin: dict, out: Path, timeout: int, keep: bool) -> dict:
    d = out / "cells" / host_name / plugin["sha"][:12] / slug(cell["id"])
    if d.exists():
        shutil.rmtree(d)
    home = d / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    (d / "home").mkdir()
    (d / "db").mkdir()
    (home / "plugins" / plugin["dir"]).symlink_to(plugin["tree"])
    (home / "config.yaml").write_text(config_yaml(cell, plugin))
    (d / "cell.json").write_text(json.dumps({**cell, "plugin": plugin, "host": host_name}, indent=1))
    env = {"HOME": str(d / "home"), "PATH": "/usr/bin:/bin", "HERMES_HOME": str(home), "PYTHONDONTWRITEBYTECODE": "1",
           "OPENROUTER_API_KEY": "test-key", "TMPDIR": str(d / "home"),
           "LCM_NATIVE_RECOVERY": "true" if cell["native_recovery"] else "false", **cell["lcm_env"]}
    rec = {"cell": cell["id"], "host": host_name, "host_sha": host["sha"], "plugin_ref": plugin["ref"],
           "plugin_sha": plugin["sha"], "targets": cell["targets"], "dir": str(d)}
    started, start_turn, last, phases_run = time.time(), 1, {}, []
    for phase in PHASES:
        try:
            done = subprocess.run([host["python"], str(PROBE), "--cell", str(d / "cell.json"), "--phase", phase,
                                   "--start-turn", str(start_turn), "--cell-dir", str(d)],
                                  cwd=host["src"], env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last = {"exit": "error", "reason": f"phase {phase} timed out after {timeout}s"}
            break
        (d / f"probe-{phase}.log").write_text(done.stdout[-20000:] + "\n--- stderr ---\n" + done.stderr[-20000:])
        phases_run.append(phase)
        lines = [x for x in done.stdout.splitlines() if x.startswith('{"exit"')]
        last = json.loads(lines[-1]) if lines else {"exit": "error", "reason": f"phase {phase} rc={done.returncode}: "
                                                    + (done.stderr.strip().splitlines() or ["no output"])[-1][:300]}
        if last["exit"] in ("crash", "clean_exit", "tip_switch"):
            start_turn = last["next_turn"]
            continue
        break
    else:
        last = {"exit": "error", "reason": "phase budget exhausted"}
    for name in ("lcm.db", "state.db"):
        try:
            backup(home / name, d / "db" / name)
        except sqlite3.Error as exc:
            last.setdefault("backup_error", repr(exc))
    fired_file = d / "faults-fired.jsonl"
    fired = {json.loads(x)["kind"] for x in fired_file.read_text().splitlines()} if fired_file.exists() else set()
    first_phase = d / "phase-A.json"
    citations = json.loads(first_phase.read_text()).get("citations", {}) if first_phase.exists() else {}
    rec.update(phases=phases_run, wall_s=round(time.time() - started, 1), citations=citations)
    if last["exit"] == "unsupported":
        rec.update(verdict="UNSUPPORTED", reason=last.get("reason"))
    elif last["exit"] != "done":
        rec.update(verdict="ERROR", reason=last.get("reason") or last)
    elif reason := unfired_reason(cell, fired, citations):
        rec.update(verdict="UNSUPPORTED", reason=reason)
    else:
        try:
            scored = bars.score(cell, d)
            rec.update(verdict=scored["verdict"], failed_bars=scored["failed_bars"], numbers=scored["numbers"])
        except Exception as exc:  # a scorer failure is a harness ERROR, never a PASS
            rec.update(verdict="ERROR", reason=f"scoring failed: {exc!r}")
    (d / "verdict.json").write_text(json.dumps(rec, indent=1, default=str))
    if not keep:
        shutil.rmtree(home, ignore_errors=True)
        shutil.rmtree(d / "files", ignore_errors=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts-file")
    ap.add_argument("--hosts", required=True, help="comma-separated host names, or 'all'")
    ap.add_argument("--plugin-ref", required=True, help="comma-separated lcm-x git refs")
    ap.add_argument("--lcm-repo", default=str(REPO))
    ap.add_argument("--cells", required=True, help="'all' or comma-separated globs")
    ap.add_argument("--jobs", type=int, default=min(8, max(1, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--timeout", type=int, default=900, help="per-phase timeout, seconds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-homes", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if str(out) == "/tmp" or str(out).startswith(("/tmp/", "/private/tmp")):
        ap.error("--out must not be under /tmp")
    hosts = H.load(H.hosts_file(a.hosts_file), None if a.hosts == "all" else a.hosts.split(","))
    selected = C.select(a.cells)
    plugins = [plugin_tree.export(Path(a.lcm_repo), ref.strip(), out / "plugins") for ref in a.plugin_ref.split(",")]
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"argv": sys.argv, "hosts": hosts, "plugins": plugins,
                                              "cells": [c["id"] for c in selected]}, indent=1))
    jobs = [(c, h, hosts[h], p) for p in plugins for h in hosts for c in selected]
    started, results = time.time(), []
    with ThreadPoolExecutor(max_workers=a.jobs) as pool, open(out / "results.jsonl", "w") as sink:
        futures = [pool.submit(run_cell, c, h, hd, p, out, a.timeout, a.keep_homes) for c, h, hd, p in jobs]
        for fut in as_completed(futures):
            rec = fut.result()
            results.append(rec)
            sink.write(json.dumps(rec, default=str) + "\n")
            sink.flush()
            print(f"{rec['verdict']:<11} {rec['host']:<16} {rec['plugin_ref']:<12} {rec['cell']}"
                  + (f"  {sorted(rec.get('failed_bars', {}))}" if rec.get("failed_bars") else "")
                  + (f"  {str(rec.get('reason'))[:120]}" if rec.get("reason") else ""), flush=True)
    report.write(out, results, time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
