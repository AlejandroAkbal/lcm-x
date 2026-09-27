"""One phase of one reliability cell, run by the HOST python with cwd = the host source tree.

Imports only the stdlib and host modules; the plugin under test loads through the Hermes plugin
loader from ``HERMES_HOME/plugins/``. The provider client is a MagicMock scripted per turn, the
host aux LLM and the LCM summariser are stubbed, and sockets are blocked. Generalises ``_PROBE`` /
``_CRASH_PROBE`` in tests/test_real_turn_loop_acp_override.py. Every host shape it emulates is
cited as ``file:line`` in the current host tree (``citations`` in phase-<X>.json).

Last stdout line: ``{"exit": done|crash|clean_exit|tip_switch|unsupported, "next_turn": N, ...}``.
"""
import argparse
import hashlib
import io
import json
import logging
import os
import pwd
import re
import socket
import sqlite3
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ANCHORS = {  # shape -> (host file, text on the cited line)
    "acp_persist": ("acp_adapter/server.py", "persist_user_message=user_text"),
    "acp_restore": ("acp_adapter/session.py", "get_messages_as_conversation(session_id, repair_alternation=True)"),
    "acp_cancel": ("acp_adapter/server.py", "request_hard_interrupt(state.agent)"),
    "acp_retry": ("acp_adapter/server.py", "def _attach_interrupted_prompt"),
    "acp_compress": ("acp_adapter/commands.py", "def _cmd_compress"),
    "gateway_transcript": ("gateway/session_transcript.py", "def load_transcript"),
    "gateway_user_text": ("gateway/run_turn.py", "def _hmwa_apply_message_timestamp"),
    "gateway_run": ("gateway/run_turn_runner.py", "return agent.run_conversation(api_message"),
    "user_merge": ("agent/agent_runtime_helpers.py", "def _merge_consecutive_users"),
    "orphan_drop": ("agent/agent_runtime_helpers.py", "def _drop_stray_tool_results"),
    "rotation_start": ("agent/conversation_compression.py", "def _notify_context_engine_compression_complete"),
    "rotation_end": ("agent/conversation_compression.py", "agent.commit_memory_session(messages)"),
    "session_transition": ("run_agent.py", "def _transition_context_engine_session"),
    "summary_aborted": ("agent/context_compressor.py", '"summary_generation_aborted"'),
    "cron_agent": ("cron/scheduler.py", 'platform="cron"'),
    "cron_close": ("cron/scheduler.py", "agent.close()"),
}
LOG_COUNTS = {
    "publication_invariant_conflict": "publication_invariant_conflict",
    "commit_logged": "as a compaction commit",
    "summary_generation_aborted": "summary_generation_aborted",
    "native_unusable": "native recovery did not produce a usable summary",
    "orphan_log": "orphaned tool result",
    "resident_engine_conflict": "resident_engine_conflict",
    "skipped_ingest_resident_conflict": "skipped ingest: stable engine use ended with resident_engine_conflict",
    "recorded_replaced": "LCM recorded host-replaced rows",
}
FILLER = "alpha beta gamma delta "


def cite(key):
    rel, needle = ANCHORS[key]
    try:
        lines = Path(rel).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    return next((f"{rel}:{i}" for i, line in enumerate(lines, 1) if needle in line), None)


def refusal(cell_dir):
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    for name in ("HERMES_HOME", "HOME"):
        value = os.environ.get(name)
        path = Path(value).resolve() if value else None
        if path is None or path == real_home or path == real_home / ".hermes" or real_home / ".hermes" in path.parents:
            return f"{name}={value!r} is unset or resolves to the real home or under ~/.hermes"
    resolved = Path(cell_dir).resolve()
    if any(str(resolved) == p or str(resolved).startswith(p + "/") for p in ("/tmp", "/private/tmp")):
        return f"--cell-dir {resolved} is under /tmp"
    return None


def user_text(cell, prefix, t):
    ut = cell["user_text"]
    if t in ut.get("continue_turns", []):
        return "continue"
    n = int(ut.get("identical_turns", {}).get(str(t), t))
    sep = "\n\n" if ut.get("separator_turns") == "all" or n in ut.get("separator_turns", []) else ""
    body = (FILLER * ut["repeat"]).rstrip()
    if sep:  # >=64 paragraph separators inside the prompt (#545)
        words = body.split(" ")
        step = max(1, len(words) // 70)
        body = sep.join(" ".join(words[i:i + step]) for i in range(0, len(words), step))
    text = f"[{prefix}{n:02d}] user turn {n}: {body} end."
    if t in ut.get("edge_ws_turns", []):
        text = "  " + text + " \n"
    return text + ("\n" if ut.get("trailing_ws") else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--start-turn", type=int, default=1)
    ap.add_argument("--cell-dir", required=True)
    a = ap.parse_args()
    if (why := refusal(a.cell_dir)) is not None:
        print(json.dumps({"exit": "refused", "reason": why}), flush=True)
        sys.exit(3)
    cell = json.loads(Path(a.cell).read_text())
    cell_dir, phase, first = Path(a.cell_dir), a.phase, a.start_turn
    out = {"phase": phase, "start_turn": first, "citations": {k: cite(k) for k in ANCHORS}}
    tfile = open(cell_dir / "transcript.jsonl", "a", encoding="utf-8")
    buf = io.StringIO()
    counters = {"compacted_turns": [], "lcm_tool_calls": 0, "orphan_drops": 0, "native_max": 0, "failed": []}

    def event(**ev):
        tfile.write(json.dumps({"phase": phase, **ev}) + "\n")
        tfile.flush()
        os.fsync(tfile.fileno())

    def finish(exit_kind, **extra):
        log = buf.getvalue()
        out.update(exit=exit_kind, **extra, counters=counters, compactions_logged=len(re.findall(r"LCM compaction #\d+", log)),
                   log_counts={k: log.count(v) for k, v in LOG_COUNTS.items()}, session_count=session_count())
        (cell_dir / f"phase-{phase}.json").write_text(json.dumps(out, indent=1, default=str))
        (cell_dir / f"probe-{phase}.hermes.log").write_text("\n".join(
            line for line in log.splitlines() if "LCM" in line or "WARNING" in line or "ERROR" in line
            or "compress" in line.lower() or "orphan" in line)[-2_000_000:])
        print(json.dumps({"exit": exit_kind, **extra}), flush=True)
        tfile.close()
        if exit_kind in ("crash", "clean_exit", "tip_switch"):  # between turns for the latter two, as _CRASH_PROBE
            os._exit(0)  # the host process dies here: no atexit, no flush, no engine shutdown

    faults = {f["kind"]: f for f in cell.get("faults", [])}
    fired_path = cell_dir / "faults-fired.jsonl"
    fired = {json.loads(x)["kind"] for x in fired_path.read_text().splitlines()} if fired_path.exists() else set()

    def fire(kind, turn, **extra):
        with open(fired_path, "a") as fh:
            fh.write(json.dumps({"kind": kind, "phase": phase, "turn": turn}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        fired.add(kind)
        out["fired"] = out.get("fired", []) + [kind]
        event(turn=turn, event="crash" if kind.startswith("crash") else kind, fault=kind, **extra)

    needed = {"acp": ["acp_persist"] + (["acp_restore"] if phase != "A" else []),
              "gateway": ["gateway_transcript", "gateway_user_text", "gateway_run"]}[cell["transport"]]
    needed += ["acp_cancel", "acp_retry"] if "cancel_then_retry" in faults else []
    needed += ["acp_compress"] if cell.get("final_compaction_check", True) else []
    needed += ["cron_agent", "cron_close"] if cell.get("cron_every") else []
    if missing := [k for k in needed if not out["citations"][k]]:
        finish("unsupported", reason=f"host shape not citable at this sha: {missing}")
        return

    def _blocked(*_a, **_k):
        raise OSError("network blocked by probe")
    socket.socket.connect = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
    from hermes_cli import plugins as P
    P.discover_plugins(force=True)
    from hermes_state import SessionDB
    from run_agent import AIAgent
    import agent.context_compressor as host_cc
    home = Path(os.environ["HERMES_HOME"])
    files = cell_dir / "files"
    files.mkdir(exist_ok=True)
    (files / "small.txt").write_text("small deterministic file\n")
    (files / "big.txt").write_text("".join(f"line {i:05d}: " + FILLER * 8 + "\n" for i in range(cell.get("big_lines", 400))))

    def aux_llm(**kwargs):
        text = "## Goal\nstub\n## Progress\nstub" if kwargs.get("task") == "compression" else "Title"
        msg = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="aux", usage=None)
    import agent.title_generator as host_tg
    host_tg.call_llm = aux_llm
    host_cc.call_llm = aux_llm
    n_summ = {"c": 0}

    def summarise(*args, **kw):  # tag-preserving; "U05" never collides with a "[T05]" user tag
        n_summ["c"] += 1
        text = kw.get("text") if "text" in kw else (args[0] if args else "")
        tags = sorted(set(re.findall(r"\[([A-Z]\d\d)\] user", text or "")))
        return f"Stub summary #{n_summ['c']} covers " + " ".join("U" + x for x in tags) + ".\nExpand for details about: stub", 1

    window = int(cell["window"])
    cur = {"turn": 0, "step": 0, "native": 0, "sess": "S0", "ended": None}

    def build(session_id, platform):
        with patch("agent.process_bootstrap.OpenAI"):
            ag = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="test/model",
                         quiet_mode=True, session_db=SessionDB(db_path=home / "state.db"), session_id=session_id,
                         skip_context_files=True, skip_memory=True, platform=platform,
                         enabled_toolsets=cell.get("toolsets", ["todo", "context_engine", "file"]))
        ag.client, ag.tool_delay, ag.save_trajectories = MagicMock(), 0, False
        ag.compression_in_place = bool(cell["in_place"])
        ag._compression_feasibility_checked = True
        before = getattr(ag.context_compressor, "context_length", None)
        ag.context_compressor.update_model("test/model", window, base_url="https://openrouter.ai/api/v1",
                                           api_key="k", provider="openrouter")
        out["context_length"] = {"host_resolved": before, "set_via_update_model": window}
        return ag

    sdb_read = SessionDB(db_path=home / "state.db")
    sid = "S0"
    if phase != "A" and cell["transport"] == "gateway":  # a restarted gateway binds the durable tip
        sid = sdb_read.get_compression_tip("S0") or "S0"
    agent = build(sid, "acp")
    engine = agent.context_compressor
    out["engine"] = getattr(engine, "name", None)
    if out["engine"] != cell["plugin"]["engine"]:
        finish("error", reason=f"engine {out['engine']!r} is not the plugin under test")
        return
    for name, module in list(sys.modules.items()):
        if name.startswith("hermes_plugins.") and hasattr(module, "summarize_with_escalation"):
            module.summarize_with_escalation = summarise
    lcm_db = home / "lcm.db"

    def depth0():
        try:
            con = sqlite3.connect(f"file:{lcm_db}?mode=ro", uri=True)
            try:
                return con.execute("SELECT COUNT(*) FROM summary_nodes WHERE depth = 0 AND source_type = 'messages'").fetchone()[0]
            finally:
                con.close()
        except sqlite3.Error:
            return None

    etype = type(engine)
    orig_compress, orig_tool = etype.compress, etype.handle_tool_call
    orig_start, orig_end = etype.on_session_start, getattr(etype, "on_session_end", None)

    def traced_compress(self, messages, *args, **kwargs):
        result = orig_compress(self, messages, *args, **kwargs)
        status = getattr(self, "_last_compression_status", None)
        if status == "compacted":
            counters["compacted_turns"].append(cur["turn"])
        event(turn=cur["turn"], event="compaction", session=getattr(self, "_session_id", None),
              compression_status=status, noop_reason=getattr(self, "_last_compression_noop_reason", None),
              depth0_nodes=depth0() if status == "compacted" else None)
        return result
    etype.compress = traced_compress

    def traced_tool(self, name, args, **kwargs):
        counters["lcm_tool_calls"] += 1
        f = faults.get("crash_mid_tool_call")
        if f and "crash_mid_tool_call" not in fired and cur["turn"] == f["turn"]:
            fire("crash_mid_tool_call", cur["turn"], tool=name)
            finish("crash", next_turn=cur["turn"] + 1, turn=cur["turn"])
        return orig_tool(self, name, args, **kwargs)
    etype.handle_tool_call = traced_tool

    def traced_start(self, session_id, *args, **kwargs):
        rotation = kwargs.get("boundary_reason") == "compression"
        if rotation and cur["ended"] and "crash_between_session_end_and_start" in faults \
                and "crash_between_session_end_and_start" not in fired:
            fire("crash_between_session_end_and_start", cur["turn"], old=cur["ended"], new=session_id)
            finish("crash", next_turn=cur["turn"] + 1, turn=cur["turn"])
        cur["ended"] = None
        result = orig_start(self, session_id, *args, **kwargs)
        if rotation and "crash_after_rotation_before_child_row" in faults and \
                "crash_after_rotation_before_child_row" not in fired:
            fire("crash_after_rotation_before_child_row", cur["turn"], new=session_id)
            finish("crash", next_turn=cur["turn"] + 1, turn=cur["turn"])
        return result
    etype.on_session_start = traced_start
    if orig_end is not None:
        def traced_end(self, session_id, *args, **kwargs):
            cur["ended"] = session_id
            return orig_end(self, session_id, *args, **kwargs)
        etype.on_session_end = traced_end

    native_cls = getattr(host_cc, "ContextCompressor", None)
    if native_cls is not None and native_cls is not etype:
        orig_native = native_cls.compress

        def counted_native(self, *args, **kwargs):
            cur["native"] += 1
            return orig_native(self, *args, **kwargs)
        native_cls.compress = counted_native
    import agent.agent_runtime_helpers as helpers
    passes = getattr(helpers, "_SEQUENCE_REPAIR_PASSES", None)
    drop = getattr(helpers, "_drop_stray_tool_results", None)
    if passes and drop in passes:
        def counted_drop(messages):
            kept, n = drop(messages)
            counters["orphan_drops"] += n
            return kept, n
        helpers._SEQUENCE_REPAIR_PASSES = tuple(counted_drop if p is drop else p for p in passes)
    else:
        out["orphan_hook"] = "unavailable at this host sha; B6 orphan count falls back to the log"

    pf = faults.get("publication_failure")
    if pf:
        mod = sys.modules.get(cell["plugin"]["module"] + ".lifecycle_state")
        lifecycle = getattr(engine, "_lifecycle", None)
        err = getattr(mod, "LifecyclePublicationConflictError", None)
        if lifecycle is None or err is None:
            finish("unsupported", reason="plugin tree has no lifecycle publication stage to inject into")
            return
        orig_stage, stages = lifecycle.stage_compaction_publication, {"n": 0}

        def inject(conn, conversation_id, session_id, *args, **kwargs):
            stages["n"] += 1
            if (pf["where"] == "rotation_child" and session_id != "S0") or pf["where"] == f"pass_{stages['n']}":
                if pf["kind"] not in fired:
                    fire(pf["kind"], cur["turn"], where=pf["where"])
                raise err(f"injected publication failure ({pf['where']})")
            return orig_stage(conn, conversation_id, session_id, *args, **kwargs)
        lifecycle.stage_compaction_publication = inject

    asst = cell["assistant"]
    plan = {}
    for group in cell.get("tool_plan", []):  # "restart": the first turn of every phase after A (the merge turn)
        for t in ([first] if phase != "A" else []) if group["turns"] == "restart" else group["turns"]:
            plan.setdefault(t, []).append(group["calls"])

    def response(content, prompt_tokens, calls=None, t=0):
        tcs = [SimpleNamespace(id=f"call_{t:02d}_{k}", type="function", function=SimpleNamespace(
            name=c["name"], arguments=json.dumps(c.get("args", {})).replace("{files}", str(files))))
            for k, c in enumerate(calls or [])] or None
        msg = SimpleNamespace(content="" if tcs else content, tool_calls=tcs)
        r = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if tcs else "stop")],
                            model="test/model")
        r.usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=20, total_tokens=prompt_tokens + 20)
        return r

    def reply_text(prefix, t):
        if asst.get("mode") == "repeat-identical" and t in asst.get("repeat_turns", []):
            return "noted, the same as before."
        return f"reply to {prefix}{t:02d}: noted item {t}."

    def scripted(ag, prefix, t, est, cancel=False):
        def provider(*_a, **kw):
            if phase == "A" and prefix == "T" and "crash_after_compaction_before_reply" in faults and \
                    "crash_after_compaction_before_reply" not in fired and t in counters["compacted_turns"]:
                fire("crash_after_compaction_before_reply", t)
                finish("crash", next_turn=t + 1, turn=t)
            step, cur["step"] = cur["step"], cur["step"] + 1
            sent = sum(len(str(m.get("content") or "")) for m in kw.get("messages") or []) // 4 + 800
            usage = int((sent if asst.get("real_usage") else est) * float(asst.get("usage_scale", 1.0)))
            if cancel and step == 0:  # the ACP cancel lands while the provider call is in flight
                from agent.interrupt_compat import request_hard_interrupt
                request_hard_interrupt(ag)
                event(turn=t, event="cancel")
                time.sleep(float(cell.get("cancel_wait", 2.0)))
            groups = plan.get(t, []) if prefix == "T" else []
            if step < len(groups):
                for c in groups[step]:
                    event(turn=t, event="tool_call", name=c["name"], session_prefix=prefix)
                return response("", usage, groups[step], t)
            return response(reply_text(prefix, t), usage)
        ag.client.chat.completions.create.side_effect = provider

    def held_after(result, text, prefix, t):
        """The user row the host holds for this turn (after its persist override / merge) and what follows it."""
        msgs = result.get("messages") if isinstance(result.get("messages"), list) else []
        tag = f"[{prefix}{int(cell['user_text'].get('identical_turns', {}).get(str(t), t)):02d}]"
        idx = [i for i, m in enumerate(msgs) if m.get("role") == "user" and isinstance(m.get("content"), str)
               and (text == "continue" or tag in m["content"])]
        tags = {}
        for m in msgs:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                for x in set(re.findall(r"\[([A-Z]\d\d)\] user turn", m["content"])):
                    tags[x] = tags.get(x, 0) + 1
        if not idx:
            return None, False, [], tags
        after = msgs[idx[-1] + 1:]
        reply = reply_text(prefix, t)
        return (msgs[idx[-1]]["content"], any(m.get("role") == "assistant" and m.get("content") == reply for m in after),
                [m for m in after if m.get("role") == "tool"], tags)

    def run_turn(ag, prefix, t, history, kind="normal", persist_strip=None, task_id="S0"):
        text = user_text(cell, prefix, t)
        if kind == "retry":  # acp_adapter/server.py: plain text after a cancel re-attaches the cancelled prompt
            from acp_adapter.server import _attach_interrupted_prompt
            text = _attach_interrupted_prompt(text.strip(), text.strip())
        persist = text.strip() if (persist_strip if persist_strip is not None else cell["transport"] == "acp") else text
        est = sum(len(str(m.get("content") or "")) for m in history) // 4 + len(text) // 4 + 800
        cur.update(turn=t, step=0, native=0)
        event(turn=t, event="user_sent" if kind != "retry" else "retry", tag=f"{prefix}{t:02d}", role="user",
              session_prefix=prefix, content=text, persist=persist,
              content_sha256=hashlib.sha256(text.encode()).hexdigest())
        scripted(ag, prefix, t, est, cancel=kind == "cancel")
        result = ag.run_conversation(user_message=text, conversation_history=history, task_id=task_id,
                                     persist_user_message=persist)
        held, reply_held, tools, user_tags = held_after(result, text, prefix, t)
        failed = bool(result.get("failed")) or not result.get("completed", True)
        if failed and kind != "cancel":
            counters["failed"].append(f"{prefix}{t:02d}")
        counters["native_max"] = max(counters["native_max"], cur["native"])
        for m in tools:
            event(turn=t, event="tool_result", role="tool", size=len(str(m.get("content") or "")))
        event(turn=t, event="turn_end", tag=f"{prefix}{t:02d}", session_prefix=prefix, held=held, kind=kind,
              reply=reply_text(prefix, t) if reply_held else None, failed=failed, native_attempts=cur["native"],
              interrupted=bool(result.get("interrupted")), session=ag.session_id, user_tags=user_tags)
        return result

    def cron_run(k):  # cron/scheduler.py: a fresh platform="cron" agent per fire, no history, closed after
        cron_sid = f"cron_job_{k:02d}"
        ag = build(cron_sid, "cron")
        box = {}

        def work():
            try:
                box["r"] = run_turn(ag, "K", k, [], persist_strip=False, task_id=cron_sid)
            except Exception as exc:  # recorded, never raised past the probe
                box["e"] = repr(exc)
        th = threading.Thread(target=work, name=f"cron-{k}")
        th.start()
        th.join()
        with_db = getattr(ag, "_session_db", None)
        if with_db is not None and hasattr(with_db, "end_session"):
            with_db.end_session(cron_sid, "cron_complete")
        ag.close()
        if "e" in box:
            counters["failed"].append(f"K{k:02d}")

    history = []
    if phase != "A":  # ACP _restore reads the stable ACP id; a gateway reads the durable tip (load_transcript)
        history = sdb_read.get_messages_as_conversation(sid, repair_alternation=True)
    turns, cancel = int(cell["turns"]), faults.get("cancel_then_retry")
    for t in range(first, turns + 1):
        f = faults.get("clean_exit_before_turn")
        if f and phase != "A" and t == f.get("turn", first + f.get("after_restart", 0)) and t != first \
                and "clean_exit_before_turn" not in fired:
            fire("clean_exit_before_turn", t)
            finish("clean_exit", next_turn=t)
        if cell["transport"] == "gateway" and not (phase == "A" and t == first):  # load_transcript, every turn
            tip = sdb_read.get_compression_tip(agent.session_id) or agent.session_id
            if tip != agent.session_id:  # a real gateway serves the next message from the tip
                finish("tip_switch", next_turn=t, tip=tip)
            history = sdb_read.get_messages_as_conversation(tip, repair_alternation=True)
        if cancel and t == cancel["turn"] and "cancel_then_retry" not in fired:
            fire("cancel_then_retry", t)
            result = run_turn(agent, "T", t, history, kind="cancel")
            history = result["messages"] if isinstance(result.get("messages"), list) else history
            result = run_turn(agent, "T", t, history, kind="retry")
        else:
            result = run_turn(agent, "T", t, history)
        if isinstance(result.get("messages"), list):
            history = result["messages"]
        if cell["transport"] == "gateway" and agent.session_id != sid:  # spec: restart the gateway at the tip, never
            finish("tip_switch", next_turn=t + 1, tip=agent.session_id)  # switch the resident engine in process
        if cell.get("cron_every") and t % int(cell["cron_every"]) == 0:
            cron_run(t // int(cell["cron_every"]))
    if cell.get("final_compaction_check", True):
        out["final_check"] = final_check(agent, history, buf)
    finish("done", next_turn=None)


def session_count():
    try:
        con = sqlite3.connect(f"file:{Path(os.environ['HERMES_HOME']) / 'state.db'}?mode=ro", uri=True)
        try:
            return con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error:
        return None


def final_check(agent, history, buf):
    """Force one compaction the way the host's ACP ``/compress`` does (acp_adapter/commands.py)."""
    before = buf.getvalue().count("publication_invariant_conflict")
    engine, rec = agent.context_compressor, {}
    saved = getattr(agent, "_session_db", None)
    try:
        agent._session_db = None  # "Stable ACP session id: suppress _compress_context's SQLite session split."
        system = getattr(agent, "_cached_system_prompt", "") or ""
        try:
            from agent.conversation_compression_manual import compress_now, parse_compress_args
            from agent.conversation_compression import finalize_context_engine_compression_notification
            res = compress_now(agent, history, parse_compress_args(""), system_message=system, task_id="S0")
            rec["host_status"] = res.status
            if res.status == "compressed":
                finalize_context_engine_compression_notification(agent, committed=True)
        except ImportError:  # older hosts: acp_adapter/commands.py _cmd_compress calls _compress_context directly
            from acp_adapter.commands import _estimate_tokens
            approx = _estimate_tokens(history, agent, system, getattr(agent, "tools", None) or None)
            agent._compress_context(list(history), system, approx_tokens=approx, task_id="S0", force=True)
            rec["host_status"] = "compressed"
    except Exception as exc:
        rec["exception"] = repr(exc)[:500]
    finally:
        agent._session_db = saved
    rec["engine_status"] = getattr(engine, "_last_compression_status", None)
    rec["noop_reason"] = getattr(engine, "_last_compression_noop_reason", None)
    rec["conflicts"] = buf.getvalue().count("publication_invariant_conflict") - before
    ok = ("compacted", "host_native") if os.environ.get("LCM_NATIVE_RECOVERY") == "true" else ("compacted",)
    rec["published"] = rec["engine_status"] in ok and not rec["conflicts"] and "exception" not in rec
    return rec


if __name__ == "__main__":
    main()
