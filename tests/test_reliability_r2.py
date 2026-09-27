"""Reliability harness R2 (bench/instruments/reliability): the fake provider, the ACP driver framing, the socket
guard, chronology and the process-cell plumbing. No Hermes: the ACP peer is a tiny stdio JSON-RPC script."""
from __future__ import annotations

import http.client
import json
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.instruments.reliability import acp_driver as AD, cells, fake_provider as FP, process_cell as PC  # noqa: E402
from bench.instruments.reliability.scorers import chronology  # noqa: E402


@pytest.fixture
def provider(tmp_path):
    calls = []

    def main(messages):
        calls.append(messages)
        last = messages[-1]["content"]
        if "fault429" in last:
            return {"status": 429}
        if "tool" in last:
            return {"tool_calls": [{"id": "call_1", "name": "lcm_grep", "arguments": '{"query": "alpha"}'}]}
        return {"content": "reply to T01: noted item 1."}
    p = FP.FakeProvider(tmp_path / "req.jsonl", main=main, usage_scale=2.0).start()
    p.calls = calls
    yield p
    p.stop()


def post(p, path, body, raw=False):
    con = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
    con.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    resp = con.getresponse()
    data = resp.read().decode()
    con.close()
    return resp.status, (data if raw else json.loads(data))


def sse_data(text):
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "[T01] user turn 1: alpha end."}]


def test_roles_route_by_model_and_usage_is_computed_from_received_messages(provider):
    status, out = post(provider, "/v1/chat/completions", {"model": "rel/main", "messages": MSGS})
    assert status == 200 and out["choices"][0]["message"]["content"] == "reply to T01: noted item 1."
    assert out["usage"]["prompt_tokens"] == ((3 + len(MSGS[1]["content"])) // 4 + 800) * 2
    nonce = "ab12" * 8
    lcm = [{"role": "system", "content": f'policy <lcm-summary nonce="{nonce}">'}, MSGS[1]]
    _, out = post(provider, "/v1/chat/completions", {"model": "rel/lcm-summary", "messages": lcm})
    text = out["choices"][0]["message"]["content"]
    assert text.startswith(f'<lcm-summary nonce="{nonce}">') and "UT01" in text and "[T01]" not in text
    assert text.rstrip().endswith("Expand for details about: stub\n</lcm-summary>")
    _, out = post(provider, "/v1/chat/completions", {"model": "aux", "messages": MSGS})
    assert out["choices"][0]["message"]["content"].startswith("## Goal")
    assert post(provider, "/v1/chat/completions", {"model": "rel/nope", "messages": MSGS})[0] == 400
    con = http.client.HTTPConnection("127.0.0.1", provider.port, timeout=10)
    con.request("GET", "/v1/models")
    assert {m["id"] for m in json.loads(con.getresponse().read())["data"]} >= {"rel/main", "rel/lcm-summary"}
    log = [json.loads(x) for x in (provider.log_path).read_text().splitlines()]
    assert [r["role"] for r in log] == ["main", "lcm-summary", "aux", "nope"] and len({r["rid"] for r in log}) == 4
    assert all(len(r["messages_sha256"]) == 64 for r in log)


def test_openai_sse_framing_text_tools_usage_and_done(provider):
    _, text = post(provider, "/v1/chat/completions", {"model": "rel/main", "messages": MSGS, "stream": True,
                                                      "stream_options": {"include_usage": True}}, raw=True)
    data = sse_data(text)
    assert data[-1] == "[DONE]"
    chunks = [json.loads(d) for d in data[:-1]]
    assert chunks[0]["object"] == "chat.completion.chunk" and chunks[0]["choices"][0]["delta"]["content"].startswith("reply")
    assert chunks[1]["choices"][0]["finish_reason"] == "stop" and chunks[2]["choices"] == [] and chunks[2]["usage"]["prompt_tokens"] > 0
    tool = [{"role": "user", "content": "please tool"}]
    _, text = post(provider, "/v1/chat/completions", {"model": "rel/main", "messages": tool, "stream": True}, raw=True)
    chunks = [json.loads(d) for d in sse_data(text)[:-1]]
    call = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["id"] == "call_1" and call["function"]["name"] == "lcm_grep" and call["index"] == 0
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls" and "usage" not in chunks[-1]


def test_anthropic_messages_json_and_sse_and_tool_result_normalisation(provider):
    body = {"model": "rel/main", "system": "sys", "max_tokens": 50, "messages": [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "R"},
                                     {"type": "text", "text": "[T01] user turn 1: alpha end."}]}]}
    _, out = post(provider, "/v1/messages", body)
    assert out["type"] == "message" and out["content"][0]["text"].startswith("reply") and out["stop_reason"] == "end_turn"
    assert provider.calls[-1][1] == {"role": "tool", "tool_call_id": "tu1", "content": "R"}
    _, text = post(provider, "/v1/messages", {**body, "stream": True}, raw=True)
    names = [line[7:] for line in text.splitlines() if line.startswith("event: ")]
    assert names == ["message_start", "content_block_start", "content_block_delta", "content_block_stop",
                     "message_delta", "message_stop"]


def test_fault_hooks_http_error_hold_until_killed_and_slow(tmp_path):
    fired = []
    replies = iter([{"status": 500}, {"hold_until_killed": True, "on_hold": lambda: fired.append("held")},
                    {"hold": 0.3, "content": "late"}])
    p = FP.FakeProvider(tmp_path / "req.jsonl", main=lambda m: next(replies), hold_cap=10).start()
    try:
        assert post(p, "/v1/chat/completions", {"model": "rel/main", "messages": MSGS})[0] == 500
        raw = json.dumps({"model": "rel/main", "messages": MSGS}).encode()
        sock = socket.create_connection(("127.0.0.1", p.port))
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(raw)).encode() + b"\r\n\r\n" + raw)
        assert p.in_flight.wait(5) and fired == ["held"]
        sock.close()  # the "host" dies while its request is held
        started = time.monotonic()
        assert post(p, "/v1/chat/completions", {"model": "rel/main", "messages": MSGS})[1]["choices"][0]["message"]["content"] == "late"
        assert time.monotonic() - started >= 0.3
        for _ in range(50):
            log = [json.loads(x) for x in p.log_path.read_text().splitlines()]
            if any(r.get("phase") == "client_closed" for r in log):
                break
            time.sleep(0.1)
        assert [r.get("fault") for r in log if r.get("phase") in (None, "held")][:2] == ["http_500", "hold_until_killed"]
        assert any(r.get("phase") == "client_closed" for r in log)
    finally:
        p.stop()


def test_driver_framing():
    assert AD.frame({"a": 1}) == b'{"a":1}\n'
    msgs, rest = AD.parse_frames(bytearray(b'{"id":1}\n\n{"method":"x"}\n{"par'))
    assert msgs == [{"id": 1}, {"method": "x"}] and rest == bytearray(b'{"par')
    with pytest.raises(AD.DriverError):
        AD.parse_frames(bytearray(b"not json\n"))
    upd = {"method": "session/update", "params": {"update": {"sessionUpdate": "agent_message_chunk", "content": {"text": "hi"}}}}
    assert AD.agent_text(upd) == "hi" and AD.agent_text({"method": "session/update", "params": {}}) is None


PEER = textwrap.dedent("""
    import json, sys
    for line in sys.stdin:
        m = json.loads(line)
        if "id" not in m or "method" not in m:
            continue
        if m["method"] == "session/prompt":
            print(json.dumps({"jsonrpc": "2.0", "id": 99, "method": "session/request_permission", "params": {}}), flush=True)
            reply = json.loads(sys.stdin.readline())
            assert reply["id"] == 99 and reply["error"]["code"] == -32601
            for part in ("he", "llo"):
                print(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {"update": {
                    "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": part}}}}), flush=True)
            print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": {"stopReason": "end_turn"}}), flush=True)
        elif m["method"] == "session/new":
            print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": {"sessionId": "s-1"}}), flush=True)
        else:
            print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": {}}), flush=True)
""")


def test_acp_process_against_a_stdio_peer(tmp_path):
    proc = AD.AcpProcess([sys.executable, "-c", PEER], {"PATH": "/usr/bin:/bin"}, tmp_path, tmp_path / "err")
    try:
        proc.initialize(10)
        sid = proc.new_session(tmp_path, 10)
        assert sid == "s-1" and proc.prompt(sid, "hi", 10) == ("hello", "end_turn")
    finally:
        proc.close()
    killed = AD.AcpProcess([sys.executable, "-c", "import time; time.sleep(30)"], {"PATH": "/usr/bin:/bin"}, tmp_path,
                           tmp_path / "err2")
    killed.kill()
    with pytest.raises(AD.ProcessGone):
        killed.request("initialize", {}, 10)
    assert killed.close() == -9


GUARD = textwrap.dedent("""
    import json, socket, sys
    sys.path.insert(0, sys.argv[1])
    import probe
    seen = []
    probe.guard_sockets(local_ok=sys.argv[2] == "1", on_refuse=lambda *a: seen.append(a[0]))
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    out = {}
    for name, fn in [("local", lambda: socket.create_connection(srv.getsockname(), timeout=1)),
                     ("remote", lambda: socket.socket().connect(("192.0.2.1", 80))),
                     ("remote_ex", lambda: socket.socket().connect_ex(("192.0.2.1", 80))),
                     ("udp", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("192.0.2.1", 53))),
                     ("dns", lambda: socket.getaddrinfo("example.com", 443))]:
        try:
            fn(); out[name] = "ok"
        except OSError:
            out[name] = "refused"
    print(json.dumps({"out": out, "seen": seen}))
""")


@pytest.mark.parametrize("local_ok", ["1", "0"])
def test_socket_guard_refuses_connect_ex_udp_literals_and_dns(local_ok):
    rel = str(Path(__file__).resolve().parent.parent / "bench" / "instruments" / "reliability")
    got = json.loads(subprocess.run([sys.executable, "-c", GUARD, rel, local_ok], capture_output=True, text=True,
                                    check=True).stdout)
    assert got["out"] == {"local": "ok" if local_ok == "1" else "refused", "remote": "refused", "remote_ex": "refused",
                          "udp": "refused", "dns": "refused"}
    assert {"connect", "connect_ex", "sendto", "getaddrinfo"} <= set(got["seen"])


def test_chronology_is_reported_per_lineage():
    rows = [(1, "S0", "user", "[T01] user turn 1"), (2, "S0", "assistant", "r"), (3, "S0", "user", "[T03] user turn 3"),
            (4, "S0", "user", "[T02] user turn 2"), (5, "X", "user", "[K01] user turn 1")]
    rep = chronology.report(rows, lambda sid: "chat" if sid == "S0" else sid)
    assert rep["user_rows_checked"] == 4 and rep["violations"] == 1
    assert rep["examples"][0] == {"lineage": "chat", "store_id": 4, "tag": "T02", "after_tag": "T03", "after_store_id": 3}


def test_process_cells_are_selected_or_unsupported_and_config_is_localhost_only():
    by = {c["id"]: c for c in cells.registry()}
    for cid in ("baseline/in-place/acp", "crash-after-compaction/in-place/acp-history", "cancel-retry/in-place",
                "lcm-tool-mid-turn/in-place"):
        assert PC.unsupported(by[cid], "acp-process") is None, cid
    assert "in-process" in PC.unsupported(by["publication-failure/pass-3-in-place"], "acp-process")
    assert "gateway-process" in PC.unsupported(by["gateway-second-restart/in-place"], "acp-process")
    assert "cron" in PC.unsupported(by["multi-session-one-process/in-place"], "acp-process")
    plugin = {"engine": "lcm-x", "enabled": "hermes-lcm-x"}
    cfg = PC.config_yaml(by["baseline/in-place/acp"], plugin, "http://127.0.0.1:5/v1")
    urls = [line.split(":", 1)[1].strip().strip('"') for line in cfg.splitlines() if "base_url" in line]
    assert len(urls) == 3 and all(u == "http://127.0.0.1:5/v1" for u in urls)
    assert "context_length: 128000" in cfg and "model_catalog:\n  enabled: false" in cfg


def test_accounting_matches_requests_to_scripted_steps(tmp_path):
    reqs = [{"rid": 1, "role": "main", "reply": {}}, {"rid": 2, "role": "main", "phase": "held", "fault": "hold_until_killed"},
            {"rid": 2, "role": "main", "phase": "client_closed"}, {"rid": 3, "role": "lcm-summary", "reply": {}}]
    (tmp_path / "provider-requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    events = [{"phase": "A", "event": "emit"}, {"phase": "A", "event": "crash"}]
    acct = PC.accounting(tmp_path, events)
    assert acct["ok"] and acct["requests_by_role"] == {"main": 2, "lcm-summary": 1}
    assert not PC.accounting(tmp_path, events[:1])["ok"]
