"""Regression tests for cross-turn ``tool_call_id`` reuse (#586, #587)."""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


@pytest.fixture
def make_engine(tmp_path):
    engines = []

    def build(**overrides):
        settings = dict(
            database_path=str(tmp_path / f"tool-call-id-reuse-{len(engines)}.db"),
            fresh_tail_count=2,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=1_000_000,
            large_output_active_replay_stubbing_enabled=True,
            large_output_active_replay_stub_threshold_tokens=5,
        )
        settings.update(overrides)
        engine = LCMEngine(
            config=LCMConfig(**settings),
            hermes_home=str(tmp_path / "hermes"),
        )
        engine_index = len(engines)
        engine.on_session_start(
            f"tool-call-id-reuse-{engine_index}",
            conversation_id=f"tool-call-id-reuse-conversation-{engine_index}",
            context_length=200_000,
        )
        engines.append(engine)
        return engine

    yield build

    for engine in engines:
        engine.shutdown()


def _tool_call(call_id, name, arguments="{}"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _tool_result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _bypass_messages(second_call_id="call_0"):
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "turn A: list the repo"},
        {
            "role": "assistant",
            "content": "listing",
            "tool_calls": [
                _tool_call("call_0", "terminal", '{"command":"ls"}')
            ],
        },
        _tool_result("call_0", "A-RESULT " + ("file.txt\n" * 400)),
        {"role": "assistant", "content": "listed the repo"},
        {"role": "user", "content": "turn B: read b.txt"},
        {
            "role": "assistant",
            "content": "reading",
            "tool_calls": [
                _tool_call(second_call_id, "read_file", '{"path":"b.txt"}')
            ],
        },
        _tool_result(second_call_id, "B-RESULT: deployment is Friday"),
        {"role": "assistant", "content": "b.txt says deployment is Friday"},
    ]


def _trim_first_tool_group(engine, messages, removed_indices):
    expected = [
        message for index, message in enumerate(messages) if index not in removed_indices
    ]
    expected = engine._sanitize_active_context_messages(expected)
    target = count_messages_tokens(expected)
    assert count_messages_tokens(messages) > target
    return engine._trim_bypass_compacted_to_cap(messages, target), expected


def test_bypass_trim_keeps_later_result_when_tool_call_id_is_reused(make_engine):
    engine = make_engine()
    messages = _bypass_messages()

    result, expected = _trim_first_tool_group(engine, messages, {2, 3})

    assert result == expected
    assert any(
        message.get("role") == "tool"
        and message.get("content") == "B-RESULT: deployment is Friday"
        for message in result
    )


def test_bypass_trim_with_distinct_ids_is_unchanged(make_engine):
    engine = make_engine()
    messages = _bypass_messages(second_call_id="call_1")

    result, expected = _trim_first_tool_group(engine, messages, {2, 3})

    assert result == expected


def test_bypass_trim_removes_parallel_tool_group_together(make_engine):
    engine = make_engine()
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "inspect both files"},
        {
            "role": "assistant",
            "content": "reading both",
            "tool_calls": [
                _tool_call("a", "read_file", '{"path":"a.txt"}'),
                _tool_call("b", "read_file", '{"path":"b.txt"}'),
            ],
        },
        _tool_result("a", "A-RESULT " + ("alpha " * 300)),
        _tool_result("b", "B-RESULT " + ("beta " * 300)),
        {"role": "assistant", "content": "both files inspected"},
    ]

    result, expected = _trim_first_tool_group(engine, messages, {2, 3, 4})

    assert result == expected


def _active_replay_messages(expand_id, terminal_id):
    return [
        {"role": "user", "content": "what did we decide last week?"},
        {
            "role": "assistant",
            "content": "expanding",
            "tool_calls": [
                _tool_call(expand_id, "lcm_expand", '{"node":"s1"}')
            ],
        },
        _tool_result(
            expand_id,
            "EXPAND-RESULT: we decided to ship on Tuesday " * 3,
        ),
        {"role": "user", "content": "now dump the build log"},
        {
            "role": "assistant",
            "content": "dumping",
            "tool_calls": [
                _tool_call(terminal_id, "terminal", '{"command":"cat build.log"}')
            ],
        },
        _tool_result(terminal_id, "TERMINAL-LOG " + ("compile ok line\n" * 300)),
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "you're welcome"},
    ]


def test_active_replay_stubs_non_lcm_result_that_reuses_lcm_id(make_engine):
    engine = make_engine()
    messages = _active_replay_messages("call_0", "call_0")

    result = engine._stub_large_tool_results_for_active_replay(messages)

    assert result[2]["content"] == messages[2]["content"]
    assert result[5]["content"].startswith("[Externalized tool output:")
    assert messages[5]["content"].startswith("TERMINAL-LOG ")


def test_active_replay_with_distinct_ids_is_unchanged(make_engine):
    engine = make_engine()
    messages = _active_replay_messages("call_0", "call_1")

    result = engine._stub_large_tool_results_for_active_replay(messages)

    assert result[2]["content"] == messages[2]["content"]
    assert result[5]["content"].startswith("[Externalized tool output:")
