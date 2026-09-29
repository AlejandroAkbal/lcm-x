"""#620: after any survival fit, /lcm doctor gives the backup-restore rollback within 0.24.x, never a
plugin-only one; #618: a lost survival-fit counter write is logged at WARNING.

Engine-level: engine.ingest / engine.compress / handle_lcm_command("doctor"). The summary provider raises
(an auxiliary model outage), so the compaction fails and the survival fit drops whole turns without a
projection. The host estimator is pinned to a rough count so the fit is the same in every environment."""

from __future__ import annotations

import json
import logging
import sqlite3

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.survival_fit as survival_fit
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.survival_fit import SURVIVAL_FIT_COUNTER_KEY

PAD = " alpha beta gamma delta" * 30
WINDOW = 6000
RESTORE = "restore the lcm.db backup"
COUNTER_FAILED = "LCM survival-fit counter write failed"


def _rough(messages) -> int:
    return sum(4 + (len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or ""))) // 4
               for m in messages)


@pytest.fixture(autouse=True)
def _pinned_host_estimate(monkeypatch):
    monkeypatch.setattr(survival_fit, "_host_estimate", _rough)


def _turn(tag, tool=False):
    rows = [{"role": "user", "content": f"[{tag}] user turn{PAD}"}]
    if tool:
        call = {"id": f"call_{tag}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        rows += [{"role": "assistant", "content": "", "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": f"call_{tag}", "content": f"result of {tag}{PAD}"}]
    return rows + [{"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _history(turns=24):
    return [r for i in range(turns) for r in _turn(f"L{i}", tool=i % 3 == 0)]


def _engine(tmp_path) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=400, context_threshold=0.001,
                                        database_path=str(tmp_path / "lcm.db")))
    engine.on_session_start("S", platform="telegram", context_length=WINDOW, conversation_id="conv")
    return engine


def _fit_after_a_summary_outage(engine, monkeypatch):
    """Ingest 24 turns, then compress while the summary provider raises: the fit drops whole turns."""
    history = _history()
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation",
                        lambda **kw: ("Earlier turns.\nExpand for details about: turns", 1))
    engine.ingest(history)

    def _down(**kwargs):
        raise RuntimeError("auxiliary provider unavailable")

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _down)
    return history, engine.compress(history, current_tokens=_rough(history))


def _survival_lines(doctor: str) -> tuple[str, str]:
    lines = [line.strip() for line in doctor.splitlines()]
    observation = next(line for line in lines if line.startswith("- survival_fit: applied"))
    guidance = next(line for line in lines if line.startswith("- survival_fit:") and " — " in line)
    return observation, guidance


def test_drop_only_fit_gets_the_backup_restore_advice(tmp_path, monkeypatch):
    """After a drop-only fit (count 1, projected_count 0) neither the observation nor the triage
    guidance calls a plugin-only rollback supported; both name the backup restore."""
    engine = _engine(tmp_path)
    try:
        history, fitted = _fit_after_a_summary_outage(engine, monkeypatch)
        assert engine._last_compression_status == "error" and len(fitted) < len(history)
        record = engine._store.read_metadata_json(SURVIVAL_FIT_COUNTER_KEY)
        assert record["count"] == 1 and record["projected_count"] == 0
        observation, guidance = _survival_lines(handle_lcm_command("doctor", engine))
        assert "projected_count 0" in observation
        for text in (observation, guidance):
            assert "plugin-only" not in text
            assert RESTORE in text
    finally:
        engine.shutdown()


@pytest.mark.parametrize("record", [None, {"count": 0, "last_reason": "noop", "projected_count": 0}],
                         ids=["no-record", "count-0"])
def test_no_survival_fit_prints_no_observation_and_no_check(tmp_path, record):
    """Count 0 (or no record): no survival-fit observation and no survival_fit check."""
    engine = _engine(tmp_path)
    try:
        if record is not None:
            engine._store.write_metadata_json([SURVIVAL_FIT_COUNTER_KEY], json.dumps(record, sort_keys=True))
        lines = [line.strip() for line in handle_lcm_command("doctor", engine).splitlines()]
        assert not any(line.startswith("- survival_fit") or "survival_fit: applied" in line for line in lines)
        assert not any(RESTORE in line or "plugin-only" in line for line in lines)
    finally:
        engine.shutdown()


def test_the_advice_is_the_same_with_or_without_a_projection(tmp_path):
    """A projected_count above 0, 0, and absent (a record from before the key) give the same advice."""
    advice = {}
    for label, projected in (("projected", 3), ("none", 0), ("absent", None)):
        engine = _engine(tmp_path / label)
        try:
            record = {"count": 2, "last_reason": "publication_invariant_conflict"}
            if projected is not None:
                record["projected_count"] = projected
            engine._store.write_metadata_json([SURVIVAL_FIT_COUNTER_KEY], json.dumps(record, sort_keys=True))
            observation, guidance = _survival_lines(handle_lcm_command("doctor", engine))
            shown = "unknown" if projected is None else projected
            assert f"projected_count {shown}; " in observation
            advice[label] = (observation.replace(f"projected_count {shown}; ", ""), guidance)
        finally:
            engine.shutdown()
    assert advice["projected"] == advice["none"] == advice["absent"]
    observation, guidance = advice["none"]
    for text in (observation, guidance):
        assert "plugin-only" not in text and RESTORE in text


def test_a_failed_counter_write_logs_one_warning_and_returns_the_fitted_list(tmp_path, monkeypatch, caplog):
    """The counter write raises 'database is locked': exactly one WARNING record for it, and the fitted
    list still reaches the host."""
    engine = _engine(tmp_path)
    original = engine._store.update_metadata_json

    def locked(key, update):
        if key == SURVIVAL_FIT_COUNTER_KEY:
            raise sqlite3.OperationalError("database is locked")
        return original(key, update)

    monkeypatch.setattr(engine._store, "update_metadata_json", locked)
    try:
        with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
            history, fitted = _fit_after_a_summary_outage(engine, monkeypatch)
        failed = [r for r in caplog.records if r.getMessage().startswith(COUNTER_FAILED)]
        assert [r.levelno for r in failed] == [logging.WARNING]
        assert isinstance(failed[0].exc_info[1], sqlite3.OperationalError)
        assert len(fitted) < len(history) and engine._last_survival_fit is not None
        assert engine._store.read_metadata_json(SURVIVAL_FIT_COUNTER_KEY) is None
    finally:
        engine.shutdown()
