"""#608 wire contract with a real Hermes host: the host's overflow recovery passes ``bypass_cooldown=True`` to
an engine's ``compress()`` only when its signature names that parameter (``_supported_compression_kwargs`` in
``agent/conversation_compression.py``), and scores the result with its own estimator. The engine is built in
the host's own interpreter.

Runs only when a real Hermes runtime is available (the rule of ``test_real_plugin_manager_compaction.py``): set
``LCM_REAL_HERMES_PYTHON`` (and ``LCM_REAL_HERMES_SRC`` when the source tree is not importable from that
interpreter's cwd).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_PROBE = textwrap.dedent(
    """
    import importlib.util, inspect, json, os, sys
    root = os.environ["PROBE_REPO"]
    spec = importlib.util.spec_from_file_location("lcm_wire_probe", os.path.join(root, "__init__.py"),
                                                  submodule_search_locations=[root])
    package = importlib.util.module_from_spec(spec)
    sys.modules["lcm_wire_probe"] = package  # the package itself is not executed (no host registration)
    from lcm_wire_probe.config import LCMConfig
    from lcm_wire_probe.engine import LCMEngine
    from agent.conversation_compression import _supported_compression_kwargs
    engine = LCMEngine(config=LCMConfig(database_path=os.path.join(os.environ["HERMES_HOME"], "lcm.db")))
    kwargs = _supported_compression_kwargs(engine.compress, current_tokens=1000, focus_topic=None, force=False,
                                           memory_context="", bypass_cooldown=True)
    engine.shutdown()
    print(json.dumps({"kwargs": sorted(kwargs)}))
    """
)

_SHRINK_PROBE = textwrap.dedent(
    """
    import importlib.util, json, os, sys, time
    root = os.environ["PROBE_REPO"]
    spec = importlib.util.spec_from_file_location("lcm_wire_probe", os.path.join(root, "__init__.py"),
                                                  submodule_search_locations=[root])
    sys.modules["lcm_wire_probe"] = importlib.util.module_from_spec(spec)
    from lcm_wire_probe import compaction
    from lcm_wire_probe.config import LCMConfig
    from lcm_wire_probe.engine import LCMEngine
    from agent.model_metadata import estimate_messages_tokens_rough

    class Clock:  # the compaction module's time, its monotonic clock moved by the first store-id map
        offset = 0.0
        def __getattr__(self, name):
            return getattr(time, name)
        def monotonic(self):
            return time.monotonic() + self.offset

    compaction.time = clock = Clock()
    pad = " alpha beta gamma delta" * 30
    out = []
    for name, turns, threshold in (("D1", 8, 3000), ("D2", 40, 3000), ("D6", 8, 4500)):
        engine = LCMEngine(config=LCMConfig(
            fresh_tail_count=2, leaf_chunk_tokens=400, context_threshold=0.001, threshold_full_sweep_enabled=True,
            max_assembly_tokens=100_000, database_path=os.path.join(os.environ["HERMES_HOME"], name + ".db")))
        engine.on_session_start("S", platform="telegram", context_length=6000, conversation_id="conv")
        engine.threshold_tokens = threshold
        store_map = engine._get_store_id_map_for_messages
        def spent(*args, _map=store_map, **kwargs):
            clock.offset += 121.0
            return _map(*args, **kwargs)
        engine._get_store_id_map_for_messages = spent
        view = [{"role": "system", "content": "system prompt"}] + [
            row for i in range(turns) for row in (
                {"role": "user", "content": f"[T{i}] user turn{pad}", "timestamp": 10.0 * (i + 1)},
                {"role": "assistant", "content": f"reply to T{i}{pad}"})]
        engine.ingest(view)
        request = engine._survival_measure(view) + 1000
        if request < threshold:  # the sweep does not run below the threshold: a compaction with no leaf
            engine._compress_impl = lambda messages, **kwargs: messages
        result = engine.compress(view, current_tokens=request, bypass_cooldown=True)
        before, after = estimate_messages_tokens_rough(view), estimate_messages_tokens_rough(result)
        out.append({"name": name, "messages": [len(view), len(result)], "rough": [before, after],
                    "shrank": len(result) < len(view) or 0 < after < before * 0.95,
                    "reason": (engine._last_survival_fit or {}).get("reason")})
        engine.shutdown()
    print(json.dumps(out))
    """
)


def _hermes_python() -> tuple[str, str] | None:
    python = os.environ.get("LCM_REAL_HERMES_PYTHON")
    src = os.environ.get("LCM_REAL_HERMES_SRC", "")
    if python:
        return python, src
    if importlib.util.find_spec("hermes_cli") is not None:
        try:
            if importlib.util.find_spec("hermes_cli.plugins") is not None:
                return sys.executable, src
        except (ImportError, ValueError):
            return None
    return None


HERMES = _hermes_python()
pytestmark = pytest.mark.skipif(HERMES is None, reason="no real Hermes runtime available")


def test_real_host_passes_bypass_cooldown_to_the_engine(tmp_path):
    """C7: the host's own kwarg filter keeps bypass_cooldown for this engine's compress()."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    python, src = HERMES
    completed = subprocess.run(
        [python, "-c", _PROBE], cwd=src or None, capture_output=True, text=True, timeout=300, check=False,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "HERMES_HOME": str(home),
             "PROBE_REPO": str(REPO_ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert "bypass_cooldown" in result["kwargs"], result


def test_real_host_scores_recovery_results_as_shrunk(tmp_path):
    """D8 (real host): the D1, D2 and D6 results pass the host's shrink score with its own estimator."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    python, src = HERMES
    completed = subprocess.run(
        [python, "-c", _SHRINK_PROBE], cwd=src or None, capture_output=True, text=True, timeout=300, check=False,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "HERMES_HOME": str(home),
             "PROBE_REPO": str(REPO_ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    rows = json.loads(completed.stdout.strip().splitlines()[-1])
    assert [row["name"] for row in rows] == ["D1", "D2", "D6"]
    assert all(row["shrank"] and str(row["reason"]).startswith("recovery_attempt:") for row in rows), rows
