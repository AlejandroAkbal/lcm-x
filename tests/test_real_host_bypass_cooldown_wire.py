"""#608 wire contract with a real Hermes host: the host's overflow recovery passes ``bypass_cooldown=True`` to
an engine's ``compress()`` only when its signature names that parameter (``_supported_compression_kwargs`` in
``agent/conversation_compression.py``). The engine is built in the host's own interpreter.

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
