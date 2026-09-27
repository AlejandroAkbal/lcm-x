"""Export an lcm-x git ref with ``git archive`` and read its plugin identity FROM the exported tree.

v0.23.x trees install as ``hermes-lcm`` / engine ``lcm``; v0.24.x as ``hermes-lcm-x`` / ``lcm-x``. Nothing
here is hard-coded: the dir name and ``plugins.enabled`` entry come from plugin.yaml ``name``, the engine
from plugin_identity.py ``ENGINE_NAME`` (or, before that file existed, engine.py's ``name`` property).
"""
from __future__ import annotations

import io
import re
import subprocess
import tarfile
from pathlib import Path


def resolve(repo: Path, ref: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"],
                          capture_output=True, text=True, check=True).stdout.strip()


def identity(tree: Path) -> dict:
    name = re.search(r"^name:\s*[\"']?([\w.-]+)", (tree / "plugin.yaml").read_text(), re.M).group(1)
    ident = tree / "plugin_identity.py"
    engine = None
    if ident.exists():
        m = re.search(r"^ENGINE_NAME\s*=\s*[\"']([\w.-]+)[\"']", ident.read_text(), re.M)
        engine = m and m.group(1)
    if engine is None:
        m = re.search(r"def name\(self\)[^:]*:\s*\n\s*return\s+[\"']([\w.-]+)[\"']", (tree / "engine.py").read_text())
        engine = m and m.group(1)
    if not engine:
        raise ValueError(f"cannot read the context-engine name from {tree}")
    return {"dir": name, "enabled": name, "engine": engine, "module": "hermes_plugins." + name.replace("-", "_")}


def export(repo: Path, ref: str, plugins_root: Path) -> dict:
    sha = resolve(repo, ref)
    dest = plugins_root / sha[:12]
    done = dest / ".export-complete"
    if not done.exists():
        dest.mkdir(parents=True, exist_ok=True)
        blob = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", sha], capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            tar.extractall(dest, filter="data")
        done.write_text(sha + "\n")
    return {"ref": ref, "sha": sha, "tree": str(dest), **identity(dest)}
