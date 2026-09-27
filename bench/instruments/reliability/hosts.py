"""hosts.local.json loader. Default path: --hosts-file, else $LCM_RELIABILITY_HOSTS, else the host-prep lane's file.

Format (see hosts.example.json): ``{"hosts": {"<name>": {"python", "src", "sha", "hermes_version"}}}``.
A host whose python, src or hermes_home resolves under the real ``~/.hermes`` is refused: that is a live
install, never a harness host.
"""
from __future__ import annotations

import json
import os
import pwd
from pathlib import Path

DEFAULT = "/Users/m1/Codex/lcmx-reliability/hosts/hosts.local.json"


def real_hermes_dir() -> Path:
    return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve() / ".hermes"


def under_real_hermes(path, hermes_dir: Path | None = None) -> bool:
    live = (hermes_dir or real_hermes_dir()).resolve()
    p = Path(path).expanduser().resolve()
    return p == live or live in p.parents


def hosts_file(arg: str | None) -> Path:
    return Path(arg or os.environ.get("LCM_RELIABILITY_HOSTS") or DEFAULT)


def load(path: Path, names: list[str] | None = None, hermes_dir: Path | None = None) -> dict:
    hosts = json.loads(Path(path).read_text())["hosts"]
    if names:
        unknown = [n for n in names if n not in hosts]
        if unknown:
            raise ValueError(f"unknown hosts {unknown}; {path} lists {sorted(hosts)}")
        hosts = {n: hosts[n] for n in names}
    for name, host in hosts.items():
        for key in ("python", "src", "sha"):
            if not host.get(key):
                raise ValueError(f"host {name}: missing {key!r}")
        for key in ("python", "src", "hermes_home"):
            if host.get(key) and under_real_hermes(host[key], hermes_dir):
                raise ValueError(f"host {name}: {key} {host[key]} is under the live ~/.hermes; refused")
        if not Path(host["src"]).is_dir():
            raise ValueError(f"host {name}: src {host['src']} is not a directory")
    return hosts
