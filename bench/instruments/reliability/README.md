# LCM-X reliability harness (R1): the in-process real-Hermes matrix

Runs the REAL Hermes turn loop (`AIAgent.run_conversation`, the plugin loader, SessionDB, the host
ContextCompressor) under each host's own python, against a deterministic scripted provider, over a
host x cell x lcm-x-ref matrix, and scores every cell against mechanical bars. $0, no network.

**Claim class: `advisory` / `code_green_local`.** A PASS proves that this lcm-x tree, on this host sha,
under this scripted in-process scenario, meets the bars. It does not prove behaviour under real models,
the real ACP/gateway processes, real transports or customer boxes.

## Run
```
uv run --no-project python bench/instruments/reliability/run_matrix.py \
  --hosts-file <hosts.local.json> --hosts eva-0.21.5,customer-0.21.2 \
  --plugin-ref origin/main[,v0.24.2,...] --cells 'crash-*,baseline/*' | all --jobs 8 --out <dir> [--keep-homes] [--keep-dbs fail|all] \
  [--lcm-env LCM_KEY=VAL ...]
```
- Hosts file: `--hosts-file`, else `$LCM_RELIABILITY_HOSTS`, else the host-prep lane's file; see
  `hosts.example.json`. A host whose path, lexical or resolved, is under the live `~/.hermes` is refused.
- Host identity is verified before any cell runs: `git rev-parse HEAD` == sha with no modified tracked file for
  a git tree; for an exported tree, `tree-manifest.json` next to it (written at host prep by
  `python hosts.py --write-manifest <src> --sha <sha> --git-repo <repo>`, which proves the tree equals
  `git archive <sha>`) must match the sha and the tree's current hash. An unverified host's cells are ERROR.
- Each ref is exported once with `git archive` into `<out>/plugins/<sha12>/`; the plugin dir name,
  `plugins.enabled` entry and engine name are read from that tree (v0.23.x = `hermes-lcm`/`lcm`).
- Per cell: `<out>/cells/<host>/<sha12>/<cell-slug>/` holds cell.json, transcript.jsonl, phase-*.json,
  probe logs, `db/` (sqlite backup-API copies) and verdict.json. `--keep-homes` keeps hermes-home.
- `--keep-dbs fail` (default) drops the db/ copies of PASS cells (regenerable); `--lcm-env` overrides LCM_*
  on every cell and is recorded in run.json and MATRIX.md.
- Output: `results.jsonl`, `MATRIX.md`, `ISSUE-MAP.md`. Re-render: `python report.py <out>`.
- Standalone scoring: `python -m bench.instruments.reliability.scorers.cli --db <lcm.db> --gauntlet-run <dir>`
  (copies the DB into a private temp dir first; the source file is never opened).

## How a cell runs
`probe.py` runs one phase: `<host python> probe.py --cell <cell.json> --phase A --start-turn N --cell-dir <dir>`
with cwd = host src, `HERMES_HOME=<cell>/hermes-home`, `HOME=<cell>/home`. It refuses (exit 3) a
HERMES_HOME/HOME at or under the real home's `.hermes`, and a cell dir under /tmp. Sockets are blocked;
the provider is a MagicMock scripted per turn (unique or repeated replies, tool plans, usage that is
estimated, provider-real or scaled); the host aux LLM and the LCM summariser (tag-preserving) are stubbed.
Tools execute for real (`lcm_*`, `todo`, `read_file` on files inside the cell dir).
Faults (`os._exit` crash after a compaction commit, after a rotation, between on_session_end and
on_session_start, mid tool call; a clean exit; an ACP cancel + re-send; an injected publication failure)
end a phase; the runner starts the next phase (a fresh host process on the same HERMES_HOME).
A gateway cell reloads the state.db transcript every turn and restarts at the tip after a rotation.
Every emulated host shape is located at run time in the host tree and recorded as `file:line`
(`citations` in phase-A.json); a shape that cannot be cited makes the cell UNSUPPORTED, as does a fault
whose trigger never fires at that host sha.

## Bars (all applicable bars must pass)
B1 and B2 are scored per session lineage: the chat lineage is S0 and its compression children (state.db
`parent_session_id`), each cron fire its own lineage; a row stored in the wrong lineage fails both.
- **B1** every `[Tnn]` user tag and `reply to Tnn` sits in exactly as many stored rows as the host holds,
  and every stored `continue` row is followed by the reply of its own turn (position-bound).
- **B2** multiset-v1 (port of the gauntlet's `lossless_bar_multiset.py`): per (role, sha256(NFC,
  whitespace-collapsed)) stored count == expected count; surplus and deficit reported apart. Every
  stored-only key and every split reply is surplus and fails. Expected =
  what the host held per attempt after its ACP strip and consecutive-user merge (a crashed prompt folded
  into the next composite counts once).
- **B3** zero `publication_invariant_conflict` log lines across phases.
- **B4** no failed turn, and the final forced compaction through the host's ACP `/compress` entry point
  (`compress_now`, or `_compress_context(force=True)` on hosts without it, selected once before invoking;
  re-invoked once after a cleanup-only `sanitized`) does not end in error, conflict or exception; an
  exception from the selected entry point fails with no fallback. Ending `sanitized`/noop is INCONCLUSIVE.
- **B5** depth-0 message-sourced summary nodes grow after every LCM pass; every host-native pass has its own
  host `commit_status: committed` telemetry line in the same turn; published passes (LCM or host-native,
  the final forced compaction excluded) >= `min_compactions`; `LCM compaction #` log lines == LCM passes.
- **B6** (tool cells) no message-sourced summary covers part of a tool group (#559 invariant); zero host
  orphan-tool-result drops.
- **B7** (native cells) no `native recovery did not produce a usable summary`, no host
  `summary_generation_aborted`, at most one native attempt per turn.

Native cells: `native-short-prefix/*` are rejected before the host summary call (`prefix_too_short`) and are
data; `native-long-prefix/*` (default tuning, 1M window) run the host ContextCompressor summary (stubbed aux
LLM, so no slow-summary timeouts) and LCM's post-summary checks; every rejection reason is recorded.

Verdicts: PASS, FAIL (failed bars with numbers), INCONCLUSIVE (no bar fails, one could not decide), ERROR (harness or host failure; never a PASS),
UNSUPPORTED (with the reason). A cell must prove its scenario ran or it is UNSUPPORTED, never PASS: every
planned tool call dispatched and its result seen by the next provider call, at least one stored tool group
for B6, at least one native attempt in a native cell, a cancel that reported `interrupted` before the
retry, and every host citation its faults and transport need. A runner job that raises is ERROR.
`sql_dup_counter.py`, `summary_nodes_report.py` and `compaction_ledger.py`
are ported as `scorers/dupes.py` and `scorers/summary.py` (diagnostics and the B5 ledger).

## Positive controls
PC-1 is a differential: lcm-x `47bd28e7` (before #498, the #494 fix) vs `ae1fb16d` on eva-0.21.5, rs34-0.21.5
and upstream-main: `baseline/in-place/acp` PASSes at both, `acp-trailing/in-place` FAILs only at
`47bd28e7` (the host's post-commit-proof persist strip). customer-0.21.2 passes both refs (no such strip).

## Limits
In-process only: no real `hermes acp`/gateway process, transport, model or timing. Gateway timestamp
rendering stays at its default (off). Upgrades from pre-fix DBs (#485/#542) and the Desktop/tui transport
(#463) are not covered; see ISSUE-MAP.md for every uncovered issue and the capability it needs.
