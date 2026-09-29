# Diagnostics

Use read-only product tools before changing configuration or running an apply path.

## Fast path

1. `hermes plugins list`: confirm `hermes-lcm-x` is enabled and the selected context engine is `lcm-x` (`lcm` is the deprecated alias; `lcm_doctor` flags it under `identity_migration`).
2. Send one normal message if the session has not been bound since restart.
3. `lcm_status`: inspect runtime identity, database path, context pressure, summary/store counts, filters, and lifecycle state.
4. `lcm_inspect`: inspect current-session lineage, frontiers, fresh tail, externalized-ref readability, and skip/no-op reasons without retrieving content.
5. `lcm_doctor`: run database, FTS, lifecycle, configuration, and context-pressure diagnostics.

If optional slash commands are enabled, `/lcm status` and `/lcm doctor` expose the corresponding operator views.

## Safe mutation order

For cleanup, repair, source normalization, or rotate:

1. run the read-only preview;
2. inspect exact candidates and paths;
3. create/confirm a backup;
4. obtain user authorization for the specific apply operation;
5. run one bounded apply and verify integrity afterward.

Cleanup apply is separately feature-gated. Never infer permission to enable it from a diagnosis request.

## Common states

- Unbound status after restart: send a normal message, then check again.
- Database exists but stays empty: verify plugin enablement, `context.engine`, profile, database path, and ignore/stateless patterns.
- Weak exact recall: verify source rows exist, query construction/scope is correct, summary health is sound, and embedding coverage/provenance matches the requested mode.
- Conflicting summary and raw evidence: prefer the newer exact raw evidence and inspect lineage.
- Path B/context-engine schema log: expected on hosts where plugin-registry handlers do not receive active messages; context-engine schemas and dispatch remain the healthy route.
- Proactive recall injects nothing and `lcm_status` shows `proactive_recall.privacy_policy_errors > 0`:
  a deterministic embedding-privacy configuration fault, not load shedding. Check the
  `LCM_SENSITIVE_PATTERNS` catalog (nonempty, recognized names) and whether the registered
  vector revision matches the current posture; re-run `/lcm embed warmup` after any change.

## Compression runs but the context never shrinks

Symptom: `/compress` returns with the same message count, `lcm_status` keeps
`total_compactions: 0`, and `summary_nodes` stays at 0 for the session. This is a
summary-route failure, not a store problem. Work the chain in order — each link leaves a
distinct log line:

1. **The host watchdog, not just LCM's own timeout.** LCM's `summary_timeout_ms`
   (default 60 s) and Hermes' `auxiliary.compression.timeout` are separate knobs, and the
   host clamps its compression idle window up to the aux budget
   (`resolve_context_compression_timeouts`: idle = max(120, aux budget), ceiling = max(600, aux budget)).
   Leave `auxiliary.compression.timeout` unset and a reasoning summarizer is killed at the LCM
   60 s default while the host believes it allowed 120 s. The attempt telemetry then shows
   `total_duration_ms: ~60500`, `failure_class: no_progress`, `chunk_count: 0`, `chunking: false`,
   and the user sees only "No changes from compression". Read that telemetry line before
   touching the store. Fix: `hermes config set auxiliary.compression.timeout 300`.
2. **Route latency.** Recheck the combo's latency with the *contract* prompt, not a plain one:
   a plain probe finished in 10-21 s while the same model through the envelope needed 65-90 s
   for one large leaf chunk.
3. **Integrity-contract rejection.** A summary is kept only when the reply is exactly one
   `<lcm-summary nonce="...">…</lcm-summary>` envelope, carries no text outside it, and its
   final body line matches `Expand for details about: <...>`. Models mirror the contract
   template and add a nested `<summary>…</summary>`, which made the closing tag the last line
   and discarded an otherwise perfect summary. Log:
   `LCM summary discarded output that violated the integrity contract (model=...); escalating`.
   Two consecutive rejections open the route circuit breaker for 300 s
   (`LCM summary route circuit opened ... cooldown=300s`), after which compaction silently
   no-ops until the cooldown expires.
3. **Publication coverage.** Only after a summary survives (1) and (2) does the DAG write.
   `Compaction publication has no durable source coverage` means the covered message-to-row
   mapping came back empty. `Compaction publication source coverage is not contiguous
   (expected_frontier=N, authoritative=[...])` means `lcm_lifecycle_state.current_frontier_store_id`
   disagrees with the session's rows — a frontier above the session's max `store_id` can never
   publish again, so compare the two values before blaming the model.

Capture the raw reply instead of guessing: monkeypatch `agent.auxiliary_client.call_llm` to dump
`resp.choices[0].message.content`, then run one `compress()`.

## Session stuck at "No changes from compression" (the re-ingest loop)

The worst failure mode: `/compress` returns "No changes from compression: N messages" forever, the
session eventually exceeds the model window, and the LCM store keeps *growing*. Verify with
`SELECT count(*), count(DISTINCT ...) FROM messages WHERE session_id=?` — 40-60% redundant rows is
the signature.

Root cause: the ingest-cursor reconciliation drifts from the host's real history, so every attempt
re-appends the whole conversation and then fails publication.

1. `_reconcile_ingest_cursor_from_store` must align the store's tail with the incoming list. Ask it
   directly and read the recorded decision (monkeypatch `_record_ingest_reconciliation` to capture
   `action`/`reason`/`cursor`). The killer is `reason='persisted ambiguous delta'` with `cursor=0`:
   the whole history is persisted as new rows on every attempt.
2. Store drift is usually a row-COUNT mismatch (store N vs incoming N+7) and/or a dropped
   `tool_name`. Identity tuples compare `(role, content, tool_call_id, …, tool_name)`, so a tool row
   stored without its name never matches again. When re-seeding through
   `engine._store.append_batch`, pass `tool_name` (not `name`) — `append_batch` reads
   `msg["tool_name"]` and only maps it back to `name` on read.
3. Once rows are duplicated, publication fails permanently with
   `source coverage is not contiguous (expected_frontier=0, authoritative=[...])`, because the
   session's owned rows in the covered range now exceed what the summarized chunk can prove.
   Cleaning the duplicates alone does not hold — the loop re-duplicates within one attempt.

Repair (the approach that worked in production):

1. Read the host's true active history from `~/.hermes/state.db`:
   `SELECT role, content, tool_calls, tool_call_id, tool_name, timestamp FROM messages
   WHERE session_id=? AND active=1 ORDER BY id` (that table also marks the pre-compaction turns
   `active=0, compacted=1`, so nothing is lost).
2. Delete the session's rows in `lcm.db` and re-seed exactly that list via
   `engine._store.append_batch(session, rows, source="tui")`; reset
   `lcm_lifecycle_state.current_frontier_store_id` to 0.
3. Let the **host** commit the compaction — do not hand-write the commit. Trigger a real turn
   (`hermes chat --resume <id> -q "Reply with exactly: ok" --oneshot`); turn-start preflight
   compresses, commits, and the session shrinks in place.
4. Confirm in `state.db` that `active=1` dropped (e.g. 655 → 34) and in `lcm.db` that a summary node
   exists and `current_frontier_store_id` advanced above 0. An advanced frontier is what finally
   breaks the re-ingest loop, because the next attempt has a durable commit proof.

- Build the config with `LCMConfig.from_env()`, never the bare `LCMConfig()`. The bare
  constructor ignores `LCM_DATABASE_PATH` and every `auxiliary.*` override, so a "test" run
  silently writes duplicate rows into the live `$HERMES_HOME/lcm.db`.
- Snapshot with the SQLite backup API (`src.backup(dst)`). `cp lcm.db copy.db` misses the
  `-wal` and yields a stale snapshot, and a leftover `copy.db-wal` from an earlier run replays
  already-deleted rows back into the copy.
- Point the engine at the copy, then set `engine._ingest_cursor = len(messages)` and
  `engine._ingest_cursor_needs_reconcile = False` so the reconstructed history counts as already
  persisted. Without it, ingest appends duplicates that break publication contiguity.
- Recovering an accidental live write: `DELETE FROM messages WHERE session_id=? AND source='unknown'`.
  Legitimate rows carry `desktop`/`ios`/`tui`; test-written rows are `unknown`.
- `escalation.py` and `LCMConfig.from_env()` are read once at engine construction, so a patched
  plugin or changed timeout only applies after the session/TUI restarts; there is no plugin-reload
  slash command (only `reload-mcp`).
- **A plugin edit in `~/.hermes/plugins/` does not reach a running gateway.** Hermes loads plugins
  from a per-environment copy at
  `~/.hermes/installs/<install>/environments/<env>/workspace/plugin-sources/<plugin>-<hash>/`.
  Each environment keeps its own copy, so several go stale. Edit the source *and* copy the file into
  every `plugin-sources/<plugin>-*/` directory, then restart; a new environment syncs the source,
  existing ones do not. Confirm which copy a process uses with
  `lsof -p <pid> | grep plugin-sources` and which venv it runs with
  `lsof -p <pid> | grep -o '/installs/[0-9a-f]*/environments/[0-9a-f]*/venv'`.
- Map sessions to processes with `~/.hermes/runtime/active_sessions.json`, and check
  `hermes sessions list` titles against store row counts before assuming which session a report
  refers to — several sessions share a title like "Infrastructure Resilience".
