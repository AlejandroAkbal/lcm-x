"""#436: host-timestamp-anchored, occurrence-bound message identity (design REVISION 1).

Hermes re-issues every durable row on every copy path (compaction generations, rotation, restore,
``replace_messages``) and rewrites, merges and re-orders scaffolding on the way. The message
``timestamp`` survives all of them and LCM stores it as ``observed_at``. Each incoming row with a host
stamp is recognised INDIVIDUALLY against the stored occurrences of the proven lineage instead of as part
of an ordered prefix, so one unmatched row no longer re-appends the rest of the list.

Rules (REVISION 1):
R1 the full replay payload identity plus ``observed_at`` finds CANDIDATES in this session and its
verified compression ancestors (host state.db ``parent_session_id``), consumed once per host
occurrence (multiset); R8 LCM's own carriers and summaries keep their DAG-verified identity.
R5 a NULL stamp is backfilled only for a uniquely proven occurrence, and H1's unstable current-turn
stamp attaches only to the one occurrence proven this turn.
R2 a merge survivor is recognised only by an exact, unique, ordered decomposition into stored
occurrences, or by a relation LCM itself recorded when it saw that composite; R3 a survivor holding
a stored head and a new remainder stores the remainder once (its own stamp unknown) with the relation.

``LCM_IDENTITY_ANCHOR`` (default on): ``0``/``false``/``no``/``off`` restores the pre-#436 ingest exactly.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .message_content import normalize_content_value, text_content_for_pattern_matching
from .store import _normalize_observed_at

logger = logging.getLogger(__name__)

_RECENT_CAP = 16  # rows this process stored lately: the R5 current-turn window
_POOL_WINDOW = 256  # store ids either side of a stamp donor searched for a composite's constituents
_MAX_DECOMPOSITIONS = 3


def identity_anchor_enabled() -> bool:
    return (os.environ.get("LCM_IDENTITY_ANCHOR") or "").strip().lower() not in {"0", "false", "no", "off"}


def _decompositions(content: str, texts: set, *, partial: bool) -> list:
    """Ordered splits of ``content`` into ``"\\n\\n"``-joined whole ``texts`` (at least two parts).
    ``partial`` also returns ``(parts, remainder)`` for a held prefix followed by a new remainder.
    Stops after ``_MAX_DECOMPOSITIONS`` (the caller only needs to know whether one is unique)."""
    out: list = []
    by_head: dict[str, list] = defaultdict(list)
    short = [text for text in texts if 0 < len(text) < 16]
    for text in texts:
        if len(text) >= 16:
            by_head[text[:16]].append(text)

    def walk(pos: int, parts: list) -> None:
        if len(out) >= _MAX_DECOMPOSITIONS:
            return
        if partial and parts and pos < len(content):
            out.append((list(parts), content[pos:]))
        for text in by_head.get(content[pos:pos + 16], []) + short:
            end = pos + len(text)
            if not content.startswith(text, pos):
                continue
            if end == len(content) and len(parts) >= 1:
                out.append((parts + [text], ""))
            elif content.startswith("\n\n", end) and end + 2 < len(content):
                walk(end + 2, parts + [text])

    walk(0, [])
    return out


class IdentityAnchorMixin:
    """Mixed into LCMEngine; reads ``self._store``, the reconcile identity helpers and ``_state_db_path``."""

    # -- lineage --------------------------------------------------------------

    def _identity_anchor_chain(self) -> list[str]:
        """The bound session's verified compression ancestors, nearest first: host state.db
        ``sessions.parent_session_id`` while the parent ended with ``end_reason='compression'``
        (written in one txn by ``publish_compression_child``). Read-only; empty when unreadable."""
        session_id = str(self._session_id or "")
        cached = getattr(self, "_identity_anchor_chain_cache", None)
        if cached is not None and cached[0] == session_id:
            return cached[1]
        chain: list[str] = []
        try:
            path = self._state_db_path()
            if session_id and path.exists():
                conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1.0)
                try:
                    current, seen = session_id, {session_id}
                    for _ in range(256):
                        row = conn.execute(
                            "SELECT p.id, p.end_reason FROM sessions c JOIN sessions p ON p.id = c.parent_session_id "
                            "WHERE c.id = ? LIMIT 1", (current,),
                        ).fetchone()
                        if not row or str(row[1] or "") != "compression" or str(row[0]) in seen:
                            break
                        current = str(row[0])
                        seen.add(current)
                        chain.append(current)
                finally:
                    conn.close()
        except Exception as exc:  # host DB drift or absence: no ancestry, session scope only
            logger.debug("LCM identity-anchor ancestry read failed: %s", exc)
            chain = []
        self._identity_anchor_chain_cache = (session_id, chain)
        return chain

    # -- R1-R5 pre-match -----------------------------------------------------

    def _identity_is_lcm_scaffold(self, message, *, verified: bool = False) -> bool:
        """R8: LCM's own summary/carrier rows keep their DAG-verified identity (``verified``: only a
        DAG-verified pure LCM scaffold; else any summary-shaped row, which the anchor leaves alone)."""
        if self._is_verified_replay_scaffold_message(message):
            return True
        text = text_content_for_pattern_matching(message.get("content")) or ""
        return not verified and bool(self._is_context_summary_content(text))

    def _identity_text(self, row) -> str:
        memo = self._identity_anchor_text_memo
        store_id = int(row.get("store_id") or 0)
        if store_id not in memo:
            memo[store_id] = self._message_replay_identity(row, stored_row=True)[1]
        return memo[store_id]

    def _identity_anchor_prematch(self, messages, identity_messages, cursor: int, audit_from: Optional[int] = None) -> Dict[str, Any]:
        """Rows at or after ``cursor`` recognised as replays of stored occurrences, plus the R3
        remainders to store, the relations to record and the R5 backfills. ``audit_from``: the host
        changed its list before the cursor from there (a positional cursor no longer proves those rows
        stored): a stamped row there that no stored occurrence explains moves ``plan["cursor"]`` back."""
        plan: Dict[str, Any] = {"replayed": set(), "remainders": {}, "relations": [], "backfill": [],
                                "cursor": cursor}
        self._identity_anchor_text_memo: dict[int, str] = {}
        n = len(messages)
        start = cursor if audit_from is None else max(0, min(audit_from, cursor))
        if not identity_anchor_enabled() or not self._session_id or start >= n:
            return plan
        stamps = {}
        for idx in range(n):
            observed_at = _normalize_observed_at(messages[idx].get("timestamp"))
            if observed_at is not None:
                stamps[idx] = observed_at
        wanted = {stamps[idx] for idx in range(start, n) if idx in stamps}
        if not wanted:
            return plan
        chain = self._identity_anchor_chain()
        rows = self._store.find_rows_by_observed_at(
            str(self._conversation_id or ""), [str(self._session_id), *chain], sorted(wanted)
        )
        self._load_host_rewrite_overrides(rows)
        by_stamp: dict[float, list] = defaultdict(list)
        for row in rows:
            by_stamp[float(row["observed_at"])].append((row, self._stored_row_forms(row)))
        identities: dict[int, Optional[tuple]] = {}

        def identity_at(idx: int) -> Optional[tuple]:
            if idx not in identities:
                message = identity_messages[idx]
                identity = None if self._identity_is_lcm_scaffold(message) else self._message_replay_identity(
                    message, strip_carrier=False
                )
                identities[idx] = None if identity is None or _lossy(identity) else identity
            return identities[idx]

        view_counts: dict = {}

        def view_count(identity) -> int:  # occurrences in the WHOLE host view (multiplicity evidence)
            if not view_counts:
                for i, message in enumerate(identity_messages):
                    key = identities.get(i) if i in identities else self._message_replay_identity(message, strip_carrier=False)
                    view_counts[key] = view_counts.get(key, 0) + 1
            return view_counts.get(identity, 0)

        consumed: set[int] = set()
        matched: dict[int, list] = {}
        # R1: per key, the host view's occurrences consume the stored ones in order; the rest are new.
        for idx in sorted(i for i, stamp in stamps.items() if stamp in wanted):
            identity = identity_at(idx)
            if identity is None:
                continue
            row = next((r for r, forms in by_stamp[stamps[idx]]
                        if int(r["store_id"]) not in consumed and identity in forms), None)
            if row is not None:
                consumed.add(int(row["store_id"]))
                matched[idx] = [row]
                if idx >= start:
                    plan["replayed"].add(idx)
        for idx in range(start, n):
            identity = identity_at(idx) if idx in stamps and idx not in plan["replayed"] else None
            if identity is not None and identity[0] == "user":
                self._identity_anchor_user_row(idx, identity, stamps[idx], by_stamp, consumed, matched, plan, view_count)
        if start < cursor:
            self._identity_anchor_audit(messages, identity_messages, cursor, start, stamps, identity_at, consumed, plan)
        self._identity_anchor_tool_segments(messages, plan["cursor"], plan["replayed"], matched, plan.get("positional", ()))
        for idx in list(plan["remainders"]):
            if idx in plan["replayed"]:
                del plan["remainders"][idx]
        plan["replayed"] = {idx for idx in plan["replayed"] if idx >= plan["cursor"]}
        return plan

    def _identity_anchor_audit(self, messages, identity_messages, cursor, start, stamps, identity_at, consumed, plan) -> None:
        """Rows in ``[start, cursor)`` of a list the host changed before the cursor: a stamped host row
        that no stored occurrence explains (key, witness, alias, or a stored copy of its content -- up to
        edge whitespace, under another stamp or none -- in the tail of this session or a verified
        ancestor, each copy used once) and no ignore pattern drops was never stored. The cursor moves back
        to the first such row; every other row of that range stays a replay (today's positional proof)."""
        from .reconcile import _proof_user_identity

        span = max(64, 2 * (cursor - start))
        held: dict = defaultdict(list)
        for session in [str(self._session_id), *self._identity_anchor_chain()]:
            for row in self._store.get_session_tail(session, limit=span):
                if int(row["store_id"]) not in consumed:
                    held[_proof_user_identity(self._message_replay_identity(row, stored_row=True))].append(row)
        missed = []
        for idx in range(start, cursor):
            identity = identity_at(idx) if idx in stamps else None
            if idx in plan["remainders"]:  # a held head plus a new remainder: the remainder is unstored
                missed.append(idx)
                continue
            if (identity is None or idx in plan["replayed"] or identity[0] not in ("user", "assistant")
                    or identity_messages[idx].get("tool_calls")  # tool rows: the host rewrites them in place
                    or self._message_replay_identity(identity_messages[idx]) != identity  # carries LCM's carrier (R8)
                    or self._matches_ignore_message_patterns(messages[idx])):
                continue
            copies = [row for row in held.get(_proof_user_identity(identity), ()) if int(row["store_id"]) not in consumed]
            if copies:
                consumed.add(int(copies[0]["store_id"]))
                if (len(copies) == 1 and copies[0].get("observed_at") is None
                        and self._message_replay_identity(copies[0], stored_row=True) == identity):
                    plan["backfill"].append((int(copies[0]["store_id"]), stamps[idx]))
                continue
            missed.append(idx)
        if missed:
            plan["cursor"] = min(missed)
            plan["positional"] = {idx for idx in range(min(missed), cursor) if idx not in missed}
            plan["replayed"].update(plan["positional"])
            logger.info("LCM identity-anchor: host changed its list before the cursor; %d unstored rows from %d: session=%s",
                        len(missed), min(missed), self._session_id)

    def _identity_anchor_user_row(self, idx, identity, stamp, by_stamp, consumed, matched, plan, view_count) -> None:
        """R2/R3/R5 for one unmatched user row: a recorded witness, an exact unique decomposition,
        the H1 unstable current-turn stamp, else a held head plus a new remainder. Else: new."""
        content = identity[1]
        donors = [row for row, _forms in by_stamp.get(stamp, ())
                  if row.get("role") == "user" and int(row["store_id"]) not in consumed]
        if not donors:
            return self._identity_anchor_constituent_copy(idx, identity, stamp, consumed, matched, plan)
        # R2 witness: a composite LCM itself saw at this stamp, or its rewritten survivor (U alone).
        for group in self._identity_anchor_witnesses(donors, stamp):
            texts = [self._identity_text(row) for row in group]
            free = all(int(row["store_id"]) not in consumed for row in group)
            if free and content == "\n\n".join(texts):
                return self._identity_anchor_take(idx, group, consumed, matched, plan)
            later = [row for row, text in zip(group, texts) if text == content and row.get("observed_at") != stamp]
            if len(later) == 1 and int(later[0]["store_id"]) not in consumed and texts.count(content) == 1:
                return self._identity_anchor_take(idx, later, consumed, matched, plan)
        if "\n\n" not in content:
            return
        pool = self._identity_anchor_pool(donors, consumed)
        texts = {self._identity_text(row) for row in pool}
        donor_texts = {self._identity_text(row) for row in donors}
        group, ambiguous = self._identity_anchor_compose(content, texts, pool, donors, consumed)
        if group is not None:
            plan["relations"].append(("composite", stamp, group, None))
            return self._identity_anchor_take(idx, group, consumed, matched, plan)
        if ambiguous:
            return  # several decompositions fit: the composite stays a pending (new) occurrence
        # R5: H1 keeps the absorbing row's stamp on a survivor the persist override rewrote to a row this
        # process stored in the current turn under the host's other stamp (in this session, or in the
        # verified ancestor a mid-turn rotation just closed): that ONE occurrence, aliased.
        scope = {self._session_id, *self._identity_anchor_chain()}
        recent = [entry for entry in getattr(self, "_identity_anchor_recent", ())
                  if entry[0] in scope and entry[1] == identity and entry[3] not in (None, stamp)]
        if (len(recent) == 1 and recent[0][2] not in consumed and view_count(identity) == 1
                and any(content.startswith(text + "\n\n") for text in donor_texts)):
            row = self._store.get_batch([recent[0][2]]).get(recent[0][2])
            if row is not None:
                plan["relations"].append(("alt_stamp", stamp, [row], None))
                return self._identity_anchor_take(idx, [row], consumed, matched, plan)
        # R3: held constituents then a new remainder, stored once with its own stamp unknown.
        partials = [(parts, rest) for parts, rest in _decompositions(content, texts, partial=True)
                    if rest and donor_texts & set(parts) and rest not in texts]
        longest = max((len(parts) for parts, _rest in partials), default=0)
        partials = [(parts, rest) for parts, rest in partials if len(parts) == longest]  # every held part accounted
        if len(partials) == 1:
            parts, rest = partials[0]
            group = self._identity_anchor_assign(parts, pool, donors, consumed)
            if group is not None:
                consumed.update(int(row["store_id"]) for row in group)
                matched[idx] = group
                plan["remainders"][idx] = (rest, stamp, group)

    def _identity_anchor_constituent_copy(self, idx, identity, stamp, consumed, matched, plan) -> None:
        """R3 + R5: a later timestamped copy of a remainder U whose own stamp was unknown: the host view
        carries it right after the occurrence of its composite's stamp donor R (the recorded relation
        orders R before U). That ONE NULL-stamped U is this occurrence; its stamp is backfilled."""
        previous = max((i for i in matched if i < idx), default=None)
        if previous is None or len(matched[previous]) != 1:
            return
        head = matched[previous][0]
        for group in self._identity_anchor_witnesses([head], head.get("observed_at")):
            members = [row for row in group[group.index(next(r for r in group if int(r["store_id"]) == int(head["store_id"]))) + 1:]
                       if row.get("observed_at") is None and int(row["store_id"]) not in consumed
                       and self._message_replay_identity(row, stored_row=True) == identity]
            if len(members) == 1:
                plan["backfill"].append((int(members[0]["store_id"]), stamp))
                return self._identity_anchor_take(idx, members, consumed, matched, plan)

    def _identity_anchor_compose(self, content, texts, pool, donors, consumed) -> tuple:
        """R2 form (i): ``(group, ambiguous)``; ``group`` is the one exact, unique, ordered decomposition
        of ``content`` into stored occurrences (a stamp donor among them), each used once."""
        donor_texts = {self._identity_text(row) for row in donors}
        full = [parts for parts, rest in _decompositions(content, texts, partial=False)
                if not rest and donor_texts & set(parts)]
        group = self._identity_anchor_assign(full[0], pool, donors, consumed) if len(full) == 1 else None
        return group, len(full) > 1

    def _identity_anchor_take(self, idx, rows, consumed, matched, plan) -> None:
        consumed.update(int(row["store_id"]) for row in rows)
        matched[idx] = list(rows)
        plan["replayed"].add(idx)

    def _identity_anchor_witnesses(self, donors, stamp) -> list:
        """Recorded composite groups at ``stamp`` headed by a donor, constituents in order."""
        groups: dict[tuple, list] = defaultdict(list)
        for rel in self._store.get_message_relations([int(row["store_id"]) for row in donors], "composite"):
            if rel["observed_at"] == stamp and rel["related_store_id"] is not None:
                groups[(rel["store_id"], rel["created_at"])].append((int(rel["ordinal"] or 0), int(rel["related_store_id"])))
        out = []
        for members in groups.values():
            ids = [store_id for _ordinal, store_id in sorted(members)]
            found = self._store.get_batch(ids)
            if len(found) == len(ids):
                out.append([found[store_id] for store_id in ids])
        return out

    def _identity_anchor_pool(self, donors, consumed) -> list:
        """Candidate constituents: unconsumed user rows of the lineage near each stamp donor, plus the
        bound session's recent rows (bounded)."""
        rows: dict[int, dict] = {}
        for donor in donors:
            store_id = int(donor["store_id"])
            for row in self._store.get_range(str(donor["session_id"]), start_id=max(0, store_id - _POOL_WINDOW),
                                             end_id=store_id + _POOL_WINDOW, limit=2 * _POOL_WINDOW + 1):
                rows[int(row["store_id"])] = row
        for row in self._store.get_session_tail(str(self._session_id), limit=64):
            rows[int(row["store_id"])] = row
        return [row for store_id, row in sorted(rows.items())
                if row.get("role") == "user" and store_id not in consumed]

    def _identity_anchor_assign(self, parts, pool, donors, consumed) -> Optional[list]:
        """Bind each part to one stored occurrence (a donor for a donor's text first), each used once."""
        taken: set[int] = set(consumed)
        donor_ids = {int(row["store_id"]) for row in donors}
        group = []
        for text in parts:
            options = sorted((row for row in pool if self._identity_text(row) == text
                              and int(row["store_id"]) not in taken),
                             key=lambda row: (int(row["store_id"]) not in donor_ids, int(row["store_id"])))
            if not options:
                return None
            taken.add(int(options[0]["store_id"]))
            group.append(options[0])
        return group

    def _identity_anchor_tool_segments(self, messages, cursor: int, hits: set, matched, proven=()) -> None:
        """An assistant tool-call segment is replayed whole or not at all, except a stored segment whose
        only new rows are trailing results resuming in place on its session's newest row. ``proven``
        rows (the audited prefix's positional replays) stay replayed."""
        index, n = max(cursor, 0), len(messages)
        while index < n:
            if str(messages[index].get("role") or "") == "assistant" and messages[index].get("tool_calls"):
                end = index + 1
                while end < n and str(messages[end].get("role") or "") == "tool":
                    end += 1
                held = [k for k in range(index, end) if k in hits]
                resumes = bool(held) and held == list(range(index, index + len(held))) and len(held) < end - index
                if resumes:
                    last = max((row for k in held for row in matched.get(k, ())), key=lambda r: int(r["store_id"]),
                               default=None)
                    tail = self._store.get_session_tail(str(last["session_id"]), limit=1) if last else None
                    resumes = bool(tail) and int(tail[-1]["store_id"]) == int(last["store_id"])
                if len(held) != end - index and not resumes:
                    hits.difference_update(k for k in range(index, end) if k not in proven)
                index = end
            else:
                index += 1

    def _identity_anchor_backfill_prefix(self, messages, identity_messages, cursor: int) -> list:
        """R5 NULL backfill for the reconciled prefix: a stamped row whose occurrence-bound mapped row
        (#488 mapper) has ``observed_at`` NULL and the exact identity, when no other mapped NULL row shares
        that identity and no stored row already holds it at that stamp."""
        pairs = [(idx, _normalize_observed_at(messages[idx].get("timestamp"))) for idx in range(min(cursor, len(messages)))]
        pairs = [(idx, stamp) for idx, stamp in pairs if stamp is not None]
        if not identity_anchor_enabled() or not pairs:
            return []
        saved = getattr(self, "_current_compress_placeholder_identity_counts", None)
        try:
            mapping = self._get_store_id_map_for_messages(messages[:cursor])
        finally:
            self._current_compress_placeholder_identity_counts = saved
        rows = self._store.get_batch(sorted({mapping[id(messages[i])] for i, _s in pairs if id(messages[i]) in mapping}))
        null = {store_id: row for store_id, row in rows.items() if row.get("observed_at") is None}
        if not null:
            return []
        identity_of = {store_id: self._message_replay_identity(row, stored_row=True) for store_id, row in null.items()}
        counts = defaultdict(int)
        for identity in identity_of.values():
            counts[identity] += 1
        anchored = self._store.find_rows_by_observed_at(
            str(self._conversation_id or ""), [str(self._session_id), *self._identity_anchor_chain()],
            [stamp for _idx, stamp in pairs],
        )
        held = {(float(row["observed_at"]), self._message_replay_identity(row, stored_row=True)) for row in anchored}
        out = []
        for idx, stamp in pairs:
            store_id = mapping.get(id(messages[idx]))
            identity = self._message_replay_identity(identity_messages[idx], strip_carrier=False)
            if (store_id in null and not _lossy(identity) and identity == identity_of[store_id]
                    and counts[identity] == 1 and (stamp, identity) not in held):
                out.append((store_id, stamp))
        return out

    # -- writes ----------------------------------------------------------------

    def _identity_anchor_commit(self, plan, remainder_ids: Optional[dict] = None) -> None:
        """Record what the pre-match proved: relation groups (witnesses, alternate stamps, remainders)
        and the R5 backfills."""
        if remainder_ids is None:
            groups = [[(int(group[0]["store_id"]), "alt_stamp", None, None, stamp)] if kind == "alt_stamp"
                      else _composite_relation(group, stamp) for kind, stamp, group, _extra in plan["relations"]]
        else:  # after the store: only the remainders' groups, which need the new ids
            groups = [_composite_relation(plan["remainders"][idx][2] + [{"store_id": store_id}], plan["remainders"][idx][1])
                      for idx, store_id in remainder_ids.items()]
        if groups:
            self._store.add_message_relations(groups)
        for store_id, stamp in plan["backfill"] if remainder_ids is None else ():
            self._store.backfill_observed_at(store_id, stamp)

    def _identity_anchor_remember(self, stored) -> None:
        """R5 current-turn window: (session, identity, store_id, observed_at) of user rows just stored."""
        recent = list(getattr(self, "_identity_anchor_recent", ()))
        for identity, store_id, message in stored:
            if identity is not None and identity[0] == "user":
                recent.append((self._session_id, identity, int(store_id), _normalize_observed_at(message.get("timestamp"))))
        self._identity_anchor_recent = recent[-_RECENT_CAP:]


def _composite_relation(group, stamp) -> list:
    """One composite relation group, keyed by the constituent whose host stamp the composite carries."""
    head = next((row for row in group if row.get("observed_at") == stamp), group[0])
    return [(int(head["store_id"]), "composite", int(row["store_id"]), ordinal, stamp) for ordinal, row in enumerate(group)]


def _lossy(identity) -> bool:
    from .reconcile import _has_lossy_redacted_identity

    return _has_lossy_redacted_identity(identity)


def _raw_remainder(message, remainder) -> Optional[str]:
    """R3's stored bytes: the survivor's raw content must be EXACTLY the held constituents' stored
    bytes, each joined by the host's ``"\n\n"``, then ``"\n\n"`` and the remainder; else None (the
    survivor is stored whole: visible duplication, never a lost or altered separator)."""
    rest, _stamp, group = remainder
    raw = normalize_content_value(message.get("content")) or ""
    head = "\n\n".join(normalize_content_value(row.get("content")) or "" for row in group) + "\n\n"
    return raw[len(head):] if raw.startswith(head) and raw[len(head):] == rest else None
