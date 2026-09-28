"""#582 survival fit: a compaction that cannot bring the list under the model window never costs the session.

When compress() is about to return a list that is still over the window (a publication failure, a sweep
deadline, a no-op, a lock after commit, an exception), the list is fitted on the way out:
- budget = effective window x (1 - LCM_SURVIVAL_RESERVE) minus the host's observed-minus-counted overhead;
- the FINAL list is measured with the host's own request estimator (and LCM's count, whichever is larger)
  and the oldest whole user turns are dropped until it fits; a cut falls only on a user row, so no
  tool result is orphaned;
- when the newest user turn alone is over budget, its largest rows are projected: tool outputs through
  the existing large-output externalization, other rows as head/tail with their store id. The raw rows
  stay in the store;
- a row that is not durably stored is never omitted, and the list is never empty (#91).
It writes no message, node or lifecycle row: only the metadata counter /lcm doctor reads. The global
assembly cap is never set (that would force overflow and trim every good compaction). The notice goes in
the system-prefix slot when the list has one, never as a conversation row; the one-shot user warning goes
through the host's automatic-compaction status hook. LCM_SURVIVAL_FIT=false turns it off.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from .externalize import maybe_externalize_tool_output
from .message_content import normalize_content_value
from .tokens import count_message_tokens, count_messages_tokens

logger = logging.getLogger(__name__)

SURVIVAL_FIT_COUNTER_KEY = "survival_fit:counter"
_NOTICE = ("[LCM survival fit: {n} earlier messages (store ids {first}..{last}) are stored verbatim but not in "
           "live context; lcm_grep / lcm_load_session reach them.]")
_WARNING = ("LCM could not summarise part of this conversation in time. To keep the session alive, {n} older "
            "messages left live context; they stay stored verbatim and searchable (lcm_grep, lcm_load_session). "
            "/lcm doctor reports it.")
_PROJECTED = ("[LCM survival fit: this {role} message ({tokens} tokens) is stored verbatim as store id {store_id}; "
              "lcm_expand / lcm_grep reach the full text.]")


def _host_estimate(messages) -> Optional[int]:
    """The host's own request estimator (Hermes agent.model_metadata), when the host provides one."""
    try:
        from agent.model_metadata import estimate_messages_tokens_rough
    except Exception:
        return None
    try:
        return int(estimate_messages_tokens_rough(messages))
    except Exception:
        return None


class SurvivalFitMixin:
    """Mixed into LCMEngine; reads ``self._store``, ``self._config`` and ``self.context_length``."""

    def _survival_measure(self, messages) -> int:
        host = _host_estimate(messages)
        counted = count_messages_tokens(messages)
        return counted if host is None else max(host, counted)

    def _survival_fit_budget(self, messages, observed_tokens) -> Optional[int]:
        window = int(getattr(self, "context_length", 0) or 0)
        if window <= 0 or not getattr(self._config, "survival_fit", True):
            return None
        reserve = min(0.9, max(0.0, float(getattr(self._config, "survival_reserve", 0.15) or 0.0)))
        counted = _host_estimate(messages)
        counted = count_messages_tokens(messages) if counted is None else counted
        # system prompt + tools the host adds; more than half the window is a stale or synthetic observation
        overhead = min(window // 2, max(0, int(observed_tokens or 0) - counted))
        return max(1, int(window * (1 - reserve)) - overhead)

    def _survival_generated(self, message) -> bool:
        """LCM's own regenerated context (summaries, carriers): derived from stored rows, never a row."""
        return (self._is_replayed_context_scaffold_message(message)
                or self._generated_context_carrier_remainder(message) is not None
                or self._is_context_summary_content(message.get("content")))  # a host summary of stored rows

    def _survival_fit(self, messages, result, observed_tokens, reason: str, *, after_exception: bool = False):
        """``result``, or the fitted list when ``result`` is over the survival budget."""
        budget = self._survival_fit_budget(messages, observed_tokens)
        if budget is None or not isinstance(result, list) or not result or not self._session_id or \
                self._bypasses_lcm_context_management():
            return result
        before = self._survival_measure(result)
        if before <= budget:
            return result
        lead = 0
        while lead < len(result) and isinstance(result[lead], dict) and result[lead].get("role") == "system":
            lead += 1
        head, body = list(result[:lead]), list(result[lead:])
        store_ids = self._get_store_id_map_for_messages(body)
        # The ingest cursor indexes this list with nothing to reconcile: every row of it is persisted,
        # including rows the identity mapper cannot pin to one stored copy (duplicates, stubbed tools).
        persisted = not self._ingest_cursor_needs_reconcile and self._ingest_cursor == len(result)

        def durable(message) -> bool:
            return persisted or id(message) in store_ids or self._survival_generated(message)

        users = [i for i, message in enumerate(body) if isinstance(message, dict) and message.get("role") == "user"]
        cut, projected = None, False
        for index in users:
            if index and not all(durable(message) for message in body[:index]):
                break  # never omit a row that is not durably stored
            if index and self._survival_measure(head + body[index:]) <= budget:
                cut = index
                break
        if cut is None:  # the newest user turn alone is over budget: a bounded projection of it
            cut = users[-1] if users else 0
            if not all(durable(message) for message in body[:cut]):
                logger.warning("LCM survival fit skipped: an over-budget list holds rows not yet stored (reason=%s)", reason)
                return result
            kept = self._survival_projection(body[cut:], store_ids, budget - self._survival_measure(head), persisted)
            projected = True
        else:
            kept = body[cut:]
        dropped = body[:cut]
        ids = sorted(store_ids[id(message)] for message in dropped if id(message) in store_ids)
        count = sum(1 for message in dropped if not self._survival_generated(message))
        notice = _NOTICE.format(n=count, first=ids[0] if ids else "-", last=ids[-1] if ids else "-")
        if head:
            head[0] = {**head[0], "content": self._survival_with_notice(head[0].get("content"), notice)}
        fitted = head + kept or result[-1:]
        after = self._survival_measure(fitted)
        if after >= before:  # nothing stored could leave: the list is already as small as it gets
            return result
        if after_exception or not persisted:
            self._ingest_cursor, self._ingest_cursor_needs_reconcile = 0, True
        else:
            self._ingest_cursor = len(fitted)
        self._survival_record(reason, count, ids, before, after, budget, projected, notice)
        return fitted

    @staticmethod
    def _survival_with_notice(content: Any, notice: str) -> Any:
        if isinstance(content, list):
            return list(content) + [{"type": "text", "text": notice}]
        return f"{normalize_content_value(content) or ''}\n\n{notice}"

    def _survival_projection(self, turn: List[Dict[str, Any]], store_ids, limit: int,
                             persisted: bool = False) -> List[Dict[str, Any]]:
        """The newest turn, its largest stored rows stubbed until it fits (tool outputs first, then others);
        never a row that is not stored, never an empty list."""
        out = [dict(message) for message in turn]
        order = sorted(range(len(out)), key=lambda i: (out[i].get("role") != "tool", -count_message_tokens(out[i])))
        for index in order:
            if self._survival_measure(out) <= limit:
                break
            message, source = out[index], turn[index]
            tokens = count_message_tokens(message)
            if (id(source) not in store_ids and not persisted) or tokens < 256:
                continue
            text = normalize_content_value(message.get("content")) or ""
            stub = None
            if message.get("role") == "tool" and text:
                externalized = maybe_externalize_tool_output(
                    text, tool_call_id=str(message.get("tool_call_id") or ""), session_id=self._session_id,
                    config=self._config, hermes_home=self._hermes_home, force=True,
                )
                if externalized:
                    stub = self._active_tool_stub_content(message.get("content"), externalized["placeholder"])
            if stub is None:
                marker = _PROJECTED.format(role=message.get("role"), tokens=tokens, store_id=store_ids.get(id(source), "(unmapped)"))
                stub = f"{text[:1200]}\n...\n{marker}\n...\n{text[-600:]}" if len(text) > 2400 else marker
            out[index] = {**message, "content": stub}
        return out

    def _survival_record(self, reason, count, ids, before, after, budget, projected, notice) -> None:
        """Loud: a WARNING line, the doctor counter (metadata only) and one user warning per conversation."""
        logger.warning(
            "LCM survival fit applied (reason=%s, conversation=%s, dropped_rows=%d, store_ids=%s..%s, "
            "projected=%s, tokens=%d->%d, budget=%d)",
            reason, self._conversation_id or self._session_id, count, ids[0] if ids else "-", ids[-1] if ids else "-",
            projected, before, after, budget,
        )
        self._last_survival_fit = {"reason": reason, "dropped_rows": count, "notice": notice, "at": time.time()}
        try:
            record = self._store.read_metadata_json(SURVIVAL_FIT_COUNTER_KEY)
            record = record if isinstance(record, dict) else {}
            record = {"count": int(record.get("count") or 0) + 1, "last_reason": reason, "last_at": time.time(),
                      "last_conversation": str(self._conversation_id or self._session_id or "")}
            self._store.write_metadata_json([SURVIVAL_FIT_COUNTER_KEY], json.dumps(record, sort_keys=True))
        except Exception:
            logger.debug("LCM survival-fit counter write failed", exc_info=True)
        key = str(self._conversation_id or self._session_id or "")
        if key not in self._survival_fit_warned:
            self._survival_fit_warned.add(key)
            self._survival_fit_pending_warning = _WARNING.format(n=count)
            self.emit_automatic_compaction_status = True  # the host asks the hook below once more

    def get_automatic_compaction_status_message(self, *, phase: str, default_message: str, **context: Any):
        """LCM keeps automatic compaction silent; a pending survival-fit warning is shown once."""
        pending = getattr(self, "_survival_fit_pending_warning", None)
        self._survival_fit_pending_warning = None
        self.emit_automatic_compaction_status = False
        return pending
