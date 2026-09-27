"""multiset-v1 lossless bar, ported from the v0.24.3-rc1 gauntlet ``lossless_bar_multiset.py``.

Key = (role, sha256(NFC(whitespace-collapsed content))) over non-empty user/assistant rows. Per key the
stored row count must equal the expected (transcript) count: fewer = loss (deficit), more = duplicates
(surplus). Repeated identical items are fine as long as the multiplicity matches. Split assistant
answers (one reply stored as two adjacent rows) are reported apart, as in the source.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").replace("\r\n", "\n")
    return re.sub(r"\s+", " ", text).strip()


def h(text: str) -> str:
    return hashlib.sha256(norm(text).encode()).hexdigest()


def score(expected: list[tuple[str, str]], stored_rows: list[tuple]) -> dict:
    """``expected``: (role, text) items; ``stored_rows``: (store_id, session_id, role, content)."""
    stored, by_session = defaultdict(list), defaultdict(list)
    for sid, session, role, content in stored_rows:
        if role not in ("user", "assistant") or not norm(content or ""):
            continue
        stored[(role, h(content))].append(sid)
        if role == "assistant":
            by_session[session].append((sid, content or ""))
    want = Counter((role, h(text)) for role, text in expected if norm(text))
    texts = {(role, h(text)): text for role, text in expected}
    missing, duplicated, split = [], [], []
    for key, n in want.items():
        have = len(stored.get(key, []))
        entry = {"role": key[0], "expected": n, "stored": have, "store_ids": stored.get(key, [])[:20],
                 "preview": norm(texts[key])[:80]}
        if have < n:
            if key[0] == "assistant" and have == 0:
                target = norm(texts[key])
                for rows in by_session.values():
                    for k in range(len(rows) - 1):
                        if target in (norm(rows[k][1] + rows[k + 1][1]), norm(rows[k][1] + " " + rows[k + 1][1])):
                            entry["split_match"] = [rows[k][0], rows[k + 1][0]]
                if "split_match" in entry:
                    split.append(entry)
                    continue
            missing.append(entry)
        elif have > n:
            duplicated.append(entry)
    extra = [{"role": k[0], "copies": len(v), "store_ids": v[:6]} for k, v in stored.items() if k not in want]
    return {
        "instrument": "multiset-v1",
        "verdict": "PASS" if not missing and not duplicated else "FAIL",
        "expected_items": sum(want.values()),
        "distinct_keys": len(want),
        "missing_keys": len(missing),
        "deficit_rows": sum(e["expected"] - e["stored"] for e in missing),
        "duplicated_keys": len(duplicated),
        "surplus_rows": sum(e["stored"] - e["expected"] for e in duplicated),
        "missing": missing[:40],
        "duplicated": duplicated[:40],
        "split_assistant_turns": split,
        "stored_rows_not_expected": len(extra),
        "extra": extra[:20],
    }
