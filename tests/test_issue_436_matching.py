"""#580 r3c: ``_match_occurrences`` is a maximum, deterministic matching, equal to an in-order greedy walk when
every row has one key, with no rows x occurrences structure (it runs inside compaction's 100k-row gap-fill)."""

from __future__ import annotations

import logging
import random
import time

from hermes_lcm import identity_anchor
from hermes_lcm.identity_anchor import _match_occurrences

KEYS = ("a", "b", "c")
BOUND = 20.0  # s: >= 20x the measured times (<= 0.9 s) on a shared box; the no-WARNING asserts are the guard


def _instance(rng, max_keys):
    rows = [{"store_id": 10 + i, "keys": rng.sample(KEYS, rng.randint(1, max_keys))} for i in range(rng.randint(0, 5))]
    return rows, [(100 + j, rng.choice(KEYS)) for j in range(rng.randint(0, 5))]


def _run(rows, occurrences, shuffle=None):  # shuffle: build each key set in another insertion order
    keys_of = (lambda row: set(shuffle.sample(row["keys"], len(row["keys"])))) if shuffle else (lambda row: set(row["keys"]))
    return _match_occurrences(rows, keys_of, occurrences)


def _maximum(rows, occurrences, used=frozenset()):
    if not rows:
        return 0
    return max([_maximum(rows[1:], occurrences, used)] + [1 + _maximum(rows[1:], occurrences, used | {o})
                                                      for o, key in occurrences if key in rows[0]["keys"] and o not in used])


def test_brute_force_maximum_valid_and_deterministic():
    rng = random.Random(580)
    for _ in range(20_000):
        rows, occurrences = _instance(rng, 2)
        result = _run(rows, occurrences)
        assert len(result) == _maximum(rows, occurrences)
        assert len(set(result.values())) == len(result)
        key_of, row_of = dict(occurrences), {row["store_id"]: row for row in rows}
        assert all(key_of[o] in row_of[sid]["keys"] for sid, o in result.items())
        assert _run(rows, occurrences, random.Random(1)) == result == _run(rows, occurrences, random.Random(2))


def test_single_key_rows_equal_an_in_order_greedy_walk():
    rng = random.Random(436)
    for _ in range(20_000):
        rows, occurrences = _instance(rng, 1)
        free, greedy = list(occurrences), {}
        for row in rows:
            hit = next((pair for pair in free if pair[1] == row["keys"][0]), None)
            if hit:
                free.remove(hit)
                greedy[row["store_id"]] = hit[0]
        assert list(_run(rows, occurrences).items()) == list(greedy.items())


def _untripped(caplog, rows, keys_of, occurrences, bound):
    started = time.perf_counter()
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.identity_anchor"):
        result = _match_occurrences(rows, keys_of, occurrences)
    assert time.perf_counter() - started < bound and not caplog.records
    return result


def test_scale_identical_rows_and_occurrences(caplog):
    rows = [{"store_id": i} for i in range(100_000)]
    for n in (100_000, 10):
        assert len(_untripped(caplog, rows, lambda _row: {"k"}, [(j, "k") for j in range(n)], BOUND)) == n


def test_scale_adversarial_chain(caplog):
    """Row i holds k_i; the extra row on k0 shifts the whole chain; each row on the top key then searches the chain
    down to k0 and fails (dead keys make every search after the first O(1))."""
    n, key = 20_000, (lambda i: (1.0, f"k{i:05d}"))
    rows = [{"store_id": i, "keys": {key(i), key(i + 1)}} for i in range(n)] + [{"store_id": n, "keys": {key(0)}}]
    rows += [{"store_id": n + e, "keys": {key(n)}} for e in range(1, 1001)]
    assert len(_untripped(caplog, rows, lambda row: row["keys"], [(j, key(j)) for j in range(n + 1)], BOUND)) == n + 1


def test_scale_hub_of_distinct_types(caplog):
    """Codex R3c probe: n rows {a, b_i} fill hub a, rows {b_i} (i < n/2) fill b_i, then n/2 rows {a} each shift one."""
    n = 50_000
    rows = [{"store_id": i, "keys": {"a", ("b", i)}} for i in range(n)]
    rows += [{"store_id": n + i, "keys": {("b", i)}} for i in range(n // 2)]
    rows += [{"store_id": 2 * n + i, "keys": {"a"}} for i in range(n // 2)]
    occurrences = [(j, "a") for j in range(n)] + [(n + i, ("b", i)) for i in range(n)]
    assert len(_untripped(caplog, rows, lambda row: row["keys"], occurrences, BOUND)) == 2 * n


def test_scale_hub_of_one_type(caplog):
    n = 50_000
    rows = [{"store_id": i, "keys": {"a", "c"}} for i in range(n)] + [{"store_id": n + i, "keys": {"a"}} for i in range(n)]
    occurrences = [(j, "a") for j in range(n)] + [(n + j, "c") for j in range(n)]
    assert len(_untripped(caplog, rows, lambda row: row["keys"], occurrences, BOUND)) == 2 * n


def test_a_spent_budget_leaves_a_valid_deterministic_direct_matching(caplog, monkeypatch):
    monkeypatch.setattr(identity_anchor, "_MATCH_WORK_PER_ITEM", 0)
    monkeypatch.setattr(identity_anchor, "_MATCH_WORK_FLOOR", 0)
    rng = random.Random(3)
    for _ in range(500):  # nothing is searched: each row takes a free key of its own in order, or nothing
        rows, occurrences = _instance(rng, 2)
        free, direct = list(occurrences), set()
        for row in rows:
            hit = next((pair for key in sorted(row["keys"]) for pair in free if pair[1] == key), None)
            if hit:
                free.remove(hit)
                direct.add(row["store_id"])
        result, key_of = _run(rows, occurrences, random.Random(1)), dict(occurrences)
        assert set(result) == direct and len(set(result.values())) == len(result)
        assert all(key_of[o] in row["keys"] for row in rows for sid, o in result.items() if sid == row["store_id"])
        assert result == _run(rows, occurrences, random.Random(2))
    caplog.clear()
    rows = [{"store_id": 1, "keys": ["a", "b"]}, {"store_id": 2, "keys": ["b"]}] + [{"store_id": i, "keys": ["a"]} for i in (3, 4, 5)]
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.identity_anchor"):
        assert _run(rows, [(10, "a"), (11, "b")]) == {1: 10, 2: 11}
    assert [record.levelname for record in caplog.records] == ["WARNING"]
