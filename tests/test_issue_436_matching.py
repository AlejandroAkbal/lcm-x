"""#580 r3c: ``_match_occurrences`` is a maximum, deterministic matching, equal to an in-order greedy walk when
every row has one key, with no rows x occurrences structure (it runs inside compaction's 100k-row gap-fill)."""

from __future__ import annotations

import random
import time

from hermes_lcm.identity_anchor import _match_occurrences

KEYS = ("a", "b", "c")


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


def test_scale_identical_rows_and_occurrences():
    rows = [{"store_id": i} for i in range(100_000)]
    for n in (100_000, 10):
        started = time.perf_counter()
        result = _match_occurrences(rows, lambda _row: {"k"}, [(j, "k") for j in range(n)])
        assert len(result) == n and time.perf_counter() - started < 2.0


def test_scale_adversarial_chain():
    """Row i holds k_i; the extra row on k0 shifts the whole chain; each row on the top key then searches the chain
    down to k0 and fails (dead keys make every search after the first O(1))."""
    n, key = 20_000, (lambda i: (1.0, f"k{i:05d}"))
    rows = [{"store_id": i, "keys": {key(i), key(i + 1)}} for i in range(n)] + [{"store_id": n, "keys": {key(0)}}]
    rows += [{"store_id": n + e, "keys": {key(n)}} for e in range(1, 1001)]
    started = time.perf_counter()
    result = _match_occurrences(rows, lambda row: row["keys"], [(j, key(j)) for j in range(n + 1)])
    assert len(result) == n + 1 and time.perf_counter() - started < 5.0
