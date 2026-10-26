"""Set theory: comparing footprints as sets of activities.

Used for questions like "which high-emission activities do these two people
share?" (intersection), "what does A do that B doesn't?" (difference), and
"how similar are their lifestyles?" (Jaccard index |A ∩ B| / |A ∪ B|).
"""

from __future__ import annotations

from itertools import chain, combinations
from typing import Callable, Hashable, Iterable, TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


def union(*sets: Iterable[T]) -> set[T]:
    out: set[T] = set()
    for s in sets:
        out |= set(s)
    return out


def intersection(*sets: Iterable[T]) -> set[T]:
    if not sets:
        return set()
    out = set(sets[0])
    for s in sets[1:]:
        out &= set(s)
    return out


def difference(a: Iterable[T], b: Iterable[T]) -> set[T]:
    return set(a) - set(b)


def symmetric_difference(a: Iterable[T], b: Iterable[T]) -> set[T]:
    return set(a) ^ set(b)


def complement(a: Iterable[T], universe: Iterable[T]) -> set[T]:
    return set(universe) - set(a)


def cartesian_product(a: Iterable[T], b: Iterable[T]) -> set[tuple[T, T]]:
    return {(x, y) for x in a for y in b}


def power_set(s: Iterable[T]) -> list[frozenset[T]]:
    """All 2^|s| subsets, in order of increasing size."""
    items = list(s)
    return [frozenset(c) for c in chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))]


def jaccard(a: Iterable[T], b: Iterable[T]) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def partition_by(items: Iterable[T], key: Callable[[T], K]) -> dict[K, set[T]]:
    """Partition ``items`` into blocks of equal ``key`` value (the fibres of ``key``)."""
    blocks: dict[K, set[T]] = {}
    for x in items:
        blocks.setdefault(key(x), set()).add(x)
    return blocks


def is_partition(blocks: Iterable[Iterable[T]], universe: Iterable[T]) -> bool:
    """Blocks are non-empty, pairwise disjoint, and their union is the universe."""
    blocks = [set(b) for b in blocks]
    if any(not b for b in blocks):
        return False
    seen: set[T] = set()
    for b in blocks:
        if seen & b:
            return False
        seen |= b
    return seen == set(universe)


def inclusion_exclusion_union_size(sets: list[set[T]]) -> int:
    """|A1 ∪ ... ∪ An| via the inclusion–exclusion principle.

    Exponential (2^n terms) on purpose: it is the textbook formula, and the
    result can be checked against ``len(union(*sets))``.
    """
    total = 0
    n = len(sets)
    for r in range(1, n + 1):
        sign = 1 if r % 2 else -1
        for combo in combinations(sets, r):
            total += sign * len(intersection(*combo))
    return total
