"""Relations, equivalence relations and partial orders.

Two relations drive the analysis:

* **Same emission band** (an equivalence relation) on activities. The
  equivalence classes group activities into bands such as "zero",
  "low", "medium" and "high".
* **Dominance** (a partial order) on profiles:
  P ≼ Q  iff  P emits no more than Q in *every* category.
  This is the product order on R^k. It is not total, because one person may
  fly more while another eats more meat. So we compute the Hasse diagram
  (the covering relation) and the minimal and maximal elements.
"""

from __future__ import annotations

from typing import Callable, Hashable, Iterable, TypeVar

T = TypeVar("T", bound=Hashable)
Relation = set[tuple[T, T]]


def relation_from_predicate(universe: Iterable[T], pred: Callable[[T, T], bool]) -> Relation:
    u = list(universe)
    return {(a, b) for a in u for b in u if pred(a, b)}


def is_reflexive(r: Relation, universe: Iterable[T]) -> bool:
    return all((a, a) in r for a in universe)


def is_symmetric(r: Relation) -> bool:
    return all((b, a) in r for a, b in r)


def is_antisymmetric(r: Relation) -> bool:
    return all(a == b or (b, a) not in r for a, b in r)


def is_transitive(r: Relation) -> bool:
    succ: dict = {}
    for a, b in r:
        succ.setdefault(a, set()).add(b)
    return all((a, c) in r for a, b in r for c in succ.get(b, ()))


def is_equivalence(r: Relation, universe: Iterable[T]) -> bool:
    universe = list(universe)
    return is_reflexive(r, universe) and is_symmetric(r) and is_transitive(r)


def is_partial_order(r: Relation, universe: Iterable[T]) -> bool:
    universe = list(universe)
    return is_reflexive(r, universe) and is_antisymmetric(r) and is_transitive(r)


def is_total_order(r: Relation, universe: Iterable[T]) -> bool:
    universe = list(universe)
    return is_partial_order(r, universe) and all(
        (a, b) in r or (b, a) in r for a in universe for b in universe
    )


def equivalence_classes(r: Relation, universe: Iterable[T]) -> list[frozenset[T]]:
    classes: list[frozenset[T]] = []
    seen: set[T] = set()
    for a in universe:
        if a in seen:
            continue
        cls = frozenset(b for (x, b) in r if x == a) | {a}
        classes.append(cls)
        seen |= cls
    return classes


def transitive_closure(r: Relation) -> Relation:
    """Warshall's algorithm."""
    nodes = sorted({x for pair in r for x in pair}, key=repr)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    m = [[False] * n for _ in range(n)]
    for a, b in r:
        m[idx[a]][idx[b]] = True
    for k in range(n):
        for i in range(n):
            if m[i][k]:
                row_k = m[k]
                row_i = m[i]
                for j in range(n):
                    if row_k[j]:
                        row_i[j] = True
    return {(nodes[i], nodes[j]) for i in range(n) for j in range(n) if m[i][j]}


def hasse_edges(r: Relation, universe: Iterable[T]) -> Relation:
    """Covering relation of a partial order: a ⋖ b iff a < b with nothing strictly between."""
    universe = list(universe)
    strict = {(a, b) for a, b in r if a != b}
    return {
        (a, b)
        for a, b in strict
        if not any((a, c) in strict and (c, b) in strict for c in universe)
    }


def minimal_elements(r: Relation, universe: Iterable[T]) -> set[T]:
    universe = list(universe)
    return {x for x in universe if not any(y != x and (y, x) in r for y in universe)}


def maximal_elements(r: Relation, universe: Iterable[T]) -> set[T]:
    universe = list(universe)
    return {x for x in universe if not any(y != x and (x, y) in r for y in universe)}


# ----------------------------------------------------------------------------
# Domain-specific relations
# ----------------------------------------------------------------------------

EMISSION_BANDS = (  # (label, lower bound inclusive, upper bound exclusive) in kg CO2e/unit
    ("zero", 0.0, 1e-9),
    ("low", 1e-9, 0.1),
    ("medium", 0.1, 1.0),
    ("high", 1.0, float("inf")),
)


def band_of(value: float) -> str:
    for label, lo, hi in EMISSION_BANDS:
        if lo <= value < hi:
            return label
    raise ValueError(value)


def same_band_relation(factors: dict[T, float]) -> Relation:
    """a ~ b iff f(a) and f(b) fall in the same emission band. This is an equivalence relation."""
    return relation_from_predicate(factors, lambda a, b: band_of(factors[a]) == band_of(factors[b]))


def dominance_relation(vectors: dict[T, dict[str, float]]) -> Relation:
    """P ≼ Q iff P[c] <= Q[c] for every category c (the product order).

    Profiles with identical vectors would break antisymmetry. Callers that need
    a strict poset should deduplicate them first.
    """
    keys = sorted({c for v in vectors.values() for c in v})

    def leq(p: T, q: T) -> bool:
        return all(vectors[p].get(c, 0.0) <= vectors[q].get(c, 0.0) + 1e-9 for c in keys)

    return relation_from_predicate(vectors, leq)
