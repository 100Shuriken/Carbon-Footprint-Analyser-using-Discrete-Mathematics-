"""Combinatorics: counting and searching reduction plans.

Given n candidate actions, each with an annual CO2 saving and an "effort cost",
a *plan* is a subset of the actions, so there are 2^n plans. The optimiser
enumerates every subset (as a bitmask) to find the plan that saves the most
while staying within an effort budget. It also respects *conflicts*: pairs of
actions that cannot be chosen together, such as "go vegetarian" and "go vegan".

The number of plans of a fixed size k is C(n, k). The number of orders in which
to roll out k chosen actions is k!. The number of valid plans (the independent
sets of the conflict graph) is also counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, factorial, perm

from .emission_factors import factor


@dataclass(frozen=True)
class Action:
    key: str
    description: str
    saving_kg: float
    effort: int  # abstract effort/cost points


def n_choose_k(n: int, k: int) -> int:
    return comb(n, k)


def n_permute_k(n: int, k: int) -> int:
    return perm(n, k)


def total_plans(n: int) -> int:
    """|P(A)| = 2^n = sum_k C(n, k). This is the binomial theorem with x = y = 1."""
    return 2 ** n


def rollout_orders(k: int) -> int:
    return factorial(k)


def pigeonhole_min_share(items: int, boxes: int) -> int:
    """At least one box receives ⌈items / boxes⌉ items."""
    return -(-items // boxes)


def _conflict_mask(actions: list[Action], conflicts: set[frozenset[str]]) -> list[int]:
    idx = {a.key: i for i, a in enumerate(actions)}
    masks = [0] * len(actions)
    for pair in conflicts:
        a, b = tuple(pair)
        if a in idx and b in idx:
            masks[idx[a]] |= 1 << idx[b]
            masks[idx[b]] |= 1 << idx[a]
    return masks


def is_valid_plan(mask: int, conflict_masks: list[int]) -> bool:
    for i, cm in enumerate(conflict_masks):
        if mask >> i & 1 and mask & cm:
            return False
    return True


def count_valid_plans(actions: list[Action], conflicts: set[frozenset[str]] = frozenset()) -> int:
    cm = _conflict_mask(actions, set(conflicts))
    return sum(1 for m in range(1 << len(actions)) if is_valid_plan(m, cm))


@dataclass
class Plan:
    actions: list[Action]

    @property
    def saving_kg(self) -> float:
        return sum(a.saving_kg for a in self.actions)

    @property
    def effort(self) -> int:
        return sum(a.effort for a in self.actions)


def best_plan(
    actions: list[Action],
    budget: int,
    conflicts: set[frozenset[str]] = frozenset(),
) -> Plan:
    """Exhaustive search over all 2^n subsets (fine for n up to about 20).

    Ties on saving are broken by lower effort.
    """
    if len(actions) > 22:
        raise ValueError("Too many actions for exhaustive search (2^n blows up); use best_plan_dp")
    cm = _conflict_mask(actions, set(conflicts))
    best_mask, best_key = 0, (0.0, 0)
    for mask in range(1 << len(actions)):
        if not is_valid_plan(mask, cm):
            continue
        eff = sav = 0
        for i, a in enumerate(actions):
            if mask >> i & 1:
                eff += a.effort
                sav += a.saving_kg
        if eff <= budget and (sav, -eff) > best_key:
            best_key, best_mask = (sav, -eff), mask
    return Plan([a for i, a in enumerate(actions) if best_mask >> i & 1])


def best_plan_dp(actions: list[Action], budget: int) -> Plan:
    """0/1 knapsack by dynamic programming, O(n·budget). It ignores conflicts.

    Used to cross-check the exhaustive search when there are no conflicts.
    """
    n = len(actions)
    dp = [[0.0] * (budget + 1) for _ in range(n + 1)]
    for i, a in enumerate(actions, 1):
        for w in range(budget + 1):
            dp[i][w] = dp[i - 1][w]
            if a.effort <= w and dp[i - 1][w - a.effort] + a.saving_kg > dp[i][w]:
                dp[i][w] = dp[i - 1][w - a.effort] + a.saving_kg
    chosen, w = [], budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen.append(actions[i - 1])
            w -= actions[i - 1].effort
    return Plan(chosen[::-1])


# ----------------------------------------------------------------------------
# Actions tailored to a profile
# ----------------------------------------------------------------------------

DIET_KEYS = ("diet_heavy_meat", "diet_average", "diet_vegetarian", "diet_vegan")
DIET_CONFLICTS = {frozenset({"go_vegetarian", "go_vegan"})}


def candidate_actions(q: dict[str, float]) -> list[Action]:
    """Actions whose saving is actually positive for this profile."""
    g = lambda k: q.get(k, 0.0)  # noqa: E731
    out: list[Action] = []

    car_em = g("car_petrol") * factor("car_petrol") + g("car_diesel") * factor("car_diesel")
    car_km = g("car_petrol") + g("car_diesel")
    if car_km:
        avg = car_em / car_km
        out.append(Action("car_to_train", "Move 40% of car km to train/metro",
                          0.4 * car_km * (avg - factor("train")), 3))
        out.append(Action("car_to_bike", "Cycle for short trips (10% of car km)",
                          0.1 * car_km * avg, 2))
    if g("flight_short"):
        out.append(Action("no_short_flights", "Replace short-haul flights with train",
                          g("flight_short") * (factor("flight_short") - factor("train")), 3))
    if g("flight_long"):
        out.append(Action("fewer_long_flights", "Cut long-haul flying by half",
                          0.5 * g("flight_long") * factor("flight_long"), 4))

    diet_days = sum(g(k) for k in DIET_KEYS)
    diet_em = sum(g(k) * factor(k) for k in DIET_KEYS)
    for key, target, effort in (("go_vegetarian", "diet_vegetarian", 4), ("go_vegan", "diet_vegan", 6)):
        saving = diet_em - diet_days * factor(target)
        if saving > 1e-9:
            label = "Switch to a vegetarian diet" if target == "diet_vegetarian" else "Switch to a vegan diet"
            out.append(Action(key, label, saving, effort))

    if g("electricity"):
        out.append(Action("efficient_appliances", "Efficient appliances and LED lighting (-15% kWh)",
                          0.15 * g("electricity") * factor("electricity"), 2))
        out.append(Action("rooftop_solar", "Rooftop solar covering 60% of use",
                          0.6 * g("electricity") * (factor("electricity") - factor("solar")), 5))
    if g("waste_landfill"):
        out.append(Action("compost_recycle", "Compost or recycle 70% of landfilled waste",
                          0.7 * g("waste_landfill") * (factor("waste_landfill") - factor("waste_recycled")), 1))
    if g("clothing") or g("electronics"):
        out.append(Action("buy_less", "Halve new clothing and electronics purchases",
                          0.5 * (g("clothing") * factor("clothing") + g("electronics") * factor("electronics")), 2))
    return [a for a in out if a.saving_kg > 1e-9]
