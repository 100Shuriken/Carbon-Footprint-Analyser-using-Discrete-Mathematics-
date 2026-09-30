"""Recurrence relations: projecting emissions year by year.

The model is a first-order linear non-homogeneous recurrence

    E_0 = initial footprint
    E_n = a · E_{n-1} + b,         a = 1 - r

where r is the fraction of emissions cut each year and b is a constant yearly
addition (new appliances, or lifestyle creep; it can be negative for offsets).

Its closed form, found with the characteristic-root method, is

    E_n = a^n · E_0 + b · (1 - a^n) / (1 - a)      if a != 1
    E_n = E_0 + n · b                              if a == 1

The fixed point (the long-run footprint) is E* = b / (1 - a) when |a| < 1.
Cumulative emissions over the first n years are S_n = sum_{k=0}^{n-1} E_k.
"""

from __future__ import annotations

import math


def iterate(e0: float, rate: float, add: float, years: int) -> list[float]:
    """Direct iteration: [E_0, E_1, ..., E_years]."""
    a = 1 - rate
    seq = [e0]
    for _ in range(years):
        seq.append(a * seq[-1] + add)
    return seq


def closed_form(e0: float, rate: float, add: float, n: int) -> float:
    a = 1 - rate
    if math.isclose(a, 1.0):
        return e0 + n * add
    return a ** n * e0 + add * (1 - a ** n) / (1 - a)


def fixed_point(rate: float, add: float) -> float | None:
    """Long-run limit, if the sequence converges (0 < rate < 2)."""
    a = 1 - rate
    if abs(a) >= 1:
        return None
    return add / (1 - a)


def cumulative(e0: float, rate: float, add: float, years: int) -> float:
    """Total emitted over years 0 .. years-1."""
    return sum(iterate(e0, rate, add, years - 1)) if years > 0 else 0.0


def years_to_reach(e0: float, rate: float, add: float, target: float, max_years: int = 500) -> int | None:
    """Smallest n with E_n <= target, or None if the target is never reached."""
    if e0 <= target:
        return 0
    fp = fixed_point(rate, add)
    if fp is not None and fp > target and 0 < rate <= 1:
        return None  # moves monotonically towards a limit above the target
    a = 1 - rate
    e = e0
    for n in range(1, max_years + 1):
        e = a * e + add
        if e <= target:
            return n
    return None


def required_rate(e0: float, target: float, years: int) -> float:
    """Constant yearly cut r such that (1-r)^years · E_0 = target (with b = 0)."""
    if years <= 0:
        raise ValueError("years must be positive")
    if target <= 0 or e0 <= 0:
        raise ValueError("footprints must be positive")
    return 1 - (target / e0) ** (1 / years)
