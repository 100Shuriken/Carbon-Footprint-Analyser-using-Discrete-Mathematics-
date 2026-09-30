"""Footprint calculation as a composition of functions.

A *profile* is a finite function  q : A' -> R>=0  (activity -> quantity per year)
defined on a subset A' of the activity set A. The footprint is

    F(q) = sum_{a in A'} f(a) * q(a)

where f is the emission-factor function from ``emission_factors``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .emission_factors import ACTIVITIES, CATEGORIES, factor


@dataclass
class Profile:
    name: str
    quantities: dict[str, float] = field(default_factory=dict)  # per year

    def __post_init__(self) -> None:
        unknown = set(self.quantities) - set(ACTIVITIES)
        if unknown:
            raise ValueError(f"Unknown activities (not in domain of f): {sorted(unknown)}")
        negative = [k for k, v in self.quantities.items() if v < 0]
        if negative:
            raise ValueError(f"Quantities must be non-negative: {negative}")

    @property
    def support(self) -> frozenset[str]:
        """The set of activities actually performed (q(a) > 0)."""
        return frozenset(k for k, v in self.quantities.items() if v > 0)

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        return cls(name=data.get("name", "unnamed"), quantities=dict(data["quantities"]))

    @classmethod
    def load(cls, path: str | Path) -> list["Profile"]:
        data = json.loads(Path(path).read_text())
        if isinstance(data, list):
            return [cls.from_dict(d) for d in data]
        return [cls.from_dict(data)]


def activity_emissions(profile: Profile) -> dict[str, float]:
    """(f * q)(a) for every activity in the support of q."""
    return {a: factor(a) * q for a, q in profile.quantities.items() if q > 0}


def category_emissions(profile: Profile) -> dict[str, float]:
    """Sum over each block of the category partition."""
    totals = {c: 0.0 for c in CATEGORIES}
    for a, e in activity_emissions(profile).items():
        totals[ACTIVITIES[a].category] += e
    return totals


def total_emissions(profile: Profile) -> float:
    """F(q) in kg CO2e per year."""
    return sum(activity_emissions(profile).values())


# Reference points (tonnes CO2e per person per year, rounded).
BENCHMARKS_T = {
    "1.5C-compatible target (2030)": 2.3,
    "India average": 2.0,
    "World average": 4.7,
    "EU average": 7.0,
    "USA average": 14.0,
}
