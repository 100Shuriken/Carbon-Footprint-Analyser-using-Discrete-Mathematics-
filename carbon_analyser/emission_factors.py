"""Emission factors: the function  f : Activity -> R>=0  (kg CO2e per unit).

The values are rounded, representative averages compiled from public sources
(UK DEFRA/DESNZ conversion factors, IPCC AR6, India CEA grid baseline, Our World
in Data food LCA tables). They are good enough for a teaching project, but they
are *not* authoritative for any particular country or year.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Activity:
    key: str
    name: str
    category: str
    unit: str
    factor: float  # kg CO2e per unit


# Categories form a partition of the activity set (see set_theory.partition_by).
CATEGORIES = ("transport", "energy", "food", "waste", "goods")

_ACTIVITIES = [
    # --- transport (per passenger-km) ---
    Activity("car_petrol", "Petrol car", "transport", "km", 0.192),
    Activity("car_diesel", "Diesel car", "transport", "km", 0.171),
    Activity("car_ev", "Electric car", "transport", "km", 0.053),
    Activity("motorbike", "Motorbike", "transport", "km", 0.114),
    Activity("bus", "Bus", "transport", "km", 0.105),
    Activity("train", "Train / metro", "transport", "km", 0.041),
    Activity("flight_short", "Short-haul flight", "transport", "km", 0.246),
    Activity("flight_long", "Long-haul flight", "transport", "km", 0.148),
    Activity("bicycle", "Bicycle", "transport", "km", 0.0),
    Activity("walk", "Walking", "transport", "km", 0.0),
    # --- household energy ---
    Activity("electricity", "Grid electricity", "energy", "kWh", 0.71),
    Activity("solar", "Rooftop solar electricity", "energy", "kWh", 0.041),
    Activity("lpg", "LPG cooking gas", "energy", "kg", 2.98),
    Activity("natural_gas", "Piped natural gas", "energy", "m3", 2.03),
    # --- food (per day of diet) ---
    Activity("diet_heavy_meat", "High-meat diet", "food", "day", 7.2),
    Activity("diet_average", "Average mixed diet", "food", "day", 5.6),
    Activity("diet_vegetarian", "Vegetarian diet", "food", "day", 3.8),
    Activity("diet_vegan", "Vegan diet", "food", "day", 2.9),
    # --- waste (per kg) ---
    Activity("waste_landfill", "Landfilled waste", "waste", "kg", 0.58),
    Activity("waste_recycled", "Recycled waste", "waste", "kg", 0.02),
    Activity("waste_compost", "Composted waste", "waste", "kg", 0.01),
    # --- goods (per unit spent / item) ---
    Activity("clothing", "New clothing item", "goods", "item", 10.0),
    Activity("electronics", "New small electronic device", "goods", "item", 70.0),
]

ACTIVITIES: dict[str, Activity] = {a.key: a for a in _ACTIVITIES}


def factor(key: str) -> float:
    """The emission function f(a). Raises KeyError for activities outside the domain."""
    return ACTIVITIES[key].factor


def activities_in(category: str) -> set[str]:
    return {a.key for a in _ACTIVITIES if a.category == category}
