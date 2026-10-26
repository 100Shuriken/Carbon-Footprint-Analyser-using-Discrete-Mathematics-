"""Build the interactive web report.

Every number on the page is computed here by the same tested modules the CLI
uses. The results are serialised to JSON and injected into
``web_template.html``, so the page itself only displays results; it does no
analysis of its own (apart from re-evaluating the recurrence's closed form
when the viewer moves a slider).

    python -m carbon_analyser web                 # writes docs/index.html
    python -m carbon_analyser web -o report.html
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from . import combinatorics as comb
from . import graphs, logic, relations, set_theory
from .calculator import BENCHMARKS_T, Profile, activity_emissions, category_emissions, total_emissions
from .emission_factors import ACTIVITIES, CATEGORIES

TEMPLATE = Path(__file__).with_name("web_template.html")
MAX_BUDGET = 20
TARGET_KG = BENCHMARKS_T["1.5C-compatible target (2030)"] * 1000

LAWS = [
    ("De Morgan", "~(a & b) <-> (~a | ~b)"),
    ("Contrapositive", "(a -> b) <-> (~b -> ~a)"),
    ("Converse (not a law)", "(a -> b) <-> (b -> a)"),
    ("Modus ponens", "(a & (a -> b)) -> b"),
    ("Hypothetical syllogism", "((a -> b) & (b -> c)) -> (a -> c)"),
    ("Implication as disjunction", "(a -> b) <-> (~a | b)"),
    ("Excluded middle", "a | ~a"),
    ("Contradiction", "a & ~a"),
]


def _round(x: float, nd: int = 2) -> float:
    return round(x, nd)


def _truth_table(formula: logic.Formula) -> dict:
    vs = logic.variables(formula)
    rows = [[*(env[v] for v in vs), val] for env, val in logic.truth_table(formula, vs)]
    return {"vars": vs, "rows": rows}


def _profile_data(p: Profile) -> dict:
    total = total_emissions(p)
    facts = logic.derive_facts(p.quantities)
    actions = comb.candidate_actions(p.quantities)
    plans = []
    for budget in range(MAX_BUDGET + 1):
        plan = comb.best_plan(actions, budget, comb.DIET_CONFLICTS)
        plans.append({"keys": [a.key for a in plan.actions], "saving": _round(plan.saving_kg), "effort": plan.effort})
    return {
        "name": p.name,
        "total": _round(total),
        "categories": {c: _round(v) for c, v in category_emissions(p).items()},
        "activities": sorted(
            (
                {"key": a, "name": ACTIVITIES[a].name, "category": ACTIVITIES[a].category,
                 "quantity": p.quantities[a], "unit": ACTIVITIES[a].unit, "kg": _round(v)}
                for a, v in activity_emissions(p).items()
            ),
            key=lambda r: -r["kg"],
        ),
        "support": sorted(p.support),
        "facts": facts,
        "fired": [i for i, r in enumerate(logic.DEFAULT_RULES) if r.formula.eval(facts)],
        "actions": [{"key": a.key, "description": a.description, "saving": _round(a.saving_kg), "effort": a.effort}
                    for a in actions],
        "plan_counts": {
            "n": len(actions),
            "subsets": comb.total_plans(len(actions)),
            "valid": comb.count_valid_plans(actions, comb.DIET_CONFLICTS),
        },
        "plans": plans,
    }


def _hasse(profiles: list[Profile]) -> dict:
    names = [p.name for p in profiles]
    vectors = {p.name: category_emissions(p) for p in profiles}
    r = relations.dominance_relation(vectors)
    edges = relations.hasse_edges(r, names)
    level: dict[str, int] = {}

    def lvl(x: str) -> int:
        if x not in level:
            below = [a for a, b in edges if b == x]
            level[x] = 0 if not below else 1 + max(lvl(a) for a in below)
        return level[x]

    for n in names:
        lvl(n)
    incomparable = [[x, y] for x, y in combinations(names, 2) if (x, y) not in r and (y, x) not in r]
    return {
        "levels": level,
        "edges": sorted([a, b] for a, b in edges),
        "relation": sorted([a, b] for a, b in r),
        "minimal": sorted(relations.minimal_elements(r, names)),
        "maximal": sorted(relations.maximal_elements(r, names)),
        "incomparable": incomparable,
        "properties": {
            "reflexive": relations.is_reflexive(r, names),
            "antisymmetric": relations.is_antisymmetric(r),
            "transitive": relations.is_transitive(r),
            "partial_order": relations.is_partial_order(r, names),
            "total_order": relations.is_total_order(r, names),
        },
    }


def _route_json(route: graphs.Route | None) -> dict | None:
    if route is None:
        return None
    return {
        "nodes": route.nodes,
        "legs": [{"mode": e.mode, "km": e.distance_km, "kg": _round(e.emissions)} for e in route.edges],
        "km": route.distance_km,
        "kg": _round(route.emissions),
    }


def _network(path: Path) -> dict:
    raw = json.loads(path.read_text())
    g = graphs.TransportGraph.from_json(path)
    modes = sorted({e.mode for e in g.edges})
    routes: dict[str, dict] = {}
    for k in range(1, len(modes) + 1):
        for subset in combinations(modes, k):
            allowed = set(subset)
            key = ",".join(subset)
            for a in g.nodes:
                for b in g.nodes:
                    if a == b:
                        continue
                    routes[f"{key}|{a}|{b}"] = {
                        "greenest": _route_json(g.greenest_route(a, b, allowed)),
                        "shortest": _route_json(g.greenest_route(a, b, allowed, weight="distance")),
                        "fewest": _route_json(g.fewest_legs(a, b, allowed)),
                    }
    mst = g.minimum_spanning_tree()
    nodes_fw, matrix = g.all_pairs_emissions()
    return {
        "nodes": g.nodes,
        "positions": raw.get("positions", {}),
        "edges": [{"u": e.u, "v": e.v, "mode": e.mode, "km": e.distance_km, "kg": _round(e.emissions)} for e in g.edges],
        "modes": modes,
        "routes": routes,
        "mst": [{"u": e.u, "v": e.v, "mode": e.mode, "km": e.distance_km, "kg": _round(e.emissions)} for e in mst],
        "floyd": {"nodes": nodes_fw, "matrix": [[_round(x, 1) for x in row] for row in matrix]},
        "degrees": {n: g.degree(n) for n in g.nodes},
        "handshake": g.handshake_holds(),
        "connected": g.is_connected(),
    }


def build_data(profiles_path: str | Path, network_path: str | Path) -> dict:
    profiles = Profile.load(profiles_path)
    names = [p.name for p in profiles]
    supports = {p.name: set(p.support) for p in profiles}

    factors = {k: a.factor for k, a in ACTIVITIES.items()}
    band_rel = relations.same_band_relation(factors)
    order = [label for label, _, _ in relations.EMISSION_BANDS]
    classes = sorted(
        relations.equivalence_classes(band_rel, sorted(factors)),
        key=lambda c: order.index(relations.band_of(factors[next(iter(c))])),
    )

    union_all = set_theory.union(*supports.values())
    rules = [{"condition": r.condition, "pretty": str(r.formula), "advice": r.advice,
              "table": _truth_table(r.formula)} for r in logic.DEFAULT_RULES]
    laws = []
    for name, src in LAWS:
        f = logic.parse(src)
        laws.append({
            "name": name, "pretty": str(f), "table": _truth_table(f),
            "tautology": logic.is_tautology(f), "contradiction": logic.is_contradiction(f),
            "dnf": logic.canonical_dnf(f), "cnf": logic.canonical_cnf(f),
        })

    return {
        "categories": list(CATEGORIES),
        "activities": {k: {"name": a.name, "category": a.category, "unit": a.unit, "factor": a.factor}
                       for k, a in ACTIVITIES.items()},
        "bands": [{"band": relations.band_of(factors[next(iter(c))]), "members": sorted(c)} for c in classes],
        "band_is_equivalence": relations.is_equivalence(band_rel, factors),
        "categories_partition": set_theory.is_partition(
            [{k for k, a in ACTIVITIES.items() if a.category == c} for c in CATEGORIES], ACTIVITIES),
        "benchmarks": BENCHMARKS_T,
        "target_kg": TARGET_KG,
        "profiles": [_profile_data(p) for p in profiles],
        "sets": {
            "intersection": sorted(set_theory.intersection(*supports.values())),
            "union_size": len(union_all),
            "union_size_ie": set_theory.inclusion_exclusion_union_size(list(supports.values())),
            "never": sorted(set_theory.complement(union_all, ACTIVITIES)),
            "jaccard": [[_round(set_theory.jaccard(supports[a], supports[b])) for b in names] for a in names],
        },
        "hasse": _hasse(profiles),
        "rules": rules,
        "laws": laws,
        "network": _network(Path(network_path)),
    }


STANDALONE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>*,*::before,*::after{box-sizing:border-box}body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>
</head>
<body>
"""
STANDALONE_TAIL = "\n</body>\n</html>\n"


def render(data: dict, standalone: bool = True) -> str:
    """Return the page with ``data`` injected.

    ``standalone`` wraps the fragment in a full HTML document so it can be
    opened directly from disk or served by GitHub Pages.
    """
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text().replace("/*__DATA__*/null", payload, 1)
    return STANDALONE_HEAD + page + STANDALONE_TAIL if standalone else page


def write(out: str | Path, profiles_path: str | Path, network_path: str | Path, standalone: bool = True) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(build_data(profiles_path, network_path), standalone))
    return out
