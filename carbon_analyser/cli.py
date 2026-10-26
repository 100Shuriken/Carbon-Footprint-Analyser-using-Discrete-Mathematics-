"""Command-line interface.

    python -m carbon_analyser demo
    python -m carbon_analyser analyse examples/profiles.json [--name Asha] [--budget 8]
    python -m carbon_analyser compare examples/profiles.json
    python -m carbon_analyser route examples/network.json Mumbai Goa [--modes train,bus]
    python -m carbon_analyser logic "drives_a_lot & ~uses_ev -> T"
    python -m carbon_analyser project --e0 4200 --rate 0.1 --add 50 --years 10 --target 2300
    python -m carbon_analyser factors
    python -m carbon_analyser web [-o docs/index.html]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import combinatorics as comb
from . import graphs, logic, recurrence, relations, set_theory
from .calculator import BENCHMARKS_T, Profile, activity_emissions, category_emissions, total_emissions
from .emission_factors import ACTIVITIES, CATEGORIES

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def heading(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def bar(value: float, max_value: float, width: int = 30) -> str:
    n = 0 if max_value <= 0 else round(width * value / max_value)
    return "█" * n + "·" * (width - n)


def fmt_kg(kg: float) -> str:
    return f"{kg / 1000:,.2f} t" if kg >= 1000 else f"{kg:,.0f} kg"


def fmt_set(s) -> str:
    return "{" + ", ".join(sorted(s)) + "}" if s else "∅"


# ----------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------

def cmd_factors(_: argparse.Namespace) -> None:
    heading("Emission factors  f : A → ℝ≥0")
    factors = {k: a.factor for k, a in ACTIVITIES.items()}
    r = relations.same_band_relation(factors)
    for cat in CATEGORIES:
        print(f"\n[{cat}]")
        for a in ACTIVITIES.values():
            if a.category == cat:
                print(f"  {a.key:<18} {a.factor:>7.3f} kg CO2e/{a.unit:<5} {a.name}")
    print("\nEquivalence classes of 'same emission band' "
          f"(equivalence relation: {relations.is_equivalence(r, factors)}):")
    order = [label for label, _, _ in relations.EMISSION_BANDS]
    classes = relations.equivalence_classes(r, sorted(factors))
    for cls in sorted(classes, key=lambda c: order.index(relations.band_of(factors[next(iter(c))]))):
        band = relations.band_of(factors[next(iter(cls))])
        print(f"  [{band:>6}] {fmt_set(cls)}")


def analyse_profile(p: Profile, budget: int) -> None:
    heading(f"Footprint analysis: {p.name}")
    total = total_emissions(p)
    print(f"Total: {fmt_kg(total)} CO2e / year\n")

    cats = category_emissions(p)
    mx = max(cats.values()) if cats else 0
    for c, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        share = 100 * v / total if total else 0
        print(f"  {c:<10} {bar(v, mx)} {fmt_kg(v):>10}  ({share:4.1f}%)")

    print("\nTop activities:")
    for a, v in sorted(activity_emissions(p).items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {ACTIVITIES[a].name:<28} {fmt_kg(v):>10}")

    print("\nBenchmarks (per person per year):")
    for label, t in BENCHMARKS_T.items():
        rel = "above" if total / 1000 > t else "below"
        print(f"  {label:<32} {t:>5.1f} t   → you are {rel}")

    # --- logic ---------------------------------------------------------------
    facts = logic.derive_facts(p.quantities)
    true_props = {k for k, v in facts.items() if v}
    print(f"\nPropositions that hold (logic): {fmt_set(true_props)}")
    fired = logic.advise(facts)
    print("Recommendations (rules P → advice where P holds):")
    for r in fired:
        print(f"  • [{r.formula}]\n    {r.advice}")
    if not fired:
        print("  (no rules fired)")

    # --- combinatorics -------------------------------------------------------
    actions = comb.candidate_actions(p.quantities)
    n = len(actions)
    if n:
        valid = comb.count_valid_plans(actions, comb.DIET_CONFLICTS)
        print(f"\nReduction plan search (combinatorics): {n} candidate actions → "
              f"2^{n} = {comb.total_plans(n)} subsets, {valid} without conflicts")
        plan = comb.best_plan(actions, budget, comb.DIET_CONFLICTS)
        print(f"Best plan within effort budget {budget}:")
        for a in plan.actions:
            print(f"  ✓ {a.description:<52} −{fmt_kg(a.saving_kg):>9}  (effort {a.effort})")
        after = total - plan.saving_kg
        print(f"  → saves {fmt_kg(plan.saving_kg)} ({100 * plan.saving_kg / total:.0f}%), "
              f"new footprint {fmt_kg(after)}; {comb.rollout_orders(len(plan.actions))} possible roll-out orders")

        # --- recurrence ------------------------------------------------------
        years = 10
        rate = min(0.9, plan.saving_kg / total / 5) if total else 0  # reach the plan's saving over about 5 years
        seq = recurrence.iterate(total, rate, 0.0, years)
        print(f"\nProjection (recurrence Eₙ = {1 - rate:.3f}·Eₙ₋₁), cutting {100 * rate:.1f}% per year:")
        for i, e in enumerate(seq):
            if i in (0, 1, 2, 5, 10):
                print(f"  year {i:>2}: {fmt_kg(e):>9}  {bar(e, total, 25)}")
        target = BENCHMARKS_T["1.5C-compatible target (2030)"] * 1000
        yrs = recurrence.years_to_reach(total, rate, 0.0, target)
        print(f"  Years to reach the 2.3 t target: {yrs if yrs is not None else 'never at this rate'}")
        if total > target:
            print(f"  Rate needed to reach it within 5 years: "
                  f"{100 * recurrence.required_rate(total, target, 5):.1f}% per year")


def cmd_analyse(args: argparse.Namespace) -> None:
    profiles = Profile.load(args.file)
    if args.name:
        profiles = [p for p in profiles if p.name.lower() == args.name.lower()]
        if not profiles:
            sys.exit(f"No profile named {args.name!r}")
    for p in profiles:
        analyse_profile(p, args.budget)


def cmd_compare(args: argparse.Namespace) -> None:
    profiles = Profile.load(args.file)
    names = [p.name for p in profiles]
    supports = {p.name: p.support for p in profiles}
    universe = set(ACTIVITIES)

    heading("Set comparison of activity supports")
    for p in profiles:
        print(f"  {p.name:<8} {fmt_set(supports[p.name])}")
    shared = set_theory.intersection(*supports.values())
    print(f"\n  Common to everyone (⋂): {fmt_set(shared)}")
    print(f"  Done by anyone (⋃):     {len(set_theory.union(*supports.values()))} activities")
    print(f"  Inclusion–exclusion check: |⋃| = "
          f"{set_theory.inclusion_exclusion_union_size(list(map(set, supports.values())))}")
    print(f"  Never done by anyone (complement of ⋃): "
          f"{fmt_set(set_theory.complement(set_theory.union(*supports.values()), universe))}")
    if len(profiles) >= 2:
        a, b = names[0], names[1]
        print(f"\n  {a} \\ {b}: {fmt_set(set_theory.difference(supports[a], supports[b]))}")
        print(f"  {a} Δ {b}: {fmt_set(set_theory.symmetric_difference(supports[a], supports[b]))}")

    print("\n  Jaccard similarity matrix  |A∩B| / |A∪B|:")
    print("  " + " " * 9 + "".join(f"{n[:7]:>8}" for n in names))
    for x in names:
        print(f"  {x[:8]:<9}" + "".join(f"{set_theory.jaccard(supports[x], supports[y]):>8.2f}" for y in names))

    heading("Dominance partial order  P ≼ Q ⇔ ∀c: P[c] ≤ Q[c]")
    vectors = {p.name: category_emissions(p) for p in profiles}
    r = relations.dominance_relation(vectors)
    print(f"  reflexive={relations.is_reflexive(r, names)}  antisymmetric={relations.is_antisymmetric(r)}  "
          f"transitive={relations.is_transitive(r)}  → partial order: {relations.is_partial_order(r, names)}")
    print(f"  total order: {relations.is_total_order(r, names)}")
    incomparable = sorted({tuple(sorted((x, y))) for x in names for y in names
                           if x != y and (x, y) not in r and (y, x) not in r})
    if incomparable:
        print("  Incomparable pairs: " + ", ".join(f"{x}∥{y}" for x, y in incomparable))
    print("  Hasse diagram (covering edges, lower ⋖ higher):")
    for x, y in sorted(relations.hasse_edges(r, names)):
        print(f"    {x} ⋖ {y}")
    print(f"  Minimal (greenest, not dominated): {fmt_set(relations.minimal_elements(r, names))}")
    print(f"  Maximal (dominates others, no one above): {fmt_set(relations.maximal_elements(r, names))}")

    print("\n  Ranking by total (a total preorder):")
    for p in sorted(profiles, key=total_emissions):
        print(f"    {p.name:<8} {fmt_kg(total_emissions(p)):>9}")


def cmd_route(args: argparse.Namespace) -> None:
    g = graphs.TransportGraph.from_json(args.file)
    modes = set(args.modes.split(",")) if args.modes else None
    heading(f"Routes {args.src} → {args.dst}" + (f" using {sorted(modes)}" if modes else ""))
    print(f"  |V|={len(g.nodes)}  |E|={len(g.edges)}  connected={g.is_connected()}  "
          f"handshake lemma Σdeg=2|E|: {g.handshake_holds()}")
    print(f"  degree sequence: {g.degree_sequence()}")

    for label, route in (
        ("Lowest-carbon (Dijkstra, w = km·f(mode))", g.greenest_route(args.src, args.dst, modes)),
        ("Shortest distance (Dijkstra, w = km)", g.greenest_route(args.src, args.dst, modes, weight="distance")),
        ("Fewest legs (BFS)", g.fewest_legs(args.src, args.dst, modes)),
    ):
        print(f"\n  {label}:")
        if route is None:
            print("    no route")
            continue
        print(f"    {route.describe()}")
        print(f"    {route.distance_km:,.0f} km, {fmt_kg(route.emissions)} CO2e")

    heading("Minimum spanning tree (Kruskal): lowest-carbon network connecting all nodes")
    mst = g.minimum_spanning_tree()
    for e in mst:
        print(f"  {e.u:<11} — {e.v:<11} {e.mode:<13} {e.distance_km:>5g} km  {fmt_kg(e.emissions):>8}")
    print(f"  {len(mst)} edges (= |V| − 1 = {len(g.nodes) - 1} for a tree), "
          f"total {fmt_kg(sum(e.emissions for e in mst))}")


def cmd_logic(args: argparse.Namespace) -> None:
    f = logic.parse(args.formula)
    heading(f"Formula: {f}")
    print(logic.format_truth_table(f))
    print(f"\n  tautology: {logic.is_tautology(f)}   contradiction: {logic.is_contradiction(f)}   "
          f"satisfiable: {logic.is_satisfiable(f)}")
    print(f"  canonical DNF: {logic.canonical_dnf(f)}")
    print(f"  canonical CNF: {logic.canonical_cnf(f)}")
    if args.equiv:
        g = logic.parse(args.equiv)
        print(f"  equivalent to {g}: {logic.equivalent(f, g)}")


def cmd_project(args: argparse.Namespace) -> None:
    heading(f"Eₙ = {1 - args.rate:g}·Eₙ₋₁ + {args.add:g},  E₀ = {args.e0:g}")
    seq = recurrence.iterate(args.e0, args.rate, args.add, args.years)
    mx = max(seq)
    for n, e in enumerate(seq):
        cf = recurrence.closed_form(args.e0, args.rate, args.add, n)
        print(f"  n={n:>2}  iterate={e:>10.1f}  closed-form={cf:>10.1f}  {bar(e, mx, 25)}")
    fp = recurrence.fixed_point(args.rate, args.add)
    print(f"\n  fixed point E* = {fp:.1f}" if fp is not None else "\n  no finite fixed point")
    print(f"  cumulative over {args.years} years: {recurrence.cumulative(args.e0, args.rate, args.add, args.years):,.1f}")
    if args.target is not None:
        y = recurrence.years_to_reach(args.e0, args.rate, args.add, args.target)
        print(f"  years to reach {args.target:g}: {y if y is not None else 'never'}")


def cmd_web(args: argparse.Namespace) -> None:
    from . import web

    out = web.write(args.output, args.profiles, args.network)
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB). Open it in a browser.")


def cmd_demo(args: argparse.Namespace) -> None:
    profiles_file = EXAMPLES / "profiles.json"
    network_file = EXAMPLES / "network.json"
    cmd_factors(args)
    analyse_profile(Profile.load(profiles_file)[0], budget=8)
    cmd_compare(argparse.Namespace(file=profiles_file))
    cmd_route(argparse.Namespace(file=network_file, src="Mumbai", dst="Goa", modes=None))
    cmd_logic(argparse.Namespace(formula="(drives_a_lot -> uses_public_transport) <-> (~drives_a_lot | uses_public_transport)",
                                 equiv=None))
    cmd_project(argparse.Namespace(e0=4200, rate=0.1, add=50, years=10, target=2300))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="carbon_analyser",
                                description="Carbon Footprint Analyser using Discrete Mathematics")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="run every analysis on the bundled examples").set_defaults(func=cmd_demo)
    sub.add_parser("factors", help="list emission factors and emission bands").set_defaults(func=cmd_factors)

    a = sub.add_parser("analyse", help="analyse profile(s) from a JSON file")
    a.add_argument("file")
    a.add_argument("--name", help="analyse only this profile")
    a.add_argument("--budget", type=int, default=8, help="effort budget for the reduction plan (default 8)")
    a.set_defaults(func=cmd_analyse)

    c = sub.add_parser("compare", help="compare profiles with sets and partial orders")
    c.add_argument("file")
    c.set_defaults(func=cmd_compare)

    r = sub.add_parser("route", help="find low-carbon routes in a transport graph")
    r.add_argument("file")
    r.add_argument("src")
    r.add_argument("dst")
    r.add_argument("--modes", help="comma-separated allowed modes, e.g. train,bus")
    r.set_defaults(func=cmd_route)

    lg = sub.add_parser("logic", help="truth table and normal forms of a formula")
    lg.add_argument("formula")
    lg.add_argument("--equiv", help="check equivalence with this formula")
    lg.set_defaults(func=cmd_logic)

    w = sub.add_parser("web", help="build the interactive HTML report")
    w.add_argument("-o", "--output", default="docs/index.html", help="output file (default docs/index.html)")
    w.add_argument("--profiles", default=str(EXAMPLES / "profiles.json"))
    w.add_argument("--network", default=str(EXAMPLES / "network.json"))
    w.set_defaults(func=cmd_web)

    pr = sub.add_parser("project", help="project emissions with a recurrence relation")
    pr.add_argument("--e0", type=float, required=True, help="initial footprint (kg)")
    pr.add_argument("--rate", type=float, default=0.1, help="fraction cut per year")
    pr.add_argument("--add", type=float, default=0.0, help="constant added per year")
    pr.add_argument("--years", type=int, default=10)
    pr.add_argument("--target", type=float)
    pr.set_defaults(func=cmd_project)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (KeyError, ValueError, SyntaxError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0
