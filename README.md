# Carbon Footprint Analyser using Discrete Mathematics

A command-line tool that estimates a person's yearly carbon footprint and
suggests ways to reduce it. Every part of the analysis uses a topic from a
Discrete Mathematics course.

It is written in pure Python (3.10+). It needs no third-party packages.

```bash
python -m carbon_analyser demo            # run everything on the bundled examples
python -m unittest discover -s tests      # run the 44 tests
```

## What it does, and which maths it uses

| Feature | Discrete-maths concept | Module |
|---|---|---|
| Emission factor per activity, footprint = Σ f(a)·q(a) | **Functions** (domain, codomain, composition) | `emission_factors.py`, `calculator.py` |
| Activities split into transport / energy / food / waste / goods | **Partitions** of a set | `set_theory.py` |
| Compare what different people do: ∩, ∪, \, Δ, complement, Jaccard similarity | **Set operations**, **inclusion–exclusion** | `set_theory.py` |
| Group activities into zero / low / medium / high emission bands | **Equivalence relations** and equivalence classes | `relations.py` |
| "P is greener than Q in every category" | **Partial orders** (product order), Hasse diagram, minimal/maximal elements, incomparable elements | `relations.py` |
| Reflexive / symmetric / antisymmetric / transitive checks, transitive closure | **Relation properties**, Warshall's algorithm | `relations.py` |
| Lowest-carbon route between cities | **Weighted graphs**, Dijkstra's algorithm | `graphs.py` |
| Fewest-legs route; connectivity | **BFS** | `graphs.py` |
| All-pairs lowest-carbon matrix | **Floyd–Warshall**, adjacency matrix | `graphs.py` |
| Cheapest-carbon network linking all cities | **Minimum spanning tree** (Kruskal + union–find) | `graphs.py` |
| Degree sequence, Σ deg = 2\|E\| | **Handshaking lemma** | `graphs.py` |
| Rule-based recommendations like `drives_a_lot ∧ ¬uses_ev → "consider an EV"` | **Propositional logic**, modus ponens | `logic.py` |
| Formula parser, truth tables, tautology/equivalence checks, canonical DNF/CNF | **Truth tables**, **normal forms** | `logic.py` |
| Choose the best set of reduction actions within an effort budget | **Power set** (2ⁿ subsets as bitmasks), exhaustive search, 0/1 knapsack DP | `combinatorics.py` |
| Count plans, conflict-free plans, roll-out orders | **Combinations** C(n,k), **permutations** k!, pigeonhole principle | `combinatorics.py` |
| Project emissions year by year, years to reach a target | **Recurrence relations** Eₙ = a·Eₙ₋₁ + b, closed form, fixed point | `recurrence.py` |

## Commands

```bash
# Emission factors and the emission-band equivalence classes
python -m carbon_analyser factors

# Full analysis of one or more people: breakdown, benchmarks, logic advice,
# best reduction plan (combinatorics) and 10-year projection (recurrence)
python -m carbon_analyser analyse examples/profiles.json --name Asha --budget 8

# Compare everyone: set operations, Jaccard matrix, dominance poset and Hasse diagram
python -m carbon_analyser compare examples/profiles.json

# Low-carbon route vs shortest route vs fewest legs, plus minimum spanning tree
python -m carbon_analyser route examples/network.json Mumbai Goa
python -m carbon_analyser route examples/network.json Pune Aurangabad --modes train,bus

# Any propositional formula: truth table, tautology check, DNF/CNF, equivalence
python -m carbon_analyser logic "(a -> b) & (b -> c) -> (a -> c)"
python -m carbon_analyser logic "a -> b" --equiv "~b -> ~a"

# Recurrence: E_n = (1 - rate)·E_{n-1} + add, closed form checked against iteration
python -m carbon_analyser project --e0 4200 --rate 0.1 --add 50 --years 10 --target 2300
```

Logic syntax: `~` / `!` / `not`, `&` / `and`, `|` / `or`, `->`, `<->`,
parentheses, and the constants `T` and `F`. Precedence runs from highest to
lowest: ¬, ∧, ∨, →, ↔. `→` is right-associative.

## Sample output (excerpt)

```
Dominance partial order  P ≼ Q ⇔ ∀c: P[c] ≤ Q[c]
  reflexive=True  antisymmetric=True  transitive=True  → partial order: True
  total order: False
  Incomparable pairs: Asha∥Kabir, Kabir∥Rohan, Meera∥Zoya
  Hasse diagram (covering edges, lower ⋖ higher):
    Meera ⋖ Kabir
    Meera ⋖ Rohan
    Rohan ⋖ Asha
    Zoya ⋖ Kabir
    Zoya ⋖ Rohan
  Minimal (greenest, not dominated): {Meera, Zoya}

  Lowest-carbon (Dijkstra, w = km·f(mode)):
    Mumbai --train 330km--> Ratnagiri --train 250km--> Goa
    580 km, 24 kg CO2e
  Shortest distance (Dijkstra, w = km):
    Mumbai --flight_short 440km--> Goa
    440 km, 108 kg CO2e
```

The greenest route is longer than the direct flight but emits about 4.5 times
less. The dominance order is not total: Asha drives and eats more meat, while
Kabir flies long-haul. Neither dominates the other, and the Hasse diagram
shows this.

## Input formats

**Profiles** (`examples/profiles.json`): a list (or a single object) of
`{"name": ..., "quantities": {activity_key: yearly_amount}}`. Run
`python -m carbon_analyser factors` for the valid activity keys and their units
(km, kWh, kg, m³, days, items).

**Transport network** (`examples/network.json`): `{"nodes": [...], "edges":
[{"from", "to", "mode", "km"}]}`. It can hold parallel edges, one per
transport mode. Each edge's weight is `km × f(mode)`.

## How the tests check the maths

The tests compare results computed in two independent ways:

- Dijkstra agrees with Floyd–Warshall and with brute-force enumeration of all simple paths.
- The exhaustive 2ⁿ subset search agrees with the knapsack dynamic programme.
- Inclusion–exclusion agrees with a directly computed union.
- The recurrence's closed form agrees with direct iteration.
- The canonical DNF and CNF are each equivalent to the original formula.
- Known tautologies are recognised: De Morgan, hypothetical syllogism, modus ponens and the contrapositive.

## Project layout

```
carbon_analyser/
  emission_factors.py   activity set A and emission function f
  calculator.py         profiles and footprint totals
  set_theory.py         set operations, partitions, inclusion–exclusion
  relations.py          relation properties, equivalence classes, posets, Hasse diagrams
  graphs.py             transport graph, Dijkstra, BFS, Floyd–Warshall, Kruskal
  logic.py              formula parser, truth tables, normal forms, rule-based advisor
  combinatorics.py      counting, plan optimisation over the power set
  recurrence.py         emission projections with linear recurrences
  cli.py                command-line interface
examples/               sample profiles and an inter-city network
tests/                  unit tests
```

## Data disclaimer

Emission factors are rounded, representative values from public sources:
UK DESNZ/DEFRA conversion factors, the India CEA grid baseline, and Our World
in Data food life-cycle tables. They suit a teaching project, but they are not
an audited carbon inventory. The inter-city distances are approximate.
