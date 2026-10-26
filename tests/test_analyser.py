import io
import itertools
import random
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from carbon_analyser import combinatorics as comb
from carbon_analyser import graphs, logic, recurrence, relations, set_theory
from carbon_analyser.calculator import Profile, category_emissions, total_emissions
from carbon_analyser.cli import main
from carbon_analyser.emission_factors import ACTIVITIES, CATEGORIES, activities_in

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


class TestCalculator(unittest.TestCase):
    def test_total_is_sum_of_factor_times_quantity(self):
        p = Profile("x", {"car_petrol": 1000, "diet_vegan": 365})
        self.assertAlmostEqual(total_emissions(p), 1000 * 0.192 + 365 * 2.9)

    def test_categories_sum_to_total(self):
        for p in Profile.load(EXAMPLES / "profiles.json"):
            self.assertAlmostEqual(sum(category_emissions(p).values()), total_emissions(p))

    def test_rejects_unknown_and_negative(self):
        with self.assertRaises(ValueError):
            Profile("x", {"teleport": 5})
        with self.assertRaises(ValueError):
            Profile("x", {"bus": -1})

    def test_support_excludes_zero_quantities(self):
        self.assertEqual(Profile("x", {"bus": 0, "train": 5}).support, {"train"})


class TestSets(unittest.TestCase):
    def test_categories_partition_activities(self):
        blocks = [activities_in(c) for c in CATEGORIES]
        self.assertTrue(set_theory.is_partition(blocks, ACTIVITIES))

    def test_is_partition_detects_overlap_and_gaps(self):
        self.assertFalse(set_theory.is_partition([{1, 2}, {2, 3}], {1, 2, 3}))
        self.assertFalse(set_theory.is_partition([{1}], {1, 2}))

    def test_power_set_size(self):
        self.assertEqual(len(set_theory.power_set("abcd")), 16)

    def test_jaccard(self):
        self.assertEqual(set_theory.jaccard({1, 2}, {2, 3}), 1 / 3)
        self.assertEqual(set_theory.jaccard(set(), set()), 1.0)

    def test_inclusion_exclusion_matches_union(self):
        rng = random.Random(0)
        for _ in range(50):
            sets = [set(rng.sample(range(12), rng.randint(0, 8))) for _ in range(rng.randint(1, 5))]
            self.assertEqual(set_theory.inclusion_exclusion_union_size(sets), len(set_theory.union(*sets)))

    def test_de_morgan(self):
        u = set(range(10))
        a, b = {1, 2, 3}, {3, 4, 5}
        self.assertEqual(set_theory.complement(a | b, u), set_theory.complement(a, u) & set_theory.complement(b, u))


class TestRelations(unittest.TestCase):
    def test_same_band_is_equivalence(self):
        factors = {k: a.factor for k, a in ACTIVITIES.items()}
        r = relations.same_band_relation(factors)
        self.assertTrue(relations.is_equivalence(r, factors))
        classes = relations.equivalence_classes(r, factors)
        self.assertTrue(set_theory.is_partition(classes, factors))

    def test_dominance_is_partial_not_total(self):
        vecs = {"a": {"x": 1, "y": 1}, "b": {"x": 2, "y": 0}, "c": {"x": 3, "y": 3}}
        r = relations.dominance_relation(vecs)
        self.assertTrue(relations.is_partial_order(r, vecs))
        self.assertFalse(relations.is_total_order(r, vecs))
        self.assertEqual(relations.minimal_elements(r, vecs), {"a", "b"})
        self.assertEqual(relations.maximal_elements(r, vecs), {"c"})
        self.assertEqual(relations.hasse_edges(r, vecs), {("a", "c"), ("b", "c")})

    def test_hasse_removes_transitive_edges(self):
        u = [1, 2, 3, 4]
        r = relations.relation_from_predicate(u, lambda a, b: a <= b)
        self.assertEqual(relations.hasse_edges(r, u), {(1, 2), (2, 3), (3, 4)})
        self.assertTrue(relations.is_total_order(r, u))

    def test_transitive_closure(self):
        r = {(1, 2), (2, 3), (3, 4)}
        c = relations.transitive_closure(r)
        self.assertIn((1, 4), c)
        self.assertTrue(relations.is_transitive(c))
        self.assertNotIn((4, 1), c)

    def test_property_checks(self):
        self.assertFalse(relations.is_symmetric({(1, 2)}))
        self.assertFalse(relations.is_antisymmetric({(1, 2), (2, 1)}))
        self.assertFalse(relations.is_transitive({(1, 2), (2, 3)}))


class TestGraphs(unittest.TestCase):
    def setUp(self):
        self.g = graphs.TransportGraph.from_json(EXAMPLES / "network.json")

    def test_basic_properties(self):
        self.assertTrue(self.g.is_connected())
        self.assertTrue(self.g.handshake_holds())

    def test_greenest_route_prefers_train_over_flight(self):
        r = self.g.greenest_route("Mumbai", "Goa")
        self.assertEqual(r.nodes, ["Mumbai", "Ratnagiri", "Goa"])
        self.assertTrue(all(e.mode == "train" for e in r.edges))
        direct_flight = 440 * ACTIVITIES["flight_short"].factor
        self.assertLess(r.emissions, direct_flight)

    def test_dijkstra_matches_floyd_warshall(self):
        nodes, d = self.g.all_pairs_emissions()
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                self.assertAlmostEqual(self.g.greenest_route(a, b).emissions, d[i][j])

    def test_dijkstra_is_optimal_against_brute_force(self):
        # enumerate every simple path on a small random graph
        rng = random.Random(1)
        g = graphs.TransportGraph()
        names = list("ABCDEF")
        for n in names:
            g.add_node(n)
        modes = ["car_petrol", "bus", "train", "flight_short"]
        for a, b in itertools.combinations(names, 2):
            if rng.random() < 0.6:
                g.add_edge(a, b, rng.choice(modes), rng.randint(10, 300))

        def best(u, target, visited):
            if u == target:
                return 0.0
            out = float("inf")
            for e in g.adj[u]:
                w = e.other(u)
                if w not in visited:
                    out = min(out, e.emissions + best(w, target, visited | {w}))
            return out

        for a, b in itertools.permutations(names, 2):
            r = g.greenest_route(a, b)
            expected = best(a, b, {a})
            if r is None:
                self.assertEqual(expected, float("inf"))
            else:
                self.assertAlmostEqual(r.emissions, expected)

    def test_mode_filter(self):
        r = self.g.greenest_route("Mumbai", "Goa", allowed_modes={"car_petrol"})
        self.assertEqual([e.mode for e in r.edges], ["car_petrol"])
        self.assertIsNone(self.g.greenest_route("Mumbai", "Goa", allowed_modes={"bicycle"}))

    def test_bfs_fewest_legs(self):
        self.assertEqual(len(self.g.fewest_legs("Mumbai", "Goa").edges), 1)
        self.assertEqual(self.g.fewest_legs("Pune", "Pune").edges, [])

    def test_mst_is_spanning_tree(self):
        mst = self.g.minimum_spanning_tree()
        self.assertEqual(len(mst), len(self.g.nodes) - 1)
        sub = graphs.TransportGraph()
        for n in self.g.nodes:
            sub.add_node(n)
        for e in mst:
            sub.add_edge(e.u, e.v, e.mode, e.distance_km)
        self.assertTrue(sub.is_connected())

    def test_unknown_node(self):
        with self.assertRaises(KeyError):
            self.g.greenest_route("Mumbai", "Atlantis")


class TestLogic(unittest.TestCase):
    def test_precedence_and_associativity(self):
        self.assertEqual(str(logic.parse("a | b & c")), "a ∨ (b ∧ c)")
        self.assertEqual(str(logic.parse("a -> b -> c")), "a → (b → c)")
        self.assertEqual(str(logic.parse("not a and b")), "¬a ∧ b")

    def test_tautologies(self):
        for f in ["a | ~a", "(a -> b) <-> (~a | b)", "~(a & b) <-> (~a | ~b)",
                  "((a -> b) & (b -> c)) -> (a -> c)", "(a & (a -> b)) -> b"]:
            self.assertTrue(logic.is_tautology(f), f)
        self.assertTrue(logic.is_contradiction("a & ~a"))
        self.assertFalse(logic.is_tautology("a -> b"))

    def test_equivalence(self):
        self.assertTrue(logic.equivalent("a -> b", "~b -> ~a"))  # contrapositive
        self.assertFalse(logic.equivalent("a -> b", "b -> a"))   # converse

    def test_normal_forms_are_equivalent(self):
        for f in ["a -> b", "(a | b) & ~c", "a <-> b"]:
            dnf = logic.canonical_dnf(f).replace("¬", "~").replace("∧", "&").replace("∨", "|")
            cnf = logic.canonical_cnf(f).replace("¬", "~").replace("∧", "&").replace("∨", "|")
            self.assertTrue(logic.equivalent(f, dnf))
            self.assertTrue(logic.equivalent(f, cnf))

    def test_truth_table_rows(self):
        self.assertEqual(len(logic.truth_table("a & b & c")), 8)

    def test_syntax_errors(self):
        for bad in ["a &", "(a", "a b", "a $ b", ""]:
            with self.assertRaises(SyntaxError):
                logic.parse(bad)

    def test_advisor(self):
        asha = Profile.load(EXAMPLES / "profiles.json")[0]
        facts = logic.derive_facts(asha.quantities)
        self.assertTrue(facts["drives_a_lot"])
        self.assertTrue(facts["heavy_meat_diet"])
        advice = [r.condition for r in logic.advise(facts)]
        self.assertIn("heavy_meat_diet", advice)
        self.assertNotIn("plant_based & (uses_public_transport | cycles_or_walks) & recycles", advice)

    def test_all_default_rules_reference_known_facts(self):
        facts = logic.derive_facts({})
        for r in logic.DEFAULT_RULES:
            self.assertTrue(set(logic.variables(r.formula)) <= set(facts), r.condition)


class TestCombinatorics(unittest.TestCase):
    def test_counting(self):
        self.assertEqual(comb.n_choose_k(9, 3), 84)
        self.assertEqual(sum(comb.n_choose_k(9, k) for k in range(10)), comb.total_plans(9))
        self.assertEqual(comb.n_permute_k(5, 2), 20)
        self.assertEqual(comb.pigeonhole_min_share(13, 12), 2)

    def test_valid_plan_count_with_one_conflict(self):
        acts = [comb.Action(k, k, 1, 1) for k in ("go_vegetarian", "go_vegan", "x")]
        # 2^3 total minus the 2 subsets that contain both conflicting actions
        self.assertEqual(comb.count_valid_plans(acts, comb.DIET_CONFLICTS), 6)

    def test_exhaustive_matches_dp_without_conflicts(self):
        rng = random.Random(2)
        for _ in range(30):
            acts = [comb.Action(f"a{i}", "", rng.randint(1, 500), rng.randint(1, 6)) for i in range(rng.randint(1, 10))]
            budget = rng.randint(0, 20)
            self.assertAlmostEqual(comb.best_plan(acts, budget).saving_kg, comb.best_plan_dp(acts, budget).saving_kg)

    def test_best_plan_respects_budget_and_conflicts(self):
        q = Profile.load(EXAMPLES / "profiles.json")[0].quantities
        acts = comb.candidate_actions(q)
        for budget in range(0, 25):
            plan = comb.best_plan(acts, budget, comb.DIET_CONFLICTS)
            keys = {a.key for a in plan.actions}
            self.assertLessEqual(plan.effort, budget)
            self.assertFalse({"go_vegetarian", "go_vegan"} <= keys)

    def test_candidate_actions_positive(self):
        for p in Profile.load(EXAMPLES / "profiles.json"):
            for a in comb.candidate_actions(p.quantities):
                self.assertGreater(a.saving_kg, 0)
                self.assertLess(a.saving_kg, total_emissions(p))


class TestRecurrence(unittest.TestCase):
    def test_closed_form_matches_iteration(self):
        for rate, add in [(0.1, 50), (0.0, 20), (0.3, -10), (-0.05, 0)]:
            seq = recurrence.iterate(4000, rate, add, 15)
            for n, e in enumerate(seq):
                self.assertAlmostEqual(recurrence.closed_form(4000, rate, add, n), e, places=6)

    def test_fixed_point(self):
        self.assertAlmostEqual(recurrence.fixed_point(0.1, 50), 500)
        self.assertIsNone(recurrence.fixed_point(0.0, 50))
        self.assertAlmostEqual(recurrence.iterate(4000, 0.1, 50, 400)[-1], 500)

    def test_years_to_reach(self):
        self.assertEqual(recurrence.years_to_reach(4200, 0.1, 50, 2300), 7)
        self.assertIsNone(recurrence.years_to_reach(4200, 0.1, 500, 2300))  # limit is 5000
        self.assertEqual(recurrence.years_to_reach(1000, 0.1, 0, 2000), 0)
        self.assertIsNone(recurrence.years_to_reach(1000, 0.0, 0, 500))

    def test_required_rate(self):
        r = recurrence.required_rate(8000, 2000, 5)
        self.assertAlmostEqual(recurrence.closed_form(8000, r, 0, 5), 2000)

    def test_cumulative(self):
        self.assertAlmostEqual(recurrence.cumulative(100, 0.5, 0, 3), 100 + 50 + 25)


class TestWeb(unittest.TestCase):
    def test_data_matches_core_modules(self):
        from carbon_analyser import web
        d = web.build_data(EXAMPLES / "profiles.json", EXAMPLES / "network.json")
        profiles = Profile.load(EXAMPLES / "profiles.json")
        for p, pd in zip(profiles, d["profiles"]):
            self.assertAlmostEqual(pd["total"], total_emissions(p), places=1)
            self.assertEqual(len(pd["plans"]), web.MAX_BUDGET + 1)
        g = graphs.TransportGraph.from_json(EXAMPLES / "network.json")
        r = d["network"]["routes"][",".join(sorted(d["network"]["modes"])) + "|Mumbai|Goa"]["greenest"]
        self.assertEqual(r["nodes"], g.greenest_route("Mumbai", "Goa").nodes)
        self.assertTrue(d["hasse"]["properties"]["partial_order"])

    def test_render_injects_data(self):
        from carbon_analyser import web
        html = web.render({"x": "</script>"}, standalone=True)
        self.assertNotIn("/*__DATA__*/null", html)
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertNotIn('"</script>"', html)


class TestCLI(unittest.TestCase):
    def run_cli(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(list(argv))
        return code, buf.getvalue()

    def test_demo_runs(self):
        code, out = self.run_cli("demo")
        self.assertEqual(code, 0)
        for s in ("Hasse diagram", "Minimum spanning tree", "canonical DNF", "fixed point"):
            self.assertIn(s, out)

    def test_commands(self):
        self.assertEqual(self.run_cli("analyse", str(EXAMPLES / "profiles.json"), "--name", "zoya")[0], 0)
        self.assertEqual(self.run_cli("route", str(EXAMPLES / "network.json"), "Pune", "Goa", "--modes", "train,bus")[0], 0)
        self.assertEqual(self.run_cli("logic", "a -> b", "--equiv", "~b -> ~a")[0], 0)

    def test_bad_input_returns_error(self):
        import contextlib
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.run_cli("logic", "a &&")[0], 1)
            self.assertEqual(self.run_cli("route", str(EXAMPLES / "network.json"), "Pune", "Nowhere")[0], 1)


if __name__ == "__main__":
    unittest.main()
