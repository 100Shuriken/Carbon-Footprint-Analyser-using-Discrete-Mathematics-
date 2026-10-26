"""Propositional logic: a formula parser, truth tables and a rule-based advisor.

Grammar, from lowest to highest precedence. ``->`` is right-associative.

    iff     := implies ( "<->" implies )*
    implies := or ( "->" implies )?
    or      := and ( ("|" | "or") and )*
    and     := not ( ("&" | "and") not )*
    not     := ("~" | "!" | "not") not | atom
    atom    := IDENT | "T" | "F" | "(" iff ")"

Recommendations are rules of the form ``condition -> advice``. The condition
is a formula over propositions that are derived from a profile, for example
``drives_a_lot & ~uses_ev``.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

# ----------------------------------------------------------------------------
# AST
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Var:
    name: str

    def eval(self, env: dict[str, bool]) -> bool:
        if self.name not in env:
            raise KeyError(f"No truth value for proposition '{self.name}'")
        return env[self.name]

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const:
    value: bool

    def eval(self, env: dict[str, bool]) -> bool:
        return self.value

    def __str__(self) -> str:
        return "T" if self.value else "F"


@dataclass(frozen=True)
class Not:
    a: "Formula"

    def eval(self, env: dict[str, bool]) -> bool:
        return not self.a.eval(env)

    def __str__(self) -> str:
        return f"¬{_wrap(self.a)}"


@dataclass(frozen=True)
class BinOp:
    op: str  # one of & | -> <->
    a: "Formula"
    b: "Formula"

    def eval(self, env: dict[str, bool]) -> bool:
        x = self.a.eval(env)
        if self.op == "&":
            return x and self.b.eval(env)
        if self.op == "|":
            return x or self.b.eval(env)
        y = self.b.eval(env)
        if self.op == "->":
            return (not x) or y
        if self.op == "<->":
            return x == y
        raise ValueError(self.op)

    def __str__(self) -> str:
        sym = {"&": "∧", "|": "∨", "->": "→", "<->": "↔"}[self.op]
        return f"{_wrap(self.a)} {sym} {_wrap(self.b)}"


Formula = Var | Const | Not | BinOp


def _wrap(f: Formula) -> str:
    return f"({f})" if isinstance(f, BinOp) else str(f)


# ----------------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(<->|->|[()&|~!]|[A-Za-z_][A-Za-z0-9_]*)")
_WORD_OPS = {"and": "&", "or": "|", "not": "~"}


def tokenize(s: str) -> list[str]:
    pos, out = 0, []
    s = s.rstrip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            raise SyntaxError(f"Unexpected character at position {pos}: {s[pos:]!r}")
        tok = m.group(1)
        out.append(_WORD_OPS.get(tok.lower(), tok))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens: list[str]):
        self.t = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self, expected: str | None = None) -> str:
        tok = self.peek()
        if tok is None or (expected is not None and tok != expected):
            raise SyntaxError(f"Expected {expected or 'token'}, got {tok!r}")
        self.i += 1
        return tok

    def parse(self) -> Formula:
        f = self.iff()
        if self.peek() is not None:
            raise SyntaxError(f"Unexpected token {self.peek()!r}")
        return f

    def iff(self) -> Formula:
        f = self.implies()
        while self.peek() == "<->":
            self.take()
            f = BinOp("<->", f, self.implies())
        return f

    def implies(self) -> Formula:
        f = self.or_()
        if self.peek() == "->":
            self.take()
            return BinOp("->", f, self.implies())
        return f

    def or_(self) -> Formula:
        f = self.and_()
        while self.peek() == "|":
            self.take()
            f = BinOp("|", f, self.and_())
        return f

    def and_(self) -> Formula:
        f = self.not_()
        while self.peek() == "&":
            self.take()
            f = BinOp("&", f, self.not_())
        return f

    def not_(self) -> Formula:
        if self.peek() in ("~", "!"):
            self.take()
            return Not(self.not_())
        return self.atom()

    def atom(self) -> Formula:
        tok = self.peek()
        if tok == "(":
            self.take("(")
            f = self.iff()
            self.take(")")
            return f
        if tok is None or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            raise SyntaxError(f"Expected proposition, got {tok!r}")
        self.take()
        if tok == "T":
            return Const(True)
        if tok == "F":
            return Const(False)
        return Var(tok)


def parse(s: str) -> Formula:
    return _Parser(tokenize(s)).parse()


# ----------------------------------------------------------------------------
# Semantics
# ----------------------------------------------------------------------------


def variables(f: Formula) -> list[str]:
    out: set[str] = set()

    def walk(g: Formula) -> None:
        if isinstance(g, Var):
            out.add(g.name)
        elif isinstance(g, Not):
            walk(g.a)
        elif isinstance(g, BinOp):
            walk(g.a)
            walk(g.b)

    walk(f)
    return sorted(out)


def truth_table(f: Formula | str, vars_: list[str] | None = None) -> list[tuple[dict[str, bool], bool]]:
    f = parse(f) if isinstance(f, str) else f
    vs = vars_ if vars_ is not None else variables(f)
    rows = []
    for values in itertools.product([True, False], repeat=len(vs)):
        env = dict(zip(vs, values))
        rows.append((env, f.eval(env)))
    return rows


def format_truth_table(f: Formula | str) -> str:
    f = parse(f) if isinstance(f, str) else f
    vs = variables(f)
    head = vs + [str(f)]
    widths = [max(len(h), 1) for h in head]
    lines = [" | ".join(h.ljust(w) for h, w in zip(head, widths))]
    lines.append("-+-".join("-" * w for w in widths))
    for env, val in truth_table(f, vs):
        cells = ["T" if env[v] else "F" for v in vs] + ["T" if val else "F"]
        lines.append(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))
    return "\n".join(lines)


def is_tautology(f: Formula | str) -> bool:
    return all(v for _, v in truth_table(f))


def is_contradiction(f: Formula | str) -> bool:
    return not any(v for _, v in truth_table(f))


def is_satisfiable(f: Formula | str) -> bool:
    return not is_contradiction(f)


def equivalent(f: Formula | str, g: Formula | str) -> bool:
    f = parse(f) if isinstance(f, str) else f
    g = parse(g) if isinstance(g, str) else g
    vs = sorted(set(variables(f)) | set(variables(g)))
    return all(f.eval(env) == g.eval(env) for env, _ in truth_table(Const(True), vs))


def canonical_dnf(f: Formula | str) -> str:
    """Sum of minterms, read directly off the truth table."""
    f = parse(f) if isinstance(f, str) else f
    vs = variables(f)
    terms = []
    for env, val in truth_table(f, vs):
        if val:
            lits = [v if env[v] else f"¬{v}" for v in vs]
            terms.append("(" + " ∧ ".join(lits) + ")" if len(lits) > 1 else (lits[0] if lits else "T"))
    return " ∨ ".join(terms) if terms else "F"


def canonical_cnf(f: Formula | str) -> str:
    """Product of maxterms, read directly off the truth table."""
    f = parse(f) if isinstance(f, str) else f
    vs = variables(f)
    clauses = []
    for env, val in truth_table(f, vs):
        if not val:
            lits = [f"¬{v}" if env[v] else v for v in vs]
            clauses.append("(" + " ∨ ".join(lits) + ")" if len(lits) > 1 else (lits[0] if lits else "F"))
    return " ∧ ".join(clauses) if clauses else "T"


# ----------------------------------------------------------------------------
# Rule-based advisor
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    condition: str
    advice: str

    @property
    def formula(self) -> Formula:
        return parse(self.condition)


def derive_facts(quantities: dict[str, float]) -> dict[str, bool]:
    """Map a profile (activity -> yearly quantity) to truth values of atomic propositions."""
    q = lambda k: quantities.get(k, 0.0)  # noqa: E731
    car_km = q("car_petrol") + q("car_diesel")
    diet_days = {k: q(k) for k in ("diet_heavy_meat", "diet_average", "diet_vegetarian", "diet_vegan")}
    main_diet = max(diet_days, key=diet_days.get) if any(diet_days.values()) else None
    waste_total = q("waste_landfill") + q("waste_recycled") + q("waste_compost")
    return {
        "drives_a_lot": car_km > 5000,
        "uses_ev": q("car_ev") > 0,
        "uses_public_transport": q("bus") + q("train") > 1000,
        "cycles_or_walks": q("bicycle") + q("walk") > 500,
        "flies_short_haul": q("flight_short") > 0,
        "flies_a_lot": q("flight_short") + q("flight_long") > 5000,
        "heavy_meat_diet": main_diet == "diet_heavy_meat",
        "plant_based": main_diet in ("diet_vegetarian", "diet_vegan"),
        "high_electricity": q("electricity") > 2400,
        "has_solar": q("solar") > 0,
        "recycles": waste_total > 0 and (q("waste_recycled") + q("waste_compost")) / waste_total >= 0.5,
        "buys_lots_of_electronics": q("electronics") >= 3,
        "buys_lots_of_clothes": q("clothing") >= 20,
    }


DEFAULT_RULES: list[Rule] = [
    Rule("drives_a_lot & ~uses_public_transport",
         "Replace some car trips with bus or train; train emits about 80% less per km than a petrol car."),
    Rule("drives_a_lot & ~uses_ev",
         "When replacing your car, consider an EV. It emits about 70% less per km, even on today's grid."),
    Rule("~cycles_or_walks & ~drives_a_lot",
         "Try cycling or walking for trips under 3 km; these emit nothing."),
    Rule("flies_short_haul",
         "Take the train instead of short-haul flights where possible; it emits about 6x less per km."),
    Rule("flies_a_lot",
         "Air travel is a large share of your footprint. Combine trips or use video calls."),
    Rule("heavy_meat_diet",
         "Cut red meat to a few days a week. A vegetarian diet emits about 47% less than a high-meat diet."),
    Rule("~plant_based & ~heavy_meat_diet",
         "Adding meat-free days to a mixed diet saves about 1.8 kg CO2e per day."),
    Rule("high_electricity & ~has_solar",
         "Your electricity use is high. Consider rooftop solar and efficient (5-star) appliances."),
    Rule("~recycles",
         "Separate wet waste for composting and dry waste for recycling. Landfilled waste emits about 30x more per kg."),
    Rule("buys_lots_of_electronics | buys_lots_of_clothes",
         "Buy fewer new goods: repair, buy second-hand, and keep devices longer."),
    Rule("plant_based & (uses_public_transport | cycles_or_walks) & recycles",
         "Great habits! You're already doing the high-impact things; keep it up."),
]


def advise(facts: dict[str, bool], rules: list[Rule] | None = None) -> list[Rule]:
    """Modus ponens: for each rule (P -> advice), if P evaluates to true, the advice fires."""
    return [r for r in (rules or DEFAULT_RULES) if r.formula.eval(facts)]


def evaluate(expr: str, env: dict[str, bool]) -> bool:
    return parse(expr).eval(env)


__all__ = [
    "parse", "tokenize", "truth_table", "format_truth_table", "is_tautology", "is_contradiction",
    "is_satisfiable", "equivalent", "canonical_dnf", "canonical_cnf", "variables", "Rule",
    "DEFAULT_RULES", "derive_facts", "advise", "evaluate",
]
