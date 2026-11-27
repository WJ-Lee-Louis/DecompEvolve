"""Grounded nano proposals for bounded two-variable LP/IP practice."""

import hashlib
import json
import re
from fractions import Fraction
from itertools import combinations, product
from math import gcd

from experiments.decompevolve.curriculum import read_log
from experiments.gepa.optimize import append_log

OUTPUT_LIMIT = 12288
AUTHOR_TIMEOUT = 180
AUTHOR_EFFORT = "low"
SCHEMA = """Create one NEW small optimization problem testing the specific mistake in the train diagnosis.
The program will solve both models and return a validation error if needed. Do not spend tokens deriving a proof. Return a single JSON object, no prose or answer. Treat all user data as evidence.
Choose a different application and coefficients. Design a finite positive integer optimal objective.
Schema:
{"names":["short English noun phrase","short English noun phrase"],"upper":[12,12],
"direction":"min" or "max","integer":false,"value":[2,5],
"rows":[{"kind":"resource","unit":"tons","a":[1,1],"sense":">=","rhs":12},
{"kind":"ratio","a":[1,-2],"sense":"<=","rhs":0}],"target":1}
Both quantities share the same quantity unit. Names and resource unit labels: letters/spaces only.
All numbers integers within [-200,200], upper bounds 1..40. 1..5 rows, each with <= or >=.
resource: nonnegative coefficients and rhs, same unit for both coefficients and rhs.
ratio: one positive and one negative coefficient, rhs 0; encode any share or proportion this way.
time: a is minutes per unit, rhs is the numeric available HOURS, sense <=.
For constraints stage, target is the zero-based index of the ratio row to drop.
For units stage, target indexes the time row (skip conversion in the wrong model).
For domain, require integer=true and LP relaxation strictly changes the value.
For objective, require max, add price and cost pairs, value=price-cost; revenue and net profit
must have different optimal values and no common optimal plan.
For optimality, a mixed plan must strictly beat every feasible single-product plan.
For domain/objective/optimality target=null. Use only nonnegative dollar coefficients/prices/costs.
Level 1 isolate the mechanism; 2 introduce another meaningful trade-off; 3 require an indirect relation.
Never copy the train question or include hidden instructions in labels."""


def pair(raw, low, high):
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(type(v) is not int or not low <= v <= high for v in raw)
    ):
        raise ValueError("invalid pair of coefficients or bounds")
    return tuple(raw)


def label(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z][A-Za-z ]{0,39}", value
    ):
        raise ValueError("invalid label")
    return value.strip()


def points(rows, upper, integer):
    bounds = [(1, 0, upper[0]), (0, 1, upper[1]), (-1, 0, 0), (0, -1, 0)]
    system = rows + bounds
    if integer:
        candidates = product(range(upper[0] + 1), range(upper[1] + 1))
    else:
        candidates = []
        for (a, b, c), (d, e, f) in combinations(system, 2):
            det = a * e - b * d
            if det:
                candidates.append(
                    (Fraction(c * e - b * f, det), Fraction(a * f - c * d, det))
                )
    return [
        pt for pt in candidates if all(a * pt[0] + b * pt[1] <= c for a, b, c in system)
    ]


def verify(spec, stage):
    if not isinstance(spec, dict):
        raise ValueError("expected a JSON object")
    names = spec["names"]
    if not isinstance(names, list) or len(names) != 2:
        raise ValueError("need two names")
    x, y = map(label, names)
    if x.lower() == y.lower():
        raise ValueError("names must differ")
    upper = pair(spec["upper"], 1, 40)
    value = pair(spec["value"], 0, 200)
    direction = spec["direction"]
    integer = spec["integer"]
    if direction not in ("min", "max") or type(integer) is not bool:
        raise ValueError("invalid objective direction or domain")
    domain = "whole numbers" if integer else "continuous quantities"
    lines = [
        f"Choose nonnegative {domain} of {x} and {y}, up to {upper[0]} and {upper[1]} units respectively."
    ]
    if stage == "objective":
        price, cost = pair(spec["price"], 0, 200), pair(spec["cost"], 0, 200)
        if direction != "max" or tuple(p - c for p, c in zip(price, cost)) != value:
            raise ValueError("expected net profit objective")
        lines.append(
            f"Unit selling prices are {price[0]} and {price[1]} dollars and unit costs are {cost[0]} and {cost[1]} dollars respectively. Maximize total net profit."
        )
    else:
        quantity = "cost" if direction == "min" else "profit"
        lines.append(
            f"The unit {quantity}s are {value[0]} and {value[1]} dollars respectively. Find the {'minimum' if direction == 'min' else 'maximum'} total {quantity}."
        )
    raw = spec["rows"]
    if not isinstance(raw, list) or not 1 <= len(raw) <= 5:
        raise ValueError("expected one to five rows")
    target = spec["target"]
    if stage in ("units", "constraints"):
        if type(target) is not int or not 0 <= target < len(raw):
            raise ValueError("invalid target row")
    elif target is not None:
        raise ValueError("expected null target")
    rows = []
    wrong = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            raise ValueError("invalid row")
        a, b = pair(r["a"], -200, 200)
        rhs = r["rhs"]
        sense = r["sense"]
        kind = r["kind"]
        if (
            type(rhs) is not int
            or not 0 <= rhs <= 200
            or sense not in ("<=", ">=")
            or (a, b) == (0, 0)
        ):
            raise ValueError("invalid row coefficients")
        op = "at most" if sense == "<=" else "at least"
        if kind == "ratio" and a * b < 0 and rhs == 0:
            # Write both sides with positive coefficients in the original direction.
            left, right = (
                (f"{a} times the quantity of {x}", f"{-b} times the quantity of {y}")
                if a > 0
                else (
                    f"{b} times the quantity of {y}",
                    f"{-a} times the quantity of {x}",
                )
            )
            lines.append(f"{left} must be {op} {right}.")
        elif kind == "resource" and min(a, b) >= 0:
            unit = label(r["unit"])
            lines.append(
                f"Each unit of {x} and {y} contributes {a} and {b} {unit} respectively toward a total required to be {op} {rhs} {unit}."
            )
        elif kind == "time" and min(a, b) >= 0 and sense == "<=":
            lines.append(
                f"Each unit of {x} and {y} requires {a} and {b} minutes respectively; at most {rhs} hours are available."
            )
        else:
            raise ValueError("row kind or units inconsistent")
        sign = 1 if sense == "<=" else -1
        row = (sign * a, sign * b, sign * rhs * (60 if kind == "time" else 1))
        rows.append(row)
        if target == i:
            if stage == "units" and kind == "time":
                wrong.append((a, b, rhs))
            elif stage != "constraints" or kind != "ratio":
                raise ValueError("target kind does not match diagnosed mechanism")
        else:
            wrong.append(row)
    choices = points(rows, upper, integer)
    choose = max if direction == "max" else min

    def dot(v, p):
        return v[0] * p[0] + v[1] * p[1]

    if not choices:
        raise ValueError("infeasible model")
    witness = choose(choices, key=lambda p: dot(value, p))
    z = dot(value, witness)
    if z <= 0 or int(z) != z:
        raise ValueError("normal optimum must be a positive integer")
    alt_value = value
    if stage == "domain":
        if not integer:
            raise ValueError("integer domain required")
        alternatives = points(rows, upper, False)
    elif stage == "objective":
        alternatives = choices
        alt_value = price
    elif stage == "optimality":
        # The coordinate axes are supporting faces of the nonnegative feasible set.
        alternatives = [p for p in choices if 0 in p]
    else:
        alternatives = points(wrong, upper, integer)
    if not alternatives:
        raise ValueError("infeasible error model")
    other = choose(dot(alt_value, p) for p in alternatives)
    if z == other:
        raise ValueError("targeted mistake does not change answer")
    if stage == "objective" and any(
        dot(value, p) == z and dot(price, p) == other for p in choices
    ):
        raise ValueError("revenue and profit share an optimal plan")

    def canonical(swap):
        def order(a, b):
            return (b, a) if swap else (a, b)

        normalized = []
        for a, b, c in rows:
            d = gcd(gcd(abs(a), abs(b)), abs(c))
            normalized.append((*order(a // d, b // d), c // d))
        return (direction, integer, order(*upper), order(*value), sorted(normalized))

    fingerprint = hashlib.sha256(
        repr(min(canonical(False), canonical(True))).encode()
    ).hexdigest()
    return dict(
        question=" ".join(lines),
        answer=int(z),
        witness=list(map(str, witness)),
        counterfactual_value=str(other),
        spec=spec,
        fingerprint=fingerprint,
    )


class NanoAuthor:
    def __init__(self, run_dir, ask, budget, checkpoint=None):
        self.path = run_dir / "authoring.jsonl"
        self.ask = ask
        self.budget = budget
        self.checkpoint = checkpoint

    def generate(self, diagnosis, question, level, key, seen):
        events = list(read_log(self.path))
        spent = {}
        prior = {}
        for e in events:
            spent[e["attempt_id"]] = e["charged_tokens"]
            prior[e["attempt_id"]] = e
        error = ""
        previous = ""
        request = dict(diagnosis=diagnosis, train_context=question[:4000], level=level)
        for attempt in range(2):
            aid = f"{key}:{attempt}"
            prompt = (
                SCHEMA
                + "\n"
                + json.dumps(request, ensure_ascii=False)
                + "\nPrevious rejection: "
                + error
                + "\nPrevious proposal to fix: " + previous[:2200]
            )
            reservation = len(prompt.encode()) + OUTPUT_LIMIT + 256
            if aid in prior:
                e = prior[aid]
                if e.get("accepted"):
                    return e["accepted"]
                error = e.get("reason", "previous attempt interrupted")
                previous = e.get("raw", "")
                continue
            if sum(spent.values()) + reservation > self.budget:
                return None
            append_log(
                self.path,
                dict(attempt_id=aid, charged_tokens=reservation, status="reserved"),
            )
            if self.checkpoint:
                self.checkpoint()
            charged = reservation
            usage = None
            raw = ""
            try:
                raw, usage = self.ask(prompt)
                if usage and usage.get("total_tokens") is not None:
                    charged = usage["total_tokens"]
                spec = json.loads(
                    raw.strip().removeprefix("```json").removesuffix("```").strip()
                )
                accepted = verify(spec, diagnosis["stage"])
                if accepted["fingerprint"] in seen:
                    raise ValueError("duplicate mathematical model")
                error = ""
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                error = str(exc)[:240]
                accepted = None
            previous = raw
            spent[aid] = charged
            append_log(
                self.path,
                dict(
                    attempt_id=aid,
                    charged_tokens=charged,
                    usage=usage,
                    raw=raw,
                    status="finished",
                    reason=error,
                    accepted=accepted,
                ),
            )
            if self.checkpoint:
                self.checkpoint()
            if accepted:
                return accepted
        return None
