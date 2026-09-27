"""Constrained binary programs and their exact penalty (QUBO/HUBO) encodings.

A :class:`ConstrainedBinaryProgram` is

    min_x  f(x)                      (any pseudo-Boolean polynomial)
    s.t.   a_k . x  (<=, >=, ==)  b_k   for linear constraints k with INTEGER a_k, b_k
           sum_{i in G} x_i <= 1        for "at most one" groups (one-hot style)

Penalty encoding (``to_penalty_model``)
---------------------------------------
* equality          : Lambda * (a.x - b)^2
* a.x <= b          : Lambda * (a.x + sum_j w_j s_j - b)^2, slack s in {0,1}^m encoding
                      every integer in [0, b - min_x a.x]  (bounded binary encoding)
* a.x >= b          : Lambda * (a.x - sum_j w_j s_j - b)^2, slack range [0, max a.x - b]
* at-most-one group : Lambda * sum_{i<j in G} x_i x_j   (no slack needed)

Exactness (corrected Theorem IV.1)
----------------------------------
The legacy maths document states that Lambda > f_max - f_min makes every
minimiser feasible because "the penalty is >= Lambda at infeasible points".
That is only true when every violated constraint contributes at least 1, i.e.
when constraint data are *integers* (g(x)^2 >= 1 whenever g(x) != 0) - with
areas in m^2 or costs in dollars the claim fails - and the legacy
``max(0, h(x))^2`` inequality penalty is not a polynomial at all.  Here we require
integer data and use slack bits, so for every (x, s):

    x infeasible            =>  P(x, s) >= Lambda
    x feasible              =>  min_s P(x, s) = 0

Hence, with f* the constrained optimum and L any lower bound of f,
    Lambda > f* - L    (in particular Lambda > f(x_feas) - L for any feasible x_feas)
guarantees that every global minimiser of F = f + P is feasible and optimal.
This bound is typically orders of magnitude smaller than the range-based one;
see :func:`safe_penalty_weight`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .polynomial import BinaryPolynomial


@dataclass
class LinearConstraint:
    coeffs: Dict[int, int]          # variable index -> integer coefficient
    sense: str                      # "<=", ">=", "=="
    rhs: int
    name: str = "c"

    def __post_init__(self) -> None:
        if self.sense not in ("<=", ">=", "=="):
            raise ValueError(f"bad sense {self.sense!r}")
        for i, a in self.coeffs.items():
            if int(a) != a:
                raise ValueError(f"constraint {self.name}: coefficient {a} of x{i} is not an "
                                 "integer; rescale to integer units (required for exact penalties)")
        if int(self.rhs) != self.rhs:
            raise ValueError(f"constraint {self.name}: rhs {self.rhs} is not an integer")
        self.coeffs = {int(i): int(a) for i, a in self.coeffs.items() if a != 0}
        self.rhs = int(self.rhs)

    def lhs(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        idx = np.fromiter(self.coeffs.keys(), dtype=int)
        a = np.fromiter(self.coeffs.values(), dtype=float)
        return x[:, idx] @ a if len(idx) else np.zeros(x.shape[0])

    def violation(self, x: np.ndarray) -> np.ndarray:
        """Non-negative violation amount per row of x."""
        v = self.lhs(x) - self.rhs
        if self.sense == "<=":
            return np.maximum(v, 0.0)
        if self.sense == ">=":
            return np.maximum(-v, 0.0)
        return np.abs(v)

    def lhs_range(self) -> Tuple[int, int]:
        lo = sum(a for a in self.coeffs.values() if a < 0)
        hi = sum(a for a in self.coeffs.values() if a > 0)
        return lo, hi


def slack_weights(max_value: int) -> List[int]:
    """Bounded binary encoding: weights whose subset sums are exactly {0..max_value}.

    Uses 1, 2, 4, ..., 2^(k-1) and a final weight max_value - (2^k - 1), so no
    representable slack exceeds max_value (plain binary would overshoot and
    admit infeasible points at zero penalty).
    """
    if max_value < 0:
        raise ValueError("max_value must be >= 0")
    if max_value == 0:
        return []
    w, total = [], 0
    p = 1
    while total + p <= max_value:
        w.append(p)
        total += p
        p *= 2
    if total < max_value:
        w.append(max_value - total)
    return w


@dataclass
class PenaltyEncoding:
    """Result of encoding: model over [decision vars | slack vars]."""

    model: BinaryPolynomial
    n_decision: int
    slack_slices: Dict[str, slice]
    penalty_weight: float

    def decision_part(self, X: np.ndarray) -> np.ndarray:
        return np.atleast_2d(X)[:, : self.n_decision]


@dataclass
class ConstrainedBinaryProgram:
    objective: BinaryPolynomial
    constraints: List[LinearConstraint] = field(default_factory=list)
    at_most_one: List[List[int]] = field(default_factory=list)
    names: Optional[list] = None

    @property
    def n(self) -> int:
        return self.objective.n

    # ----------------------------------------------------------- feasibility
    def violations(self, X: np.ndarray) -> np.ndarray:
        """Total violation per row (0 means feasible)."""
        X = np.atleast_2d(X)[:, : self.n]
        tot = np.zeros(X.shape[0])
        for c in self.constraints:
            tot += c.violation(X)
        for g in self.at_most_one:
            tot += np.maximum(X[:, g].sum(axis=1) - 1, 0)
        return tot

    def is_feasible(self, X: np.ndarray) -> np.ndarray:
        return self.violations(X) == 0

    def objective_values(self, X: np.ndarray) -> np.ndarray:
        return self.objective.energies(np.atleast_2d(X)[:, : self.n])

    # --------------------------------------------------------------- penalty
    def penalty_polynomial(self, n_total: int) -> Tuple[BinaryPolynomial, Dict[str, slice], int]:
        """Unweighted penalty P(x, s) >= 0 (integer-valued, 0 iff consistent+feasible)."""
        P = BinaryPolynomial({}, 0.0, n_total)
        slices: Dict[str, slice] = {}
        nxt = self.n
        for c in self.constraints:
            lo, hi = c.lhs_range()
            lin: Dict[int, float] = {i: float(a) for i, a in c.coeffs.items()}
            if (c.sense == "<=" and c.rhs >= hi) or (c.sense == ">=" and c.rhs <= lo):
                slices[c.name] = slice(nxt, nxt)     # redundant: always satisfied
                continue
            if c.sense == "<=":
                # need slack = b - a.x for every feasible x, i.e. range [0, b - lo]
                smax = c.rhs - lo
                if smax < 0:
                    raise ValueError(f"constraint {c.name} infeasible for all x")
                for w in slack_weights(smax):
                    lin[nxt] = float(w)
                    nxt += 1
            elif c.sense == ">=":
                smax = hi - c.rhs
                if smax < 0:
                    raise ValueError(f"constraint {c.name} infeasible for all x")
                for w in slack_weights(smax):
                    lin[nxt] = -float(w)
                    nxt += 1
            elif not (lo <= c.rhs <= hi):
                raise ValueError(f"constraint {c.name} infeasible for all x")
            start = nxt - (len(lin) - len(c.coeffs))
            slices[c.name] = slice(start, nxt)
            # (sum_i a_i z_i - b)^2 = sum_i a_i^2 z_i + 2 sum_{i<j} a_i a_j z_i z_j
            #                         - 2 b sum_i a_i z_i + b^2
            items = sorted(lin.items())
            for i, a in items:
                P.add_term((i,), a * a - 2.0 * c.rhs * a)
            for p in range(len(items)):
                for q in range(p + 1, len(items)):
                    P.add_term((items[p][0], items[q][0]), 2.0 * items[p][1] * items[q][1])
            P.offset += float(c.rhs) ** 2
        for g in self.at_most_one:
            for p in range(len(g)):
                for q in range(p + 1, len(g)):
                    P.add_term((g[p], g[q]), 1.0)
        return P, slices, nxt

    def num_slack(self) -> int:
        return self.penalty_polynomial(10 ** 9)[2] - self.n

    def to_penalty_model(self, penalty_weight: Optional[float] = None,
                         feasible_hint: Optional[np.ndarray] = None,
                         safety: float = 1.0, form: str = "quadratic_slack",
                         lambdas: Optional[Tuple[float, float]] = None) -> PenaltyEncoding:
        """Unconstrained model F(x, s) = f(x) + Lambda * P(x, s).

        ``penalty_weight=None`` uses the provably exact :func:`safe_penalty_weight`
        (times ``safety``).  Pass a number to override (e.g. for tuning studies).

        ``form`` selects how *inequality* constraints are penalised (equalities and
        at-most-one groups always use their exact, slack-free quadratic penalties):

        * ``"quadratic_slack"`` (default): (a.x + slack - b)^2, exact (see module docstring).
        * ``"unbalanced"``: -l1*h + l2*h^2 with h = b - a.x for ``<=`` (h = a.x - b for ``>=``),
          l1, l2 = ``lambdas`` (default: both = ``penalty_weight``). No slack bits.
          **Not exact**: violations cost l1 + l2 per unit (for a unit violation), but some
          feasible points are also penalised (large h) and ground states can be infeasible
          (Montanez-Barrera et al. 2023, unbalanced penalisation).
        * ``"linear"``: ``penalty_weight`` * (a.x - b) for ``<=`` (and b - a.x for ``>=``),
          a Lagrangian-style term. No slack bits. **Not exact**: feasibility must be checked
          after sampling.
        """
        if form != "quadratic_slack":
            return self._inexact_penalty_model(penalty_weight, feasible_hint, safety, form, lambdas)
        n_total = self.n + self.num_slack()
        P, slices, _ = self.penalty_polynomial(n_total)
        if penalty_weight is None:
            penalty_weight = safety * safe_penalty_weight(self, feasible_hint)
        f = BinaryPolynomial(dict(self.objective.terms), self.objective.offset, n_total)
        model = f + P * float(penalty_weight)
        model.n = n_total
        names = list(self.names) if self.names is not None else [f"x{i}" for i in range(self.n)]
        for name, sl in slices.items():
            names += [f"slack[{name}]{k}" for k in range(sl.stop - sl.start)]
        model.names = names
        return PenaltyEncoding(model, self.n, slices, float(penalty_weight))


def _inexact_penalty_model(self, penalty_weight, feasible_hint, safety, form, lambdas):
    if form not in ("unbalanced", "linear"):
        raise ValueError(f"unknown penalty form {form!r}")
    if penalty_weight is None:
        penalty_weight = safety * safe_penalty_weight(self, feasible_hint)
    lam = float(penalty_weight)
    l1, l2 = lambdas if lambdas is not None else (lam, lam)
    n = self.n
    model = BinaryPolynomial(dict(self.objective.terms), self.objective.offset, n)
    for c in self.constraints:
        items = sorted(c.coeffs.items())
        if c.sense == "==":                                   # exact, no slack
            for i, a in items:
                model.add_term((i,), lam * (a * a - 2.0 * c.rhs * a))
            for p_ in range(len(items)):
                for q in range(p_ + 1, len(items)):
                    model.add_term((items[p_][0], items[q][0]), 2.0 * lam * items[p_][1] * items[q][1])
            model.offset += lam * float(c.rhs) ** 2
            continue
        sgn = 1.0 if c.sense == "<=" else -1.0               # h = sgn*(b - a.x) >= 0 when feasible
        # h = sgn*b - sum_i sgn*a_i x_i
        hb = sgn * c.rhs
        ha = [(i, -sgn * a) for i, a in items]
        if form == "linear":
            model.offset -= lam * hb
            for i, a in ha:
                model.add_term((i,), -lam * a)
        else:                                                 # -l1*h + l2*h^2
            model.offset += -l1 * hb + l2 * hb * hb
            for i, a in ha:
                model.add_term((i,), -l1 * a + l2 * (a * a + 2.0 * hb * a))
            for p_ in range(len(ha)):
                for q in range(p_ + 1, len(ha)):
                    model.add_term((ha[p_][0], ha[q][0]), 2.0 * l2 * ha[p_][1] * ha[q][1])
    for g in self.at_most_one:
        for p_ in range(len(g)):
            for q in range(p_ + 1, len(g)):
                model.add_term((g[p_], g[q]), lam)
    model.n = n
    model.names = list(self.names) if self.names is not None else [f"x{i}" for i in range(n)]
    return PenaltyEncoding(model, n, {}, lam)


ConstrainedBinaryProgram._inexact_penalty_model = _inexact_penalty_model


def greedy_feasible(cbp: ConstrainedBinaryProgram, rng: Optional[np.random.Generator] = None,
                    tries: int = 200) -> Optional[np.ndarray]:
    """Cheap feasible point search: start at 0 or 1 and repair by single flips."""
    rng = rng or np.random.default_rng(0)
    n = cbp.n
    for start in [np.zeros(n, dtype=np.int8)] + [rng.integers(0, 2, n).astype(np.int8)
                                                 for _ in range(tries)]:
        x = start.copy()
        for _ in range(4 * n):
            v = cbp.violations(x)[0]
            if v == 0:
                return x
            best, best_v = None, v
            for i in rng.permutation(n):
                x[i] ^= 1
                vi = cbp.violations(x)[0]
                x[i] ^= 1
                if vi < best_v:
                    best, best_v = i, vi
            if best is None:
                break
            x[best] ^= 1
    return None


def safe_penalty_weight(cbp: ConstrainedBinaryProgram,
                        feasible_hint: Optional[np.ndarray] = None,
                        margin: float = 1e-3) -> float:
    """Smallest weight certified exact by the corrected Theorem IV.1.

    Lambda = f(x_feas) - L(f) + margin, where x_feas is any feasible point and L is
    the sign-based lower bound of f.  Falls back to the (looser) coefficient range
    U(f) - L(f) if no feasible point is known.
    """
    L, U = cbp.objective.coefficient_bounds()
    x = feasible_hint
    if x is None:
        x = greedy_feasible(cbp)
    if x is not None and cbp.is_feasible(x)[0]:
        x = feasible_descent(cbp, np.array(x, dtype=np.int8)[: cbp.n])
        ub = float(cbp.objective_values(x)[0])
    else:
        ub = U
    return max(ub - L, 0.0) + margin


def feasible_descent(cbp: ConstrainedBinaryProgram, x: np.ndarray, max_steps: int = 10_000) -> np.ndarray:
    """Best-improvement 1-flip descent on f restricted to feasible points.

    Tightens the upper bound f(x_feas) >= f* used by :func:`safe_penalty_weight`.
    """
    x = x.copy()
    n = cbp.n
    f = cbp.objective_values(x)[0]
    for _ in range(max_steps):
        Y = np.repeat(x[None, :], n, axis=0)
        Y[np.arange(n), np.arange(n)] ^= 1
        ok = cbp.is_feasible(Y)
        if not ok.any():
            break
        fy = np.where(ok, cbp.objective_values(Y), np.inf)
        i = int(np.argmin(fy))
        if fy[i] >= f - 1e-12:
            break
        x, f = Y[i], fy[i]
    return x
