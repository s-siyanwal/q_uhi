"""HUBO -> QUBO quadratisation by Rosenberg substitution (Rosenberg 1975).

Repeatedly pick the variable pair (u, v) that occurs in the most terms of degree
>= 3, introduce an auxiliary y meant to equal u*v, replace u*v by y in those
terms, and add the penalty

    M * (u v - 2 u y - 2 v y + 3 y)            (>= 0, and = 0  iff  y = u v).

Exactness (proved in docs/BACKGROUND.md, sec. 3):  let G be the reduced
objective (without penalties) and S = sum |c_T| over terms of G that contain an
auxiliary.  If some auxiliary is "wrong", the earliest-created wrong one has
correct inputs, so its penalty is >= 1 and F >= G(x, y*) - S + M.  Hence any
M > S makes  min_y F(x, y) = f(x)  for every x, so ground states coincide.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from .polynomial import BinaryPolynomial, Term


@dataclass
class Quadratization:
    qubo: BinaryPolynomial
    n_original: int
    aux: List[Tuple[int, int, int]]     # (y, u, v): y represents u*v
    penalty_weight: float

    def original_part(self, X: np.ndarray) -> np.ndarray:
        return np.atleast_2d(X)[:, : self.n_original]

    def lift(self, x: np.ndarray) -> np.ndarray:
        """Extend an original assignment with the consistent auxiliary values."""
        z = np.zeros(self.qubo.n, dtype=np.int8)
        z[: self.n_original] = x
        for y, u, v in self.aux:
            z[y] = z[u] & z[v]
        return z


def quadratize(poly: BinaryPolynomial, penalty_weight: Optional[float] = None,
               safety: float = 1.0) -> Quadratization:
    """Return an exact QUBO for ``poly`` (identity if already quadratic)."""
    terms: Dict[Term, float] = dict(poly.terms)
    n = poly.n
    aux: List[Tuple[int, int, int]] = []
    while True:
        high = [t for t in terms if len(t) > 2]
        if not high:
            break
        cnt: Counter = Counter()
        for t in high:
            cnt.update(combinations(t, 2))
        (u, v), _ = max(cnt.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        y = n
        n += 1
        aux.append((y, u, v))
        new_terms: Dict[Term, float] = {}
        for t, c in terms.items():
            if len(t) > 2 and u in t and v in t:
                t = tuple(sorted([i for i in t if i not in (u, v)] + [y]))
            new_terms[t] = new_terms.get(t, 0.0) + c
        terms = new_terms
    aux_set = {a[0] for a in aux}
    S = sum(abs(c) for t, c in terms.items() if aux_set.intersection(t))
    M = penalty_weight if penalty_weight is not None else safety * (S + 1e-3 * max(1.0, S))
    out = BinaryPolynomial(terms, poly.offset, n)
    for y, u, v in aux:
        out.add_term((u, v), M)
        out.add_term((u, y), -2.0 * M)
        out.add_term((v, y), -2.0 * M)
        out.add_term((y,), 3.0 * M)
    names = list(poly.names) if poly.names is not None else [f"x{i}" for i in range(poly.n)]
    names += [f"aux[{u}*{v}]" for _, u, v in aux]
    out.names = names
    return Quadratization(out, poly.n, aux, float(M))
