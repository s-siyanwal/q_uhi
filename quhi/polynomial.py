"""Pseudo-Boolean polynomials: the common currency of QUBO and HUBO models.

A pseudo-Boolean polynomial over binary variables x in {0,1}^n is

    f(x) = c_0 + sum_T c_T prod_{i in T} x_i ,

where every term T is a set of distinct variable indices.  Because x_i^2 = x_i,
no index needs to repeat inside a term, so this representation is canonical.

* degree(f) <= 2  -> QUBO   (quadratic unconstrained binary optimisation)
* degree(f) >  2  -> HUBO   (higher-order unconstrained binary optimisation)

Conventions used everywhere in ``quhi``
---------------------------------------
QUBO matrices are **upper-triangular** with ``f(x) = x^T Q x + offset``:
``Q[i, i]`` holds the linear coefficient of ``x_i`` and ``Q[i, j]`` (i < j) holds
the full coefficient of ``x_i x_j``.  (The legacy code mixed symmetric and
upper-triangular conventions, which silently doubled couplings - see
docs/AUDIT.md.)  :meth:`BinaryPolynomial.from_qubo_matrix` accepts either form
and interprets a general matrix exactly as ``x^T Q x``.

Ising form uses spins s_i = 2 x_i - 1 in {-1,+1} and the same polynomial
container: ``to_spin()`` returns g(s) with f(x) = g(2x - 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np

Term = Tuple[int, ...]


def _canon(term: Iterable[int]) -> Term:
    """Sorted tuple of distinct indices (x_i^2 = x_i collapses repeats)."""
    return tuple(sorted(set(int(i) for i in term)))


@dataclass
class BinaryPolynomial:
    """Sparse pseudo-Boolean polynomial ``offset + sum_T c_T prod_{i in T} x_i``.

    Parameters
    ----------
    terms:
        Mapping from index tuples to coefficients.  Keys are canonicalised on
        construction (sorted, de-duplicated) and equal keys are summed.
    offset:
        Constant term.
    n:
        Number of variables.  Defaults to ``1 + max index``.
    names:
        Optional human-readable variable labels (e.g. ``"g[3,4]"``, ``"slack_budget_2"``).
    """

    terms: Dict[Term, float] = field(default_factory=dict)
    offset: float = 0.0
    n: Optional[int] = None
    names: Optional[list] = None

    def __post_init__(self) -> None:
        clean: Dict[Term, float] = {}
        extra = 0.0
        for key, c in self.terms.items():
            t = _canon(key)
            if len(t) == 0:
                extra += float(c)
                continue
            clean[t] = clean.get(t, 0.0) + float(c)
        self.terms = {t: c for t, c in clean.items() if c != 0.0}
        self.offset = float(self.offset) + extra
        max_idx = max((t[-1] for t in self.terms), default=-1)
        if self.n is None:
            self.n = max_idx + 1
        elif max_idx >= self.n:
            raise ValueError(f"term index {max_idx} out of range for n={self.n}")
        if self.names is not None and len(self.names) != self.n:
            raise ValueError("names must have length n")

    # ------------------------------------------------------------------ basics
    @property
    def degree(self) -> int:
        return max((len(t) for t in self.terms), default=0)

    @property
    def is_quadratic(self) -> bool:
        return self.degree <= 2

    @property
    def num_terms(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterator[Tuple[Term, float]]:
        return iter(self.terms.items())

    def copy(self) -> "BinaryPolynomial":
        return BinaryPolynomial(dict(self.terms), self.offset, self.n,
                                None if self.names is None else list(self.names))

    def __repr__(self) -> str:
        kind = "QUBO" if self.is_quadratic else f"HUBO(deg={self.degree})"
        return f"BinaryPolynomial<{kind}, n={self.n}, terms={self.num_terms}, offset={self.offset:.4g}>"

    # -------------------------------------------------------------- arithmetic
    def add_term(self, term: Iterable[int], coeff: float) -> None:
        t = _canon(term)
        if not t:
            self.offset += float(coeff)
            return
        if t[-1] >= self.n:
            self.n = t[-1] + 1
            if self.names is not None:
                self.names.extend(f"x{i}" for i in range(len(self.names), self.n))
        v = self.terms.get(t, 0.0) + float(coeff)
        if v == 0.0:
            self.terms.pop(t, None)
        else:
            self.terms[t] = v

    def __add__(self, other: "BinaryPolynomial | float") -> "BinaryPolynomial":
        if isinstance(other, (int, float)):
            out = self.copy()
            out.offset += float(other)
            return out
        n = max(self.n, other.n)
        out = BinaryPolynomial(dict(self.terms), self.offset + other.offset, n)
        for t, c in other.terms.items():
            out.add_term(t, c)
        out.names = self.names if (self.names is not None and self.n == n) else (
            other.names if (other.names is not None and other.n == n) else None)
        return out

    __radd__ = __add__

    def __mul__(self, other: "BinaryPolynomial | float") -> "BinaryPolynomial":
        if isinstance(other, (int, float)):
            return BinaryPolynomial({t: c * other for t, c in self.terms.items()},
                                    self.offset * other, self.n,
                                    None if self.names is None else list(self.names))
        n = max(self.n, other.n)
        out = BinaryPolynomial({}, 0.0, n)
        a = list(self.terms.items()) + [((), self.offset)]
        b = list(other.terms.items()) + [((), other.offset)]
        for ta, ca in a:
            for tb, cb in b:
                out.add_term(ta + tb, ca * cb)
        return out

    __rmul__ = __mul__

    def __neg__(self) -> "BinaryPolynomial":
        return self * -1.0

    def __sub__(self, other: "BinaryPolynomial | float") -> "BinaryPolynomial":
        return self + (-1.0 * other)

    def prune(self, tol: float) -> "BinaryPolynomial":
        """Drop terms with ``|c_T| <= tol`` (returns a new polynomial)."""
        return BinaryPolynomial({t: c for t, c in self.terms.items() if abs(c) > tol},
                                self.offset, self.n,
                                None if self.names is None else list(self.names))

    def truncate(self, max_degree: int) -> "BinaryPolynomial":
        """Drop all terms of degree > ``max_degree`` (returns a new polynomial)."""
        return BinaryPolynomial({t: c for t, c in self.terms.items() if len(t) <= max_degree},
                                self.offset, self.n,
                                None if self.names is None else list(self.names))

    def fix(self, assignment: Mapping[int, int]) -> "BinaryPolynomial":
        """Substitute fixed values for some variables (indices are kept; fixed vars vanish)."""
        out = BinaryPolynomial({}, self.offset, self.n,
                               None if self.names is None else list(self.names))
        for t, c in self.terms.items():
            keep = []
            zero = False
            for i in t:
                if i in assignment:
                    if assignment[i] == 0:
                        zero = True
                        break
                else:
                    keep.append(i)
            if not zero:
                out.add_term(keep, c)
        return out

    # ------------------------------------------------------------- evaluation
    def energy(self, x: Sequence[int]) -> float:
        """f(x) for a single assignment."""
        x = np.asarray(x)
        if x.shape[-1] != self.n:
            raise ValueError(f"expected {self.n} variables, got {x.shape[-1]}")
        e = self.offset
        for t, c in self.terms.items():
            if all(x[i] for i in t):
                e += c
        return float(e)

    def energies(self, X: np.ndarray) -> np.ndarray:
        """Vectorised f(x) for a batch ``X`` of shape (m, n)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.int8))
        out = np.full(X.shape[0], self.offset, dtype=float)
        for t, c in self.terms.items():
            out += c * np.all(X[:, list(t)] == 1, axis=1)
        return out

    def coefficient_bounds(self) -> Tuple[float, float]:
        """Cheap bounds  L <= min f  and  max f <= U  from term signs."""
        neg = sum(c for c in self.terms.values() if c < 0)
        pos = sum(c for c in self.terms.values() if c > 0)
        return self.offset + neg, self.offset + pos

    def interaction_graph(self) -> Dict[int, set]:
        """Adjacency sets of the (hyper)graph's primal graph."""
        adj: Dict[int, set] = {i: set() for i in range(self.n)}
        for t in self.terms:
            for i in t:
                adj[i].update(j for j in t if j != i)
        return adj

    # ------------------------------------------------------------ conversions
    @classmethod
    def from_qubo_matrix(cls, Q: np.ndarray, offset: float = 0.0) -> "BinaryPolynomial":
        """Interpret ``Q`` exactly as ``x^T Q x + offset`` (Q need not be symmetric)."""
        Q = np.asarray(Q, dtype=float)
        n = Q.shape[0]
        if Q.shape != (n, n):
            raise ValueError("Q must be square")
        terms: Dict[Term, float] = {}
        for i in range(n):
            if Q[i, i] != 0:
                terms[(i,)] = Q[i, i]
            for j in range(i + 1, n):
                c = Q[i, j] + Q[j, i]
                if c != 0:
                    terms[(i, j)] = c
        return cls(terms, offset, n)

    def to_qubo_matrix(self) -> Tuple[np.ndarray, float]:
        """Upper-triangular ``Q`` and ``offset`` with ``f(x) = x^T Q x + offset``."""
        if not self.is_quadratic:
            raise ValueError(f"degree {self.degree} > 2: quadratize first (quhi.quadratize)")
        Q = np.zeros((self.n, self.n))
        for t, c in self.terms.items():
            if len(t) == 1:
                Q[t[0], t[0]] += c
            else:
                Q[t[0], t[1]] += c
        return Q, self.offset

    def to_spin(self) -> "BinaryPolynomial":
        """Spin polynomial g with f(x) = g(s), s = 2x - 1 (keys then index spins).

        Uses x_i = (1 + s_i)/2, so a degree-k term c prod x_i expands into
        c / 2^k * sum_{S subset T} prod_{i in S} s_i.  Note s_i^2 = 1, which the
        subset expansion respects because each index appears at most once.
        """
        out: Dict[Term, float] = {}
        off = self.offset
        for t, c in self.terms.items():
            w = c / (2 ** len(t))
            for r in range(len(t) + 1):
                for sub in combinations(t, r):
                    if sub:
                        out[sub] = out.get(sub, 0.0) + w
                    else:
                        off += w
        return BinaryPolynomial(out, off, self.n)

    def to_ising(self) -> Tuple[np.ndarray, Dict[Tuple[int, int], float], float]:
        """(h, J, offset) with f(x) = offset + sum h_i s_i + sum_{i<j} J_ij s_i s_j.

        Sign convention: *plus* signs (dimod / D-Wave convention).  Only valid for
        QUBOs; HUBOs have higher-order spin terms - use :meth:`to_spin`.
        """
        if not self.is_quadratic:
            raise ValueError("to_ising requires degree <= 2; use to_spin() for HUBO")
        g = self.to_spin()
        h = np.zeros(self.n)
        J: Dict[Tuple[int, int], float] = {}
        for t, c in g.terms.items():
            if len(t) == 1:
                h[t[0]] = c
            else:
                J[t] = c
        return h, J, g.offset

    @classmethod
    def from_spin(cls, g: "BinaryPolynomial") -> "BinaryPolynomial":
        """Inverse of :meth:`to_spin`: s_i = 2 x_i - 1."""
        out = BinaryPolynomial({}, g.offset, g.n)
        for t, c in g.terms.items():
            # prod_{i in T} (2 x_i - 1) = sum_{S subset T} 2^{|S|} (-1)^{|T|-|S|} prod_S x_i
            for r in range(len(t) + 1):
                sign = -1.0 if (len(t) - r) % 2 else 1.0
                for sub in combinations(t, r):
                    out.add_term(sub, c * sign * (2 ** r))
        return out

    def to_dimod(self):  # pragma: no cover - optional dependency
        """Convert to a ``dimod.BinaryQuadraticModel`` (QUBO) or ``BinaryPolynomial`` (HUBO)."""
        import dimod
        if self.is_quadratic:
            lin = {t[0]: c for t, c in self.terms.items() if len(t) == 1}
            quad = {t: c for t, c in self.terms.items() if len(t) == 2}
            return dimod.BinaryQuadraticModel(lin, quad, self.offset, dimod.BINARY)
        poly = {t: c for t, c in self.terms.items()}
        poly[()] = self.offset
        return dimod.BinaryPolynomial(poly, dimod.BINARY)

    # --------------------------------------------------------------- compiled
    def compile(self) -> "CompiledPolynomial":
        return CompiledPolynomial.from_polynomial(self)


@dataclass
class CompiledPolynomial:
    """CSR-style arrays consumed by the numba solver kernels.

    ``term_vars[term_ptr[t]:term_ptr[t+1]]`` are the variables of term t and
    ``var_terms[var_ptr[i]:var_ptr[i+1]]`` are the terms containing variable i.
    """

    n: int
    offset: float
    coeff: np.ndarray       # (T,)  float64
    term_ptr: np.ndarray    # (T+1,) int64
    term_vars: np.ndarray   # (sum |T|,) int64
    var_ptr: np.ndarray     # (n+1,) int64
    var_terms: np.ndarray   # (sum |T|,) int64
    degree: int

    @classmethod
    def from_polynomial(cls, poly: BinaryPolynomial) -> "CompiledPolynomial":
        items = sorted(poly.terms.items())
        coeff = np.array([c for _, c in items], dtype=np.float64)
        lens = np.array([len(t) for t, _ in items], dtype=np.int64)
        term_ptr = np.zeros(len(items) + 1, dtype=np.int64)
        term_ptr[1:] = np.cumsum(lens)
        term_vars = np.array([i for t, _ in items for i in t], dtype=np.int64)
        per_var = [[] for _ in range(poly.n)]
        for k, (t, _) in enumerate(items):
            for i in t:
                per_var[i].append(k)
        var_ptr = np.zeros(poly.n + 1, dtype=np.int64)
        var_ptr[1:] = np.cumsum([len(v) for v in per_var])
        var_terms = np.array([k for v in per_var for k in v], dtype=np.int64)
        return cls(poly.n, poly.offset, coeff, term_ptr, term_vars, var_ptr, var_terms,
                   poly.degree)

    def max_abs_delta(self) -> np.ndarray:
        """Upper bound on |f(x with x_i flipped) - f(x)| per variable."""
        out = np.zeros(self.n)
        for i in range(self.n):
            ts = self.var_terms[self.var_ptr[i]:self.var_ptr[i + 1]]
            out[i] = np.abs(self.coeff[ts]).sum()
        return out
