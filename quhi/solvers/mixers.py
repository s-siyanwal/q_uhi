"""Constraint-preserving QAOA (Quantum Alternating Operator Ansatz, Hadfield et al. 2019).

Standard QAOA uses the transverse-field (X) mixer, which explores all 2^n bitstrings, so
constraints must enter the cost as penalties (slack bits, large Λ; see E6).  Here the
constraints are respected by the *dynamics*:

* the initial state is the uniform superposition over the feasible set F (for "exactly k
  of n" this is the Dicke state |D_n^k>);
* the mixer H_M = Π_F (Σ_moves) Π_F only connects feasible bitstrings, so the state never
  leaves span(F);
* the phase separator is the native objective f(x) (any degree, no slack, no Λ).

Mixer moves
-----------
``swap``        exchange one 1 with one 0 (XY term (X_iX_j + Y_iY_j)/2 on qubits i, j).
                ``swap_graph="complete"`` (all pairs) or ``"ring"`` (i, i±1 mod n).
``add_remove``  single bit flip (X_i), kept only when the result is feasible.

For an exact-cardinality constraint with complete-graph swaps, Π_F(Σ XY)Π_F *is* the
complete XY mixer: XY preserves Hamming weight, so the projection is a no-op
(verified against the explicit 2^n Pauli construction in tests/test_mixers.py).  For
knapsack budgets with unequal costs, one-option-per-cell groups and equity constraints,
the projection removes moves that would leave F.  This is option (i) of docs/NEXT.md:
exact and tested, but written as a projected operator, not compiled to a gate circuit.

Simulation
----------
Exact state-vector evolution restricted to span(F).  This is equivalent to the full
2^n simulation, because the initial state and both operators keep the state in span(F).
The cost is O(|F|^2) memory for the dense eigendecomposition of the mixer, so the limit
is on |F| (``max_feasible``), not on n.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..constraints import ConstrainedBinaryProgram
from .base import SampleSet


# ------------------------------------------------------------ feasible set
def enumerate_feasible(cbp: ConstrainedBinaryProgram, max_states: int = 2_000_000) -> np.ndarray:
    """All feasible decision vectors, as an (|F|, n) int8 array in lexicographic block order.

    Depth-first search over blocks (one block per at-most-one group; singleton blocks
    for ungrouped variables). Branches that exceed a ``<=`` constraint with non-negative
    coefficients are pruned; the full feasibility check is applied at the leaves.
    """
    n = cbp.n
    grouped = set(i for g in cbp.at_most_one for i in g)
    blocks: List[List[int]] = [list(g) for g in cbp.at_most_one] + \
        [[i] for i in range(n) if i not in grouped]
    prunable = [c for c in cbp.constraints if c.sense == "<=" and all(a >= 0 for a in c.coeffs.values())]
    A = np.zeros((len(prunable), n))
    for r, c in enumerate(prunable):
        for i, a in c.coeffs.items():
            A[r, i] = a
    rhs = np.array([c.rhs for c in prunable], dtype=float)
    out: List[np.ndarray] = []
    x = np.zeros(n, dtype=np.int8)
    lhs = np.zeros(len(prunable))

    def rec(b: int) -> None:
        if len(out) > max_states:
            raise ValueError(f"more than {max_states} feasible states")
        if b == len(blocks):
            out.append(x.copy())
            return
        rec(b + 1)                                   # block empty
        for i in blocks[b]:
            lhs[:] += A[:, i]
            if np.all(lhs <= rhs + 1e-9):
                x[i] = 1
                rec(b + 1)
                x[i] = 0
            lhs[:] -= A[:, i]

    rec(0)
    X = np.array(out, dtype=np.int8).reshape(-1, n)
    return X[cbp.is_feasible(X)]


def move_graph(F: np.ndarray, moves: Sequence[str] = ("swap", "add_remove"),
               swap_graph: str = "complete"):
    """Sparse symmetric 0/1 adjacency on F: x ~ y iff y is one allowed move from x."""
    from scipy.sparse import coo_matrix

    m, n = F.shape
    weights = (1 << np.arange(n, dtype=np.int64))
    keys = F.astype(np.int64) @ weights
    index: Dict[int, int] = {int(k): a for a, k in enumerate(keys)}
    if swap_graph == "complete":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    elif swap_graph == "ring":
        pairs = sorted({tuple(sorted((i, (i + 1) % n))) for i in range(n)} - {(0, 0)})
    else:
        raise ValueError(swap_graph)
    rows, cols = [], []
    for a, k in enumerate(keys):
        k = int(k)
        if "swap" in moves:
            for i, j in pairs:
                bi, bj = (k >> i) & 1, (k >> j) & 1
                if bi != bj:
                    b = index.get(k ^ (1 << i) ^ (1 << j))
                    if b is not None and b > a:
                        rows.append(a); cols.append(b)
        if "add_remove" in moves:
            for i in range(n):
                b = index.get(k ^ (1 << i))
                if b is not None and b > a:
                    rows.append(a); cols.append(b)
    data = np.ones(len(rows))
    A = coo_matrix((data, (rows, cols)), shape=(m, m))
    return (A + A.T).tocsr()


def connected_components(A) -> int:
    from scipy.sparse.csgraph import connected_components as cc
    return int(cc(A, directed=False)[0])


# ------------------------------------------------------------- the ansatz
class ConstrainedQAOA:
    """QAOA inside the feasible subspace of a :class:`ConstrainedBinaryProgram`.

    ``sample_program(cbp)`` returns a SampleSet of *feasible* decision vectors (shots) and,
    in ``info``, the exact probability of the constrained optimum (``p_opt``), the expected
    objective, the number of feasible states and mixer connectivity.  The cost is
    standardised over F, C' = (f - mean_F f)/std_F f, and the mixer is scaled to unit
    spectral norm so that angles are comparable across instances.
    """

    name = "XY-QAOA"

    def __init__(self, p: int = 3, shots: int = 256, moves: Sequence[str] = ("swap", "add_remove"),
                 swap_graph: str = "complete", max_feasible: int = 4000, optimize: bool = True,
                 maxiter: int = 300, restarts: int = 3, ramp: float = 0.75):
        self.p = p
        self.shots = shots
        self.moves = tuple(moves)
        self.swap_graph = swap_graph
        self.max_feasible = max_feasible
        self.optimize = optimize
        self.maxiter = maxiter
        self.restarts = restarts
        self.ramp = ramp

    def prepare(self, cbp: ConstrainedBinaryProgram, F: Optional[np.ndarray] = None) -> dict:
        F = enumerate_feasible(cbp) if F is None else F
        if len(F) > self.max_feasible:
            raise ValueError(f"|F| = {len(F)} exceeds max_feasible = {self.max_feasible}")
        A = move_graph(F, self.moves, self.swap_graph).toarray()
        lam, V = np.linalg.eigh(A)
        scale = max(np.abs(lam).max(), 1e-12)
        cost = cbp.objective_values(F)
        return {"F": F, "cost": cost, "lam": lam / scale, "V": V, "mixer_norm": scale,
                "components": connected_components(A) if len(F) > 1 else 1}

    @staticmethod
    def state(prep: dict, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
        c = prep["cost"]
        sd = c.std() or 1.0
        cs = (c - c.mean()) / sd
        V, lam = prep["V"], prep["lam"]
        psi = np.full(len(c), 1 / np.sqrt(len(c)), dtype=np.complex128)
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * cs)
            psi = V @ (np.exp(-1j * b * lam) * (V.conj().T @ psi))
        return psi

    def sample_program(self, cbp: ConstrainedBinaryProgram, seed: Optional[int] = None,
                       prep: Optional[dict] = None) -> SampleSet:
        from scipy.optimize import minimize

        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        prep = self.prepare(cbp) if prep is None else prep
        c = prep["cost"]
        sd = c.std() or 1.0
        cs = (c - c.mean()) / sd
        p = self.p

        def f(th):
            psi = self.state(prep, th[:p], th[p:])
            return float(np.real(np.vdot(psi, cs * psi)))

        k = (np.arange(p) + 0.5) / p
        starts = [np.concatenate([self.ramp * k, self.ramp * (1 - k)])]
        starts += [rng.uniform(0, np.pi, 2 * p) for _ in range(self.restarts)]
        best, nfev = None, 0
        for x0 in starts:
            if self.optimize:
                res = minimize(f, x0, method="L-BFGS-B", options={"maxiter": self.maxiter})
                th, val, nfev = res.x, res.fun, nfev + res.nfev
            else:
                th, val = x0, f(x0)
            if best is None or val < best[1]:
                best = (th, val)
            if not self.optimize:
                break
        th = best[0]
        psi = self.state(prep, th[:p], th[p:])
        prob = np.abs(psi) ** 2
        prob /= prob.sum()
        fmin = c.min()
        opt = c <= fmin + 1e-7 * max(1.0, abs(fmin))
        idx = rng.choice(len(c), size=self.shots, p=prob)
        wall = time.perf_counter() - t0
        return SampleSet(prep["F"][idx], c[idx], self.name, {
            "wall_time": wall, "angles": th, "p": p, "nfev": nfev, "p_opt": float(prob[opt].sum()),
            "p_opt_uniform": float(opt.mean()), "expected_objective": float(prob @ c),
            "n_feasible": int(len(c)), "mixer_components": prep["components"],
            "moves": self.moves, "swap_graph": self.swap_graph, "prob": prob,
        })
