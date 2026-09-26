"""Feasible-space simulated annealing: the fair classical annealer for constrained programs.

:class:`~quhi.solvers.classical.SimulatedAnnealing` flips one bit of the penalty model
f + ΛP, including slack bits.  A legal budget change then needs a coordinated chain of
slack flips that single-flip Metropolis almost never proposes (docs/RESULTS.md, E3/E4).
FeasibleSA instead walks on the decision vector only and never leaves the feasible set:

* ``add/remove``: flip one bit;
* ``swap``: turn one on-bit off and one off-bit on (this covers equal-cost exchanges,
  option changes inside a cell, and moves between cells).

A proposal is rejected outright if the result violates any linear constraint or
at-most-one group.  The energy is the constrained objective f(x); there is no penalty
and no Λ.  Starting points are feasible (random start + ``quhi.postprocess.repair``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit

from ..constraints import ConstrainedBinaryProgram, greedy_feasible
from ..postprocess import repair
from ..schedules import suggest_beta_range
from . import _kernels as K
from .base import SampleSet


@njit(cache=True)
def _ok(lhs, lo, hi, gcnt):
    for r in range(lhs.shape[0]):
        if lhs[r] < lo[r] - 1e-9 or lhs[r] > hi[r] + 1e-9:
            return False
    for g in range(gcnt.shape[0]):
        if gcnt[g] > 1:
            return False
    return True


@njit(cache=True)
def _fsa_run(n, offset, coeff, term_ptr, term_vars, var_ptr, var_terms,
             A, lo, hi, group, n_groups, betas, X0, trace_every):
    R = X0.shape[0]
    S = betas.shape[0]
    out = np.empty((R, n), dtype=np.int8)
    out_e = np.empty(R)
    ntr = (S + trace_every - 1) // trace_every
    trace = np.empty((R, ntr))
    accepted = 0
    for r in range(R):
        x = X0[r].copy()
        zc = K.zero_counts(x, term_ptr, term_vars)
        e = K.energy_of(x, offset, coeff, term_ptr, term_vars)
        lhs = A @ x.astype(np.float64)
        gcnt = np.zeros(max(n_groups, 1), dtype=np.int64)
        for i in range(n):
            if group[i] >= 0 and x[i] == 1:
                gcnt[group[i]] += 1
        best_e = e
        best_x = x.copy()
        for s in range(S):
            b = betas[s]
            for _ in range(n):
                i = np.random.randint(n)
                if np.random.random() < 0.5:
                    j = -1                                   # add / remove
                else:
                    j = np.random.randint(n)                 # swap: need x_i != x_j
                    if j == i or x[j] == x[i]:
                        j = -1
                # apply tentatively to feasibility bookkeeping
                si = 1 - 2 * x[i]
                lhs += si * A[:, i]
                if group[i] >= 0:
                    gcnt[group[i]] += si
                if j >= 0:
                    sj = 1 - 2 * x[j]
                    lhs += sj * A[:, j]
                    if group[j] >= 0:
                        gcnt[group[j]] += sj
                feasible = _ok(lhs, lo, hi, gcnt)
                if feasible:
                    d = K.delta_of(i, x, zc, coeff, var_ptr, var_terms)
                    K.flip(i, x, zc, var_ptr, var_terms)
                    if j >= 0:
                        d += K.delta_of(j, x, zc, coeff, var_ptr, var_terms)
                        K.flip(j, x, zc, var_ptr, var_terms)
                    if d <= 0.0 or np.random.random() < np.exp(-b * d):
                        e += d
                        accepted += 1
                        continue
                    if j >= 0:
                        K.flip(j, x, zc, var_ptr, var_terms)
                    K.flip(i, x, zc, var_ptr, var_terms)
                # undo bookkeeping (x is back to the pre-move state here)
                lhs -= si * A[:, i]
                if group[i] >= 0:
                    gcnt[group[i]] -= si
                if j >= 0:
                    sj = 1 - 2 * x[j]
                    lhs -= sj * A[:, j]
                    if group[j] >= 0:
                        gcnt[group[j]] -= sj
            if e < best_e - 1e-12:
                best_e = e
                best_x[:] = x
            if s % trace_every == 0:
                trace[r, s // trace_every] = best_e
        out[r] = best_x
        out_e[r] = best_e
    return out, out_e, trace, accepted


class FeasibleSA:
    """Metropolis SA on feasible decision vectors of a ConstrainedBinaryProgram.

    ``beta_range`` defaults to :func:`quhi.schedules.suggest_beta_range` on the
    *constrained objective* (no slack, no penalty) with the given ``resolution`` (°C).
    One sweep = n proposals; a swap proposal flips two bits.
    """

    name = "FeasibleSA"

    def __init__(self, num_sweeps: int = 1000, num_reads: int = 32, beta_range: Optional[tuple] = None,
                 resolution: float = 1e-3, trace_every: int = 1):
        self.num_sweeps = num_sweeps
        self.num_reads = num_reads
        self.beta_range = beta_range
        self.resolution = resolution
        self.trace_every = trace_every

    @staticmethod
    def constraint_arrays(cbp: ConstrainedBinaryProgram):
        n = cbp.n
        A = np.zeros((len(cbp.constraints), n))
        lo = np.full(len(cbp.constraints), -np.inf)
        hi = np.full(len(cbp.constraints), np.inf)
        for r, c in enumerate(cbp.constraints):
            for i, a in c.coeffs.items():
                A[r, i] = a
            if c.sense in ("<=", "=="):
                hi[r] = c.rhs
            if c.sense in (">=", "=="):
                lo[r] = c.rhs
        group = np.full(n, -1, dtype=np.int64)
        for g, members in enumerate(cbp.at_most_one):
            group[list(members)] = g
        return A, lo, hi, group, len(cbp.at_most_one)

    def feasible_starts(self, cbp: ConstrainedBinaryProgram, rng: np.random.Generator) -> np.ndarray:
        X0 = []
        for _ in range(self.num_reads):
            x = repair(cbp, rng.integers(0, 2, cbp.n).astype(np.int8))
            tries = 0
            while not cbp.is_feasible(x)[0] and tries < 20:
                x = repair(cbp, rng.integers(0, 2, cbp.n).astype(np.int8) * (rng.random(cbp.n) < 0.2))
                tries += 1
            if not cbp.is_feasible(x)[0]:
                x = greedy_feasible(cbp, rng)           # multi-start violation descent
                if x is None:
                    raise RuntimeError("could not find a feasible starting point")
            X0.append(x)
        return np.array(X0, dtype=np.int8)

    def sample_program(self, cbp: ConstrainedBinaryProgram, seed: Optional[int] = None,
                       x0: Optional[np.ndarray] = None) -> SampleSet:
        import time

        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        cp = cbp.objective.compile()
        b0, b1 = self.beta_range or suggest_beta_range(cbp.objective, self.resolution)
        betas = np.geomspace(b0, b1, self.num_sweeps)
        X0 = self.feasible_starts(cbp, rng) if x0 is None else np.atleast_2d(x0).astype(np.int8)
        A, lo, hi, group, ng = self.constraint_arrays(cbp)
        K.seed_rng(int(rng.integers(2 ** 31)))
        X, E, tr, acc = _fsa_run(cp.n, cp.offset, cp.coeff, cp.term_ptr, cp.term_vars, cp.var_ptr,
                                 cp.var_terms, A, lo, hi, group, ng, betas, X0, self.trace_every)
        attempts = X0.shape[0] * self.num_sweeps * cp.n
        return SampleSet(X, E, self.name, {
            "wall_time": time.perf_counter() - t0, "traces": tr, "beta_range": (b0, b1),
            "flips_attempted": attempts, "acceptance": acc / max(attempts, 1),
        })
