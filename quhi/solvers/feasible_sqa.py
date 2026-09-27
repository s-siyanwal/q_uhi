"""Constraint-preserving path-integral Monte Carlo (FeasibleSQA).

Every Trotter slice x^1..x^P is a *feasible* decision vector of a
:class:`~quhi.constraints.ConstrainedBinaryProgram`.  The target distribution is the
Suzuki-Trotter form of Tr exp(-beta (f - Γ Σ σ^x)) restricted to moves that keep every
slice in F:

    pi(X) ∝ exp( -(beta/P) Σ_k f(x^k) + J(Γ) Σ_{k,i} s_i^k s_i^{k+1} ),
    J(Γ) = 1/2 ln coth(beta Γ / P),   s = 2x - 1,   periodic in k.

The problem term is the native objective f(x) (any degree) with beta/P per slice (the
F3 fix, docs/AUDIT.md).  There is no penalty, no Λ and no slack bit.  Moves are the
FeasibleSA neighbourhood applied to one slice at a time (add/remove one bit, or swap an
on-bit with an off-bit); a move that would leave F is rejected before it is evaluated.

This is quantum-*inspired*: a classical Monte Carlo sampler of a Trotterised transverse
field restricted to F, not a model of any hardware.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from numba import njit

from ..constraints import ConstrainedBinaryProgram
from ..schedules import suggest_beta_range
from . import _kernels as K
from .base import SampleSet
from .feasible import FeasibleSA


@njit(cache=True)
def can_move(lhs, A, lo, hi, gcnt, group, i, si, j, sj):
    """Would flipping bit i (change si = ±1) and, if j >= 0, bit j keep the state feasible?
    Assumes the current state is feasible."""
    for r in range(lhs.shape[0]):
        v = lhs[r] + si * A[r, i]
        if j >= 0:
            v += sj * A[r, j]
        if v < lo[r] - 1e-9 or v > hi[r] + 1e-9:
            return False
    gi = group[i]
    if gi >= 0:
        c = gcnt[gi] + si
        if j >= 0 and group[j] == gi:
            c += sj
        if c > 1:
            return False
    if j >= 0:
        gj = group[j]
        if gj >= 0 and gj != gi and gcnt[gj] + sj > 1:
            return False
    return True


@njit(cache=True)
def apply_move(lhs, A, gcnt, group, i, si):
    for r in range(lhs.shape[0]):
        lhs[r] += si * A[r, i]
    if group[i] >= 0:
        gcnt[group[i]] += si


@njit(cache=True)
def _fsqa_run(n, offset, coeff, term_ptr, term_vars, var_ptr, var_terms,
              A, lo, hi, group, n_groups, betas, gammas, P, X0):
    R = X0.shape[0]
    S = betas.shape[0]
    T = coeff.shape[0]
    out = np.empty((R, n), dtype=np.int8)
    out_e = np.empty(R)
    final = np.empty((R, P, n), dtype=np.int8)
    trace = np.empty((R, S))
    for r in range(R):
        X = np.empty((P, n), dtype=np.int8)
        ZC = np.empty((P, T), dtype=np.int64)
        E = np.empty(P)
        L = np.empty((P, A.shape[0]))
        G = np.zeros((P, max(n_groups, 1)), dtype=np.int64)
        for k in range(P):
            X[k] = X0[r]
            ZC[k] = K.zero_counts(X[k], term_ptr, term_vars)
            E[k] = K.energy_of(X[k], offset, coeff, term_ptr, term_vars)
            L[k] = A @ X[k].astype(np.float64)
            for i in range(n):
                if group[i] >= 0 and X[k, i] == 1:
                    G[k, group[i]] += 1
        best_e = E[0]
        best_x = X[0].copy()
        for s in range(S):
            beta = betas[s]
            bp = beta / P
            arg = beta * gammas[s] / P
            if arg > 20.0:
                J = 0.0
            elif arg < 1e-12:
                J = 0.5 * np.log(1e12)
            else:
                J = 0.5 * np.log(1.0 / np.tanh(arg))
            for k in range(P):
                km = (k - 1) % P
                kp = (k + 1) % P
                xk = X[k]
                zk = ZC[k]
                for _ in range(n):
                    i = np.random.randint(n)
                    j = -1
                    if np.random.random() >= 0.5:
                        j = np.random.randint(n)
                        if j == i or xk[j] == xk[i]:
                            j = -1
                    si = 1 - 2 * xk[i]
                    sj = 0
                    if j >= 0:
                        sj = 1 - 2 * xk[j]
                    if not can_move(L[k], A, lo, hi, G[k], group, i, si, j, sj):
                        continue
                    # coupling change uses the old spins: flipping s -> -s changes
                    # -J s (s_prev + s_next) by +2 J s (s_prev + s_next)
                    spin_i = 2 * xk[i] - 1
                    dC = 2.0 * J * spin_i * ((2 * X[km, i] - 1) + (2 * X[kp, i] - 1))
                    dE = K.delta_of(i, xk, zk, coeff, var_ptr, var_terms)
                    K.flip(i, xk, zk, var_ptr, var_terms)
                    if j >= 0:
                        spin_j = 2 * xk[j] - 1                   # j not flipped yet
                        dC += 2.0 * J * spin_j * ((2 * X[km, j] - 1) + (2 * X[kp, j] - 1))
                        dE += K.delta_of(j, xk, zk, coeff, var_ptr, var_terms)
                        K.flip(j, xk, zk, var_ptr, var_terms)
                    dS = bp * dE + dC
                    if dS <= 0.0 or np.random.random() < np.exp(-dS):
                        E[k] += dE
                        apply_move(L[k], A, G[k], group, i, si)
                        if j >= 0:
                            apply_move(L[k], A, G[k], group, j, sj)
                    else:
                        if j >= 0:
                            K.flip(j, xk, zk, var_ptr, var_terms)
                        K.flip(i, xk, zk, var_ptr, var_terms)
            for k in range(P):
                if E[k] < best_e - 1e-12:
                    best_e = E[k]
                    best_x[:] = X[k]
            trace[r, s] = best_e
        out[r] = best_x
        out_e[r] = best_e
        final[r] = X
    return out, out_e, final, trace


class FeasibleSQA:
    """Path-integral quantum-inspired annealing on the feasible set F (no penalty).

    Schedule ("joint" as in SimulatedQuantumAnnealing): beta rises geometrically over
    ``beta_range`` (default: :func:`suggest_beta_range` on f) while Γ falls linearly over
    ``gamma_range``.  ``fixed=(beta, gamma)`` runs at constant beta and Γ (used to check
    the PIMC against exact thermal values).  One sweep = n proposals per slice.
    ``time_limit_s`` repeats batches of ``num_reads`` reads until the next batch would
    start after the limit; only completed batches are returned.
    """

    name = "FeasibleSQA"

    def __init__(self, num_sweeps: int = 500, num_reads: int = 8, trotter_slices: int = 8,
                 beta_range: Optional[tuple] = None, gamma_range: tuple = (0.3, 1e-6),
                 resolution: float = 1e-3, fixed: Optional[tuple] = None,
                 time_limit_s: Optional[float] = None):
        self.num_sweeps = num_sweeps
        self.num_reads = num_reads
        self.P = trotter_slices
        self.beta_range = beta_range
        self.gamma_range = gamma_range
        self.resolution = resolution
        self.fixed = fixed
        self.time_limit_s = time_limit_s

    def _schedule(self, cbp):
        S = self.num_sweeps
        if self.fixed is not None:
            return np.full(S, float(self.fixed[0])), np.full(S, float(self.fixed[1]))
        b0, b1 = self.beta_range or suggest_beta_range(cbp.objective, self.resolution)
        return np.geomspace(b0, b1, S), np.linspace(self.gamma_range[0], self.gamma_range[1], S)

    def sample_program(self, cbp: ConstrainedBinaryProgram, seed: Optional[int] = None,
                       x0: Optional[np.ndarray] = None) -> SampleSet:
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        cp = cbp.objective.compile()
        betas, gammas = self._schedule(cbp)
        A, lo, hi, group, ng = FeasibleSA.constraint_arrays(cbp)
        starter = FeasibleSA(num_reads=self.num_reads)
        Xs, Es, finals, traces = [], [], [], []
        batches = 0
        while True:
            tb = time.perf_counter()
            if x0 is not None and batches == 0:
                X0 = np.atleast_2d(x0).astype(np.int8)
            else:
                X0 = starter.feasible_starts(cbp, rng)
            K.seed_rng(int(rng.integers(2 ** 31)))
            X, E, fin, tr = _fsqa_run(cp.n, cp.offset, cp.coeff, cp.term_ptr, cp.term_vars, cp.var_ptr,
                                      cp.var_terms, A, lo, hi, group, ng, betas, gammas, self.P, X0)
            now = time.perf_counter()
            if self.time_limit_s is not None and batches > 0 and now - t0 > self.time_limit_s:
                break                                   # batch finished after the limit
            Xs.append(X); Es.append(E); finals.append(fin); traces.append(tr)
            batches += 1
            if self.time_limit_s is None or now - t0 + (now - tb) > self.time_limit_s:
                break
        X, E = np.concatenate(Xs), np.concatenate(Es)
        return SampleSet(X, E, self.name, {
            "wall_time": time.perf_counter() - t0, "traces": np.concatenate(traces),
            "final_slices": np.concatenate(finals), "trotter_slices": self.P,
            "batches": batches, "beta_range": (float(betas[0]), float(betas[-1])),
            "gamma_range": (float(gammas[0]), float(gammas[-1])),
        })
