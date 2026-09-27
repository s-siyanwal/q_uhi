"""Tabu search on the feasible set F (classical control for FeasibleSA / FeasibleSQA).

Same neighbourhood as FeasibleSA: add/remove one bit, or swap an on-bit with an off-bit.
Every iteration evaluates all feasible single flips and all (on-bit, off-bit) swaps
(``swap_samples`` random swaps if there are more pairs than that), and takes the best move whose bits are not tabu (aspiration: a tabu move is allowed
if it improves the best objective found).  Flipped bits are tabu for ``tenure`` (+ a random
0..tenure/2) iterations.  The objective is the constrained f(x): no penalty, no slack.
A read ends after ``stall`` non-improving iterations; ``time_limit_s`` repeats reads.
``max_proposals`` (equal-work runs) repeats batches until the number of examined proposals
(n single flips + the swap pairs examined, per iteration) reaches it.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from numba import njit

from ..constraints import ConstrainedBinaryProgram
from . import _kernels as K
from .base import SampleSet
from .feasible import FeasibleSA
from .feasible_sqa import apply_move, can_move


@njit(cache=True)
def _tabu_f_run(n, offset, coeff, term_ptr, term_vars, var_ptr, var_terms,
                A, lo, hi, group, n_groups, X0, max_iter, tenure, swap_samples, stall):
    R = X0.shape[0]
    out = np.empty((R, n), dtype=np.int8)
    out_e = np.empty(R)
    iters = 0
    evals = 0
    for r in range(R):
        x = X0[r].copy()
        zc = K.zero_counts(x, term_ptr, term_vars)
        e = K.energy_of(x, offset, coeff, term_ptr, term_vars)
        lhs = A @ x.astype(np.float64)
        gcnt = np.zeros(max(n_groups, 1), dtype=np.int64)
        for i in range(n):
            if group[i] >= 0 and x[i] == 1:
                gcnt[group[i]] += 1
        tabu_until = np.zeros(n, dtype=np.int64)
        best_e = e
        best_x = x.copy()
        since = 0
        for it in range(max_iter):
            iters += 1
            bd = np.inf
            bi = -1
            bj = -1
            start = np.random.randint(n)
            for q in range(n):
                i = (start + q) % n
                si = 1 - 2 * x[i]
                if not can_move(lhs, A, lo, hi, gcnt, group, i, si, -1, 0):
                    continue
                d = K.delta_of(i, x, zc, coeff, var_ptr, var_terms)
                if tabu_until[i] > it and e + d >= best_e - 1e-12:
                    continue
                if d < bd:
                    bd = d
                    bi = i
                    bj = -1
            # full swap neighbourhood: every (on-bit, off-bit) pair, capped at swap_samples
            # random pairs when the product is larger
            n_on = 0
            for i in range(n):
                n_on += x[i]
            full = n_on * (n - n_on) <= swap_samples
            m = n_on * (n - n_on) if full else swap_samples
            evals += n + m                  # proposals examined this iteration
            on_idx = np.empty(n_on, dtype=np.int64)
            off_idx = np.empty(n - n_on, dtype=np.int64)
            a = 0
            b = 0
            for i in range(n):
                if x[i] == 1:
                    on_idx[a] = i
                    a += 1
                else:
                    off_idx[b] = i
                    b += 1
            for q in range(m):
                if n_on == 0 or n_on == n:
                    break
                if full:
                    i = on_idx[q // (n - n_on)]
                    j = off_idx[q % (n - n_on)]
                else:
                    i = on_idx[np.random.randint(n_on)]
                    j = off_idx[np.random.randint(n - n_on)]
                si = -1
                sj = 1
                if not can_move(lhs, A, lo, hi, gcnt, group, i, si, j, sj):
                    continue
                d = K.delta_of(i, x, zc, coeff, var_ptr, var_terms)
                K.flip(i, x, zc, var_ptr, var_terms)
                d += K.delta_of(j, x, zc, coeff, var_ptr, var_terms)
                K.flip(i, x, zc, var_ptr, var_terms)
                if (tabu_until[i] > it or tabu_until[j] > it) and e + d >= best_e - 1e-12:
                    continue
                if d < bd:
                    bd = d
                    bi = i
                    bj = j
            if bi < 0:                  # every admissible move is tabu: clear the list
                tabu_until[:] = 0
                since += 1
                if since >= stall:
                    break
                continue
            si = 1 - 2 * x[bi]
            K.flip(bi, x, zc, var_ptr, var_terms)
            apply_move(lhs, A, gcnt, group, bi, si)
            tabu_until[bi] = it + tenure + np.random.randint(tenure // 2 + 1)
            if bj >= 0:
                sj = 1 - 2 * x[bj]
                K.flip(bj, x, zc, var_ptr, var_terms)
                apply_move(lhs, A, gcnt, group, bj, sj)
                tabu_until[bj] = it + tenure + np.random.randint(tenure // 2 + 1)
            e += bd
            if e < best_e - 1e-12:
                best_e = e
                best_x[:] = x
                since = 0
            else:
                since += 1
                if since >= stall:
                    break
        out[r] = best_x
        out_e[r] = best_e
    return out, out_e, iters, evals


class TabuOnF:
    """Multi-start tabu search on feasible decision vectors (see module docstring)."""

    name = "Tabu-on-F"

    def __init__(self, max_iter: int = 2000, num_reads: int = 8, tenure: Optional[int] = None,
                 swap_samples: Optional[int] = None, stall: Optional[int] = None,
                 time_limit_s: Optional[float] = None, max_proposals: Optional[int] = None):
        self.max_proposals = max_proposals
        self.max_iter = max_iter
        self.num_reads = num_reads
        self.tenure = tenure
        self.swap_samples = swap_samples
        self.stall = stall
        self.time_limit_s = time_limit_s

    def sample_program(self, cbp: ConstrainedBinaryProgram, seed: Optional[int] = None,
                       x0: Optional[np.ndarray] = None) -> SampleSet:
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        cp = cbp.objective.compile()
        n = cp.n
        tenure = self.tenure if self.tenure is not None else max(2, min(20, n // 4))
        swaps = self.swap_samples if self.swap_samples is not None else 4000
        stall = self.stall if self.stall is not None else max(100, 5 * n)
        A, lo, hi, group, ng = FeasibleSA.constraint_arrays(cbp)
        starter = FeasibleSA(num_reads=self.num_reads)
        Xs, Es, iters, batches, evals = [], [], 0, 0, 0
        while True:
            tb = time.perf_counter()
            X0 = np.atleast_2d(x0).astype(np.int8) if (x0 is not None and batches == 0) \
                else starter.feasible_starts(cbp, rng)
            K.seed_rng(int(rng.integers(2 ** 31)))
            X, E, it, ev = _tabu_f_run(cp.n, cp.offset, cp.coeff, cp.term_ptr, cp.term_vars, cp.var_ptr,
                                   cp.var_terms, A, lo, hi, group, ng, X0, self.max_iter, tenure, swaps, stall)
            now = time.perf_counter()
            if self.time_limit_s is not None and batches > 0 and now - t0 > self.time_limit_s:
                break
            Xs.append(X); Es.append(E); iters += it; batches += 1; evals += ev
            if self.max_proposals is not None:
                if evals >= self.max_proposals:
                    break
                continue
            if self.time_limit_s is None or now - t0 + (now - tb) > self.time_limit_s:
                break
        return SampleSet(np.concatenate(Xs), np.concatenate(Es), self.name, {
            "wall_time": time.perf_counter() - t0, "iterations": iters, "tenure": tenure,
            "swap_samples": swaps, "batches": batches, "proposals": evals})
