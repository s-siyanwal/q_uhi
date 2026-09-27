"""Classical heuristics: simulated annealing, tabu search, greedy descent, random."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..polynomial import CompiledPolynomial
from . import _kernels as K
from .base import SampleSet, Solver, cp_args, energy_scale, initial_states


def geometric_betas(beta0: float, beta1: float, num_sweeps: int) -> np.ndarray:
    return np.geomspace(beta0, beta1, num_sweeps)


class SimulatedAnnealing(Solver):
    """Metropolis single-flip SA (Kirkpatrick et al. 1983) with a geometric beta schedule.

    Default schedule follows the dwave-neal heuristic: the hottest beta accepts
    the largest possible uphill flip with probability 1/2, the coldest accepts
    the smallest coefficient-sized uphill flip with probability 1/100.
    """

    name = "SA"

    def __init__(self, num_sweeps: int = 1000, num_reads: int = 32,
                 beta_range: Optional[tuple] = None, trace_every: int = 1):
        self.num_sweeps = num_sweeps
        self.num_reads = num_reads
        self.beta_range = beta_range
        self.trace_every = trace_every

    def default_beta_range(self, cp: CompiledPolynomial) -> tuple:
        sc = energy_scale(cp)
        return (np.log(2.0) / sc["max_delta"], np.log(100.0) / sc["min_coeff"])

    def _sample(self, cp, rng, x0: Optional[np.ndarray] = None, **_) -> SampleSet:
        b0, b1 = self.beta_range or self.default_beta_range(cp)
        betas = geometric_betas(b0, b1, self.num_sweeps)
        K.seed_rng(int(rng.integers(2 ** 31)))
        X0 = initial_states(cp, rng, self.num_reads, x0)
        X, E, tr = K.sa_run(*cp_args(cp), betas, X0, self.num_reads, self.trace_every)
        return SampleSet(X, E, self.name, {"traces": tr, "beta_range": (b0, b1),
                                           "num_sweeps": self.num_sweeps,
                                           "flips_attempted": self.num_reads * self.num_sweeps * cp.n})


class TabuSearch(Solver):
    """Single-flip tabu search with aspiration and randomised tenure; multi-start."""

    name = "Tabu"

    def __init__(self, max_iter: int = 2000, num_reads: int = 32, tenure: Optional[int] = None,
                 stall_limit: Optional[int] = None):
        self.max_iter = max_iter
        self.num_reads = num_reads
        self.tenure = tenure
        self.stall_limit = stall_limit

    def _sample(self, cp, rng, x0: Optional[np.ndarray] = None, **_) -> SampleSet:
        tenure = self.tenure if self.tenure is not None else max(1, min(20, cp.n // 4))
        stall = self.stall_limit if self.stall_limit is not None else max(200, 20 * cp.n)
        K.seed_rng(int(rng.integers(2 ** 31)))
        X0 = initial_states(cp, rng, self.num_reads, x0)
        X, E, tr = K.tabu_run(*cp_args(cp), X0, self.num_reads, self.max_iter, tenure, stall)
        return SampleSet(X, E, self.name, {"traces": tr, "tenure": tenure,
                                           "flips_attempted": self.num_reads * self.max_iter * cp.n})


class SteepestDescent(Solver):
    """Greedy 1-flip descent from random starts (weak baseline / polishing step)."""

    name = "Greedy"

    def __init__(self, num_reads: int = 32):
        self.num_reads = num_reads

    def _sample(self, cp, rng, x0: Optional[np.ndarray] = None, **_) -> SampleSet:
        X = initial_states(cp, rng, self.num_reads, x0).copy()
        E = K.steepest_descent(*cp_args(cp), X)
        return SampleSet(X, E, self.name, {})


class RandomSampler(Solver):
    """Uniform random assignments (sanity baseline)."""

    name = "Random"

    def __init__(self, num_reads: int = 32):
        self.num_reads = num_reads

    def _sample(self, cp, rng, **_) -> SampleSet:
        X = rng.integers(0, 2, size=(self.num_reads, cp.n)).astype(np.int8)
        E = np.array([K.energy_of(x, cp.offset, cp.coeff, cp.term_ptr, cp.term_vars) for x in X])
        return SampleSet(X, E, self.name, {})


def polish(cp: CompiledPolynomial, X: np.ndarray) -> tuple:
    """Apply steepest descent to each row of X (copy); returns (X, energies)."""
    X = np.array(X, dtype=np.int8, copy=True)
    E = K.steepest_descent(*cp_args(cp), X)
    return X, E
