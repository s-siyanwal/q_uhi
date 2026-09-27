"""QAOA-seeded FeasibleSA: a declared hybrid.

If the feasible set is small enough for the subspace simulation (|F| <= max_feasible),
ConstrainedQAOA is run at p=2 and then p=4 (one restart each, existing L-BFGS-B), K
seeds are drawn from the final |ψ|², and FeasibleSA runs from those seeds for the rest of
the wall-clock budget.  Otherwise no QAOA is run and FeasibleSA is seeded from the greedy
planner (repaired) plus random feasible plans; the result is labelled "no quantum seed".
The split of wall-clock between QAOA and SA is logged.  ConstrainedQAOA is a state-vector
simulation on span(F), not a compiled circuit.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..constraints import ConstrainedBinaryProgram
from ..postprocess import repair
from .base import SampleSet
from .feasible import FeasibleSA
from .mixers import ConstrainedQAOA, enumerate_feasible


class QaoaSeededFeasibleSA:
    name = "QAOA-seeded FeasibleSA"

    def __init__(self, time_limit_s: float, k_seeds: int = 4, ps=(2, 4), max_feasible: int = 4000,
                 sa_sweeps: int = 1000, moves=("swap", "add_remove"), greedy: Optional[np.ndarray] = None):
        self.time_limit_s = time_limit_s
        self.k_seeds = k_seeds
        self.ps = tuple(ps)
        self.max_feasible = max_feasible
        self.sa_sweeps = sa_sweeps
        self.moves = tuple(moves)
        self.greedy = greedy

    def sample_program(self, cbp: ConstrainedBinaryProgram, seed: Optional[int] = None,
                       F: Optional[np.ndarray] = None) -> SampleSet:
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        info = {"quantum_seed": False, "qaoa_p_opt": None}
        if F is None:
            try:
                F = enumerate_feasible(cbp, max_states=self.max_feasible)
            except ValueError:
                F = None
        seeds = None
        if F is not None and len(F) <= self.max_feasible:
            q0 = ConstrainedQAOA(moves=self.moves)
            prep = q0.prepare(cbp, F)
            prob = None
            for p in self.ps:
                left = self.time_limit_s - (time.perf_counter() - t0)
                if left <= 0.25 * self.time_limit_s and prob is not None:
                    break
                q = ConstrainedQAOA(p=p, moves=self.moves, restarts=1, shots=1,
                                    time_limit_s=max(0.05, left - 0.25 * self.time_limit_s))
                ss = q.sample_program(cbp, seed=int(rng.integers(2 ** 31)), prep=prep)
                prob = ss.info["prob"]
                info["qaoa_p"] = p
            seeds = F[rng.choice(len(F), size=self.k_seeds, p=prob)]
            info.update(quantum_seed=True)
        else:
            X = [repair(cbp, self.greedy)] if self.greedy is not None else []
            X += list(FeasibleSA(num_reads=self.k_seeds).feasible_starts(cbp, rng))
            seeds = np.array([x for x in X if cbp.is_feasible(x)[0]][: self.k_seeds], dtype=np.int8)
        t_q = time.perf_counter() - t0
        left = max(0.0, self.time_limit_s - t_q)
        sa = FeasibleSA(num_sweeps=self.sa_sweeps, num_reads=len(seeds), time_limit_s=left)
        ss = sa.sample_program(cbp, seed=int(rng.integers(2 ** 31)), x0=seeds)
        wall = time.perf_counter() - t0
        info.update(wall_time=wall, t_qaoa=t_q, frac_qaoa=t_q / max(wall, 1e-12),
                    n_feasible=None if F is None else len(F))
        return SampleSet(ss.samples, ss.energies, self.name, info)
