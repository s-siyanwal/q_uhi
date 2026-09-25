"""Run solvers on encoded UHI programs and score them against certified optima."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..constraints import ConstrainedBinaryProgram, PenaltyEncoding
from ..polynomial import BinaryPolynomial
from ..quadratize import Quadratization
from ..solvers.base import SampleSet, Solver
from .metrics import benefit_ratio, success_probability, tts, wilson_interval


@dataclass
class Instance:
    """A constrained program plus the unconstrained model actually handed to solvers."""

    name: str
    program: ConstrainedBinaryProgram
    encoding: PenaltyEncoding
    model: BinaryPolynomial                     # = encoding.model, or its quadratisation
    f_opt: float                                 # certified constrained optimum of program.objective
    f_baseline: float                            # objective of "do nothing" (x = 0)
    quadratization: Optional[Quadratization] = None
    meta: Optional[dict] = None

    def decision(self, X: np.ndarray) -> np.ndarray:
        return np.atleast_2d(X)[:, : self.program.n]


def score(inst: Instance, ss: SampleSet) -> Dict[str, float]:
    """Constrained-problem metrics for a sample set.

    A read "succeeds" if its decision part is feasible AND attains the certified
    constrained optimum f_opt of the original objective.
    """
    Xd = inst.decision(ss.samples)
    feas = inst.program.is_feasible(Xd)
    f = inst.program.objective_values(Xd)
    ok = feas & (f <= inst.f_opt + 1e-7 * max(1.0, abs(inst.f_opt)))
    k, n = int(ok.sum()), len(ok)
    t_read = ss.wall_time / max(n, 1)
    br = benefit_ratio(np.where(feas, f, inst.f_baseline), inst.f_baseline, inst.f_opt)
    lo, hi = wilson_interval(k, n)
    best_feas = float(f[feas].min()) if feas.any() else np.nan
    return {
        "reads": n,
        "p_success": k / n,
        "p_success_lo": lo,
        "p_success_hi": hi,
        "p_feasible": float(feas.mean()),
        "best_feasible_obj": best_feas,
        "best_gap_C": best_feas - inst.f_opt if feas.any() else np.nan,
        "mean_benefit_ratio": float(br.mean()),
        "best_benefit_ratio": float(br.max()),
        "time_per_read_s": t_read,
        "tts99_s": tts(k / n, t_read),
        "wall_time_s": ss.wall_time,
    }


def run(inst: Instance, solvers: Dict[str, Solver], seeds: Sequence[int] = (0,),
        extra: Optional[dict] = None, keep: bool = False) -> List[dict]:
    rows, kept = [], {}
    for label, s in solvers.items():
        for seed in seeds:
            ss = s.sample(inst.model, seed=seed)
            row = {"instance": inst.name, "solver": label, "seed": seed, "n_vars": inst.model.n,
                   "n_decision": inst.program.n, "degree": inst.model.degree,
                   **(inst.meta or {}), **(extra or {}), **score(inst, ss)}
            rows.append(row)
            if keep:
                kept.setdefault(label, []).append(ss)
    return (rows, kept) if keep else rows
