"""Run solvers on encoded UHI programs and score them against certified optima."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..constraints import ConstrainedBinaryProgram, PenaltyEncoding
from ..polynomial import BinaryPolynomial
from ..postprocess import postprocess as _postprocess
from ..quadratize import Quadratization, quadratize
from ..solvers.base import SampleSet, Solver
from ..solvers.exact import MILPSolver
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


def build_instance(name: str, program: ConstrainedBinaryProgram, penalty_mult: float = 1.0,
                   quadratized: bool = False, meta: Optional[dict] = None,
                   f_opt: Optional[float] = None) -> Instance:
    """Encode ``program`` with ``penalty_mult`` x the certified-safe penalty weight and
    certify its constrained optimum with MILP (unless ``f_opt`` is given)."""
    safe = program.to_penalty_model()
    enc = program.to_penalty_model(penalty_weight=penalty_mult * safe.penalty_weight) \
        if penalty_mult != 1.0 else safe
    q = quadratize(enc.model) if quadratized and not enc.model.is_quadratic else None
    model = q.qubo if q is not None else enc.model
    if f_opt is None:
        ms = MILPSolver().solve(program)
        if not ms.info["optimal"]:
            raise RuntimeError(f"MILP did not certify optimality for {name}: {ms.info['message']}")
        f_opt = ms.best_energy
    f0 = float(program.objective_values(np.zeros(program.n))[0])
    m = {"penalty_mult": penalty_mult, "penalty_weight": enc.penalty_weight,
         "safe_penalty_weight": safe.penalty_weight, "quadratized": q is not None,
         "n_slack": enc.model.n - program.n, "n_aux": (q.qubo.n - enc.model.n) if q else 0,
         **(meta or {})}
    return Instance(name, program, enc, model, float(f_opt), f0, q, m)


def score(inst: Instance, ss: SampleSet, postprocess: bool = False) -> Dict[str, float]:
    """Constrained-problem metrics for a sample set.

    A read "succeeds" if its decision part is feasible AND attains the certified
    constrained optimum f_opt of the original objective.  With ``postprocess``
    the same metrics are also reported (suffix ``_pp``) after repair + feasible
    local search (quhi.postprocess), including its extra time.
    """
    out = _score_decisions(inst, inst.decision(ss.samples), ss.wall_time)
    if postprocess:
        t0 = time.perf_counter()
        Xp = _postprocess(inst.program, inst.decision(ss.samples))
        tpp = time.perf_counter() - t0
        pp = _score_decisions(inst, Xp, ss.wall_time + tpp)
        out.update({f"{k}_pp": v for k, v in pp.items() if k not in ("reads",)})
    return out


def _score_decisions(inst: Instance, Xd: np.ndarray, wall: float) -> Dict[str, float]:
    feas = inst.program.is_feasible(Xd)
    f = inst.program.objective_values(Xd)
    ok = feas & (f <= inst.f_opt + 1e-7 * max(1.0, abs(inst.f_opt)))
    k, n = int(ok.sum()), len(ok)
    t_read = wall / max(n, 1)
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
        "wall_time_s": wall,
    }


def run(inst: Instance, solvers: Dict[str, Solver], seeds: Sequence[int] = (0,),
        extra: Optional[dict] = None, keep: bool = False, postprocess: bool = False):
    rows, kept = [], {}
    for label, s in solvers.items():
        for seed in seeds:
            ss = s.sample(inst.model, seed=seed)
            row = {"instance": inst.name, "solver": label, "seed": seed, "n_vars": inst.model.n,
                   "n_decision": inst.program.n, "degree": inst.model.degree,
                   **(inst.meta or {}), **(extra or {}), **score(inst, ss, postprocess)}
            rows.append(row)
            if keep:
                kept.setdefault(label, []).append(ss)
    return (rows, kept) if keep else rows
