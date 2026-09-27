"""Exact solvers: exhaustive enumeration and MILP (HiGHS via scipy)."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..constraints import ConstrainedBinaryProgram
from ..polynomial import BinaryPolynomial, CompiledPolynomial
from . import _kernels as K
from .base import SampleSet, Solver, cp_args


class ExhaustiveSolver(Solver):
    """Gray-code enumeration of all 2^n states (n <= ~30).  Certifies optimality."""

    name = "exhaustive"

    def __init__(self, max_n: int = 30):
        self.max_n = max_n

    def _sample(self, cp: CompiledPolynomial, rng, keep_all: bool = False) -> SampleSet:
        if cp.n > self.max_n:
            raise ValueError(f"n={cp.n} too large for exhaustive search (max_n={self.max_n})")
        e, bits, ndeg, energies = K.brute_force(*cp_args(cp), keep_all)
        info = {"n_degenerate": int(ndeg)}
        if keep_all:
            info["all_energies"] = energies
        return SampleSet(bits[None, :], np.array([e]), self.name, info)


def all_energies(model: BinaryPolynomial) -> np.ndarray:
    """Energy of every basis state, little-endian index b = sum_i x_i 2^i."""
    cp = model.compile()
    return K.brute_force(*cp_args(cp), True)[3]


class MILPSolver:
    """Linearise the (HUBO) objective and solve the *constrained* program with HiGHS.

    Each term T gets y_T in [0,1] with y_T <= x_i (i in T) and
    y_T >= sum_{i in T} x_i - (|T| - 1)  (standard product linearisation, exact
    for binary x).  Linear constraints are passed natively - no penalties, no
    slack bits - so this is the ground truth for the constrained problem.
    """

    name = "milp"

    def __init__(self, time_limit: float = 600.0, mip_rel_gap: float = 0.0):
        self.time_limit = time_limit
        self.mip_rel_gap = mip_rel_gap

    def solve(self, problem: "ConstrainedBinaryProgram | BinaryPolynomial") -> SampleSet:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix

        cbp = problem if isinstance(problem, ConstrainedBinaryProgram) else \
            ConstrainedBinaryProgram(problem)
        f = cbp.objective
        n = f.n
        lin = np.zeros(n)
        prods = []
        for t, c in f.terms.items():
            if len(t) == 1:
                lin[t[0]] += c
            else:
                prods.append((t, c))
        m = len(prods)
        nv = n + m
        cvec = np.concatenate([lin, [c for _, c in prods]])
        rows, cols, vals, lo, hi = [], [], [], [], []
        r = 0
        for k, (t, c) in enumerate(prods):
            y = n + k
            # only the side of the linearisation that binds for this sign is needed,
            # but adding both keeps y exact for all x and is cheap.
            for i in t:                                   # y - x_i <= 0
                rows += [r, r]; cols += [y, i]; vals += [1.0, -1.0]; lo.append(-np.inf); hi.append(0.0); r += 1
            rows.append(r); cols.append(y); vals.append(1.0)   # y - sum x_i >= 1 - |T|
            for i in t:
                rows.append(r); cols.append(i); vals.append(-1.0)
            lo.append(1.0 - len(t)); hi.append(np.inf); r += 1
        for con in cbp.constraints:
            for i, a in con.coeffs.items():
                rows.append(r); cols.append(i); vals.append(float(a))
            if con.sense == "<=":
                lo.append(-np.inf); hi.append(con.rhs)
            elif con.sense == ">=":
                lo.append(con.rhs); hi.append(np.inf)
            else:
                lo.append(con.rhs); hi.append(con.rhs)
            r += 1
        for g in cbp.at_most_one:
            for i in g:
                rows.append(r); cols.append(i); vals.append(1.0)
            lo.append(-np.inf); hi.append(1.0); r += 1
        integrality = np.concatenate([np.ones(n), np.zeros(m)])
        cons = []
        if r:
            A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
            cons = [LinearConstraint(A, lo, hi)]
        t0 = time.perf_counter()
        res = milp(cvec, constraints=cons, integrality=integrality, bounds=Bounds(0, 1),
                   options={"time_limit": self.time_limit, "mip_rel_gap": self.mip_rel_gap,
                            "disp": False})
        wall = time.perf_counter() - t0
        if res.x is None:
            raise RuntimeError(f"MILP failed: {res.message}")
        x = np.round(res.x[:n]).astype(np.int8)
        e = f.energy(x)
        info = {"wall_time": wall, "status": int(res.status), "message": res.message,
                "optimal": res.status == 0, "mip_gap": getattr(res, "mip_gap", None),
                "n_aux": m}
        return SampleSet(x[None, :], np.array([e]), self.name, info)
