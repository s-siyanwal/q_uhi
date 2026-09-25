"""Constraint-aware post-processing of raw solver samples.

Penalty models trade feasibility against landscape ruggedness (docs/RESULTS.md,
experiment E4): large weights make every sample feasible but trap local search,
small weights give good but often infeasible samples.  The standard remedy
(used e.g. by hybrid annealing workflows) is classical post-processing:

1. **repair**: greedily flip the variable that most reduces total violation
   (ties broken by objective) until feasible;
2. **feasible descent**: best-improvement over 1-flip and 2-flip (swap) moves that
   keep feasibility.

This works on the *decision* variables of the original constrained program, so
it is independent of the penalty weight, slack encoding, and quadratisation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .constraints import ConstrainedBinaryProgram


def repair(cbp: ConstrainedBinaryProgram, x: np.ndarray, max_steps: Optional[int] = None) -> np.ndarray:
    x = np.array(x[: cbp.n], dtype=np.int8, copy=True)
    n = cbp.n
    for _ in range(max_steps or 4 * n):
        v = cbp.violations(x)[0]
        if v == 0:
            break
        Y = np.repeat(x[None, :], n, axis=0)
        Y[np.arange(n), np.arange(n)] ^= 1
        vy = cbp.violations(Y)
        fy = cbp.objective_values(Y)
        best = np.lexsort((fy, vy))[0]
        if vy[best] >= v:
            break
        x = Y[best]
    return x


def feasible_local_search(cbp: ConstrainedBinaryProgram, x: np.ndarray, swaps: bool = True,
                          max_steps: int = 10_000) -> np.ndarray:
    x = np.array(x[: cbp.n], dtype=np.int8, copy=True)
    n = cbp.n
    f = cbp.objective_values(x)[0]
    for _ in range(max_steps):
        Y = np.repeat(x[None, :], n, axis=0)
        Y[np.arange(n), np.arange(n)] ^= 1
        if swaps:
            on, off = np.flatnonzero(x == 1), np.flatnonzero(x == 0)
            if on.size and off.size:
                a, b = np.meshgrid(on, off, indexing="ij")
                Z = np.repeat(x[None, :], a.size, axis=0)
                Z[np.arange(a.size), a.ravel()] = 0
                Z[np.arange(a.size), b.ravel()] = 1
                Y = np.vstack([Y, Z])
        ok = cbp.is_feasible(Y)
        if not ok.any():
            break
        fy = np.where(ok, cbp.objective_values(Y), np.inf)
        i = int(np.argmin(fy))
        if fy[i] >= f - 1e-12:
            break
        x, f = Y[i], fy[i]
    return x


def postprocess(cbp: ConstrainedBinaryProgram, X: np.ndarray, swaps: bool = True) -> np.ndarray:
    """Repair + feasible local search for every row of X (decision part)."""
    X = np.atleast_2d(X)
    out = np.empty((X.shape[0], cbp.n), dtype=np.int8)
    for r, x in enumerate(X):
        y = repair(cbp, x)
        out[r] = feasible_local_search(cbp, y, swaps) if cbp.is_feasible(y)[0] else y
    return out
