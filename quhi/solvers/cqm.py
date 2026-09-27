"""Constrained-quadratic-model (CQM) export, for hybrid solvers that take constraints natively.

``cqm_spec`` needs no dependency: it returns the objective polynomial and every constraint
(linear budget / equity rows and one-hot "at most one" groups) as plain data, and
``spec_is_feasible`` checks a plan against it.  ``to_cqm`` builds a ``dimod.CQM`` from the
same spec when dimod is importable.  ``sample_cqm`` is a skip-hook for Leap's hybrid CQM
solver: it returns ``(None, "skipped: ...")`` unless ``DWAVE_API_TOKEN`` (or ``token``) is set and
dwave-system is installed.  It only ever submits the CQM (native constraints); it never builds a
slack-penalty QUBO for a QPU.  No D-Wave result is reported anywhere in this repository.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..constraints import ConstrainedBinaryProgram


def cqm_spec(cbp: ConstrainedBinaryProgram) -> Dict:
    cons: List[Dict] = []
    for c in cbp.constraints:
        cons.append({"label": f"{c.name}_{len(cons)}", "coeffs": dict(c.coeffs), "sense": c.sense,
                     "rhs": float(c.rhs)})
    for g, members in enumerate(cbp.at_most_one):
        cons.append({"label": f"one_hot_{g}", "coeffs": {int(i): 1 for i in members}, "sense": "<=", "rhs": 1.0})
    return {"n": cbp.n, "objective": cbp.objective, "constraints": cons}


def spec_is_feasible(spec: Dict, X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(X)
    ok = np.ones(len(X), dtype=bool)
    for c in spec["constraints"]:
        lhs = sum(a * X[:, i] for i, a in c["coeffs"].items())
        if c["sense"] == "<=":
            ok &= lhs <= c["rhs"] + 1e-9
        elif c["sense"] == ">=":
            ok &= lhs >= c["rhs"] - 1e-9
        else:
            ok &= np.abs(lhs - c["rhs"]) <= 1e-9
    return ok


def to_cqm(cbp: ConstrainedBinaryProgram):
    """dimod.ConstrainedQuadraticModel with f as objective (degree <= 2 required)."""
    import dimod  # noqa: F401  (optional dependency)

    if not cbp.objective.is_quadratic:
        raise ValueError("CQM objectives must be at most quadratic")
    spec = cqm_spec(cbp)
    x = [dimod.Binary(i) for i in range(cbp.n)]
    obj = cbp.objective.offset
    for vs, c in cbp.objective.terms.items():
        t = c
        for v in vs:
            t = t * x[v]
        obj = obj + t
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(obj)
    for c in spec["constraints"]:
        lhs = sum(a * x[i] for i, a in c["coeffs"].items())
        cqm.add_constraint(lhs <= c["rhs"] if c["sense"] == "<=" else
                           (lhs >= c["rhs"] if c["sense"] == ">=" else lhs == c["rhs"]), label=c["label"])
    return cqm


def sample_cqm(cbp: ConstrainedBinaryProgram, token: Optional[str] = None, label: str = "quhi",
               time_limit: Optional[float] = None) -> Tuple[Optional[object], str]:
    """Submit ``to_cqm(cbp)`` to LeapHybridCQMSampler.  Returns (SampleSet | None, status).

    Without a token nothing is imported and no network call is made.  Returned plans are
    re-scored with ``cbp`` (objective and feasibility) rather than trusting the solver's flags.
    """
    token = token or os.environ.get("DWAVE_API_TOKEN")
    if not token:
        return None, "skipped: no DWAVE_API_TOKEN"
    if not cbp.objective.is_quadratic:
        return None, "skipped: objective degree > 2 (CQM needs a quadratic objective)"
    try:
        from dwave.system import LeapHybridCQMSampler
    except ImportError:
        return None, "skipped: dwave-system not installed (pip install -e .[dwave])"
    from .base import SampleSet

    kw = {"label": label}
    if time_limit is not None:
        kw["time_limit"] = time_limit
    res = LeapHybridCQMSampler(token=token).sample_cqm(to_cqm(cbp), **kw)
    order = [res.variables.index(i) for i in range(cbp.n)]
    X = np.asarray(res.record.sample)[:, order].astype(np.int8)
    feas = cbp.is_feasible(X)
    return SampleSet(X, cbp.objective_values(X), "LeapHybridCQM", {
        "feasible": feas, "timing": dict(res.info.get("timing", {})), "problem_id": res.info.get("problem_id"),
    }), f"sampled: {len(X)} plans, {int(feas.sum())} feasible"

