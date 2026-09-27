"""Reproduce the findings of docs/AUDIT.md against the legacy code in UHIP_Quantum-main.zip.

    python scripts/audit_legacy.py            # writes results/audit/audit.json

The legacy sources are imported unmodified from the zip (extracted to a temp dir).
"""

from __future__ import annotations

import io
import itertools
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from quhi import BinaryPolynomial  # noqa: E402
from quhi.solvers import _kernels as K  # noqa: E402
from quhi.solvers.base import cp_args  # noqa: E402


def extract_legacy() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="uhip_legacy_"))
    with zipfile.ZipFile(REPO / "UHIP_Quantum-main.zip") as z:
        z.extractall(tmp)
    root = tmp / "UHIP_Quantum-main"
    sys.path.insert(0, str(root))
    return root


def all_x(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int8)


def a1_ising_roundtrip():
    """Legacy IsingConverter.ising_to_qubo does not invert qubo_to_ising."""
    from core.qubo_formulation.ising_converter import IsingConverter
    Q = np.array([[1, -2, 0], [-2, 3, -1], [0, -1, 2]], dtype=float)
    ising = IsingConverter.qubo_to_ising(Q)
    Q2 = IsingConverter.ising_to_qubo(ising.h, ising.J, ising.offset)
    X = all_x(3)
    e1 = np.einsum("bi,ij,bj->b", X, Q, X)
    e2 = np.einsum("bi,ij,bj->b", X, Q2, X)
    S = 2 * X - 1
    ei = np.array([ising.energy(s) for s in S])
    return {"forward_energy_match": bool(np.allclose(e1, ei)),
            "roundtrip_matrix_match": bool(np.allclose(Q, Q2)),
            "roundtrip_energy_match": bool(np.allclose(e1, e2)),
            "Q": Q.tolist(), "Q_roundtrip": Q2.tolist()}


def a2_green_layout_qubo():
    """Legacy UHIGreenLayoutProblem: coefficient scale and infeasible minimiser."""
    return {f"area_budget_fraction={fr}": _a2(fr) for fr in (0.30, 0.33)}


def _a2(fraction):
    from core.problem_modeling.uhi_green_layout import UHIGreenLayoutProblem, UHIProblemConfig
    cfg = UHIProblemConfig(grid_rows=4, grid_cols=4, area_budget_fraction=fraction)
    p = UHIGreenLayoutProblem(cfg)
    p.setup_uniform_grid(candidate_fraction=0.75, random_seed=0)
    p.setup_cooling_model()
    Q, meta = p.build_qubo()
    Qphys = p.qubo_builder.Q_physical
    n = Q.shape[0]
    X = all_x(n)
    E = np.einsum("bi,ij,bj->b", X, Q, X)
    areas = np.array([p.cells[i].area for i in p.candidate_indices])
    amax = meta["area_budget"]
    feas = X @ areas <= amax + 1e-9
    phys = np.einsum("bi,ij,bj->b", X, Qphys, X)
    xq = X[np.argmin(E)]
    xc = X[feas][np.argmin(phys[feas])]
    return {"n": n, "area_budget_m2": float(amax), "cell_area_m2": float(areas[0]),
            "budget_in_cells": float(amax / areas[0]),
            "max_abs_Q": float(np.abs(Q).max()), "max_abs_Q_physical": float(np.abs(Qphys).max()),
            "scale_ratio": float(np.abs(Q).max() / np.abs(Qphys).max()),
            "qubo_minimiser_cells": int(xq.sum()),
            "qubo_minimiser_area_m2": float(xq @ areas),
            "qubo_minimiser_feasible": bool(xq @ areas <= amax + 1e-9),
            "true_constrained_opt_cells": int(xc.sum()),
            "note": "(sum a_i x_i - A_max)^2 penalises under-use and, with A_max not a multiple "
                    "of a_i, prefers the infeasible over-budget count"}


def a3_sqa_temperature():
    """Legacy SQA applies beta (not beta/P) to the problem energy of each Trotter slice.

    Legacy acceptance: exp(-beta * [dE_problem + 2 J_tau s (s_prev + s_next)]) with
    J_tau = -(1/2beta) ln tanh(beta*Gamma/P).  Our kernel uses exp(-(beta/P) dE + ...).
    Calling our kernel with (beta' = P*beta, Gamma' = Gamma/P) reproduces the legacy
    action exactly (same beta on dE, same J), so we can measure both against the exact
    single-qubit thermal expectation <σz> = -h/E tanh(beta E).
    """
    h, G, beta, P = 0.5, 0.7, 2.0, 16
    cp = BinaryPolynomial({(0,): 2 * h}).compile()
    K.seed_rng(3)
    ours = float(K.pimc_magnetization(*cp_args(cp), beta, G, P, 100000, 2000)[0])
    K.seed_rng(3)
    legacy = float(K.pimc_magnetization(*cp_args(cp), P * beta, G / P, P, 100000, 2000)[0])
    E = np.hypot(h, G)
    exact = float(-h / E * np.tanh(beta * E))
    # legacy samples the problem at an effective inverse temperature P*beta:
    exact_at_Pbeta_small_field = float(-h / np.hypot(h, G / P) * np.tanh(P * beta * np.hypot(h, G / P)))
    return {"h": h, "Gamma": G, "beta": beta, "P": P, "exact": exact, "ours": ours,
            "legacy_action": legacy, "exact_for_legacy_parameters": exact_at_Pbeta_small_field}


def a4_theorem_iv1_counterexample():
    """Theorem IV.1 with non-integer constraint data: Lambda > f_max - f_min is NOT sufficient."""
    # f = -x0 - x1 ; constraint 0.6 x0 + 0.6 x1 <= 1  (only one may be chosen)
    f = BinaryPolynomial({(0,): -1.0, (1,): -1.0})
    X = all_x(2)
    fv = f.energies(X)
    lam = (fv.max() - fv.min()) * 1.01           # the theorem's sufficient weight
    g = X @ np.array([0.6, 0.6]) - 1.0
    F = fv + lam * np.maximum(g, 0) ** 2          # even the ideal (non-polynomial) max-penalty
    xm = X[np.argmin(F)]
    return {"lambda": lam, "penalised_minimiser": xm.tolist(),
            "minimiser_feasible": bool((xm @ [0.6, 0.6]) <= 1.0),
            "violation_squared": float(max(0.0, xm @ [0.6, 0.6] - 1) ** 2)}


def a5_four_cell_example():
    """maths3 Example IV.1: the printed symmetric Q doubles couplings if read as x^T Q x."""
    f = BinaryPolynomial({(0,): -0.4, (1,): -0.3, (2,): -0.3, (3,): -0.2,
                          (0, 1): -0.1, (1, 2): -0.1, (2, 3): -0.1})
    X = all_x(4)
    card = X.sum(1) == 2
    fv = f.energies(X)
    x_star = X[card][np.argmin(fv[card])]
    Qdoc = np.array([[-6.4, 3.9, 4, 4], [3.9, -6.3, 3.9, 4], [4, 3.9, -6.3, 3.9], [4, 4, 3.9, -6.2]])
    E_sym = np.einsum("bi,ij,bj->b", X, Qdoc, X)                  # x^T Q x (symmetric reading)
    E_upper = np.einsum("bi,ij,bj->b", X, np.triu(Qdoc), X)        # upper-triangular reading
    pen = f.energies(X) + 2 * (X.sum(1) - 2) ** 2 - 8               # intended F minus constant
    return {"constrained_opt": x_star.tolist(), "constrained_opt_value": float(fv[card].min()),
            "upper_reading_matches_intended": bool(np.allclose(E_upper, pen)),
            "symmetric_reading_matches_intended": bool(np.allclose(E_sym, pen)),
            "symmetric_reading_minimiser": X[np.argmin(E_sym)].tolist(),
            "symmetric_reading_minimiser_cardinality": int(X[np.argmin(E_sym)].sum()),
            "plan_4cell_claim": _plan_claim()}


def _plan_claim():
    """UHI_QUBO_maths.docx benchmark_suite: alpha=[-3,-2,-3,-2], beta01=beta23=0.5,
    claimed optimum x=[0,1,1,0] with energy -19.25 (2 greens)."""
    f = BinaryPolynomial({(0,): -3.0, (1,): -2.0, (2,): -3.0, (3,): -2.0, (0, 1): 0.5, (2, 3): 0.5})
    X = all_x(4)
    two = X.sum(1) == 2
    fv = f.energies(X)
    best = X[two][np.argmin(fv[two])]
    return {"claimed_x": [0, 1, 1, 0], "claimed_f": float(f.energy([0, 1, 1, 0])),
            "best_two_cell_x": best.tolist(), "best_two_cell_f": float(fv[two].min())}


def a6_research_results():
    """The legacy 'research results' JSON shipped in the zip."""
    with zipfile.ZipFile(REPO / "UHIP_Quantum-main.zip") as z:
        d = json.load(io.TextIOWrapper(z.open("UHIP_Quantum-main/notebooks/research_results/research_results.json")))
    ar = d["algorithm_results"]
    return {"qubo_min_value": d["qubo_properties"]["min_value"],
            "energies": {k: v["energy"] for k, v in ar.items()},
            "green_spaces": {k: v["green_spaces"] for k, v in ar.items()},
            "sqa_all_zero": bool(sum(ar["SQA"]["solution"]) == 0)}


def a7_auto_penalty_underestimate():
    """Legacy calculate_sufficient_penalty_weight vs the true objective range (Q = all-ones)."""
    from core.qubo_formulation.constraint_penalty import ConstraintPenaltyEncoder
    n = 10
    Q = np.ones((n, n))
    w = ConstraintPenaltyEncoder.calculate_sufficient_penalty_weight(Q, n_constraints=1, safety_factor=1.0)
    return {"n": n, "legacy_weight": float(w), "true_range": float(n * n)}


def main():
    extract_legacy()
    out = {}
    for name, fn in [("A1_ising_roundtrip", a1_ising_roundtrip), ("A2_green_layout_qubo", a2_green_layout_qubo),
                     ("A3_sqa_temperature", a3_sqa_temperature),
                     ("A4_theorem_iv1_counterexample", a4_theorem_iv1_counterexample),
                     ("A5_four_cell_example", a5_four_cell_example),
                     ("A6_research_results", a6_research_results),
                     ("A7_auto_penalty_underestimate", a7_auto_penalty_underestimate)]:
        try:
            out[name] = fn()
        except Exception as e:  # keep going; record the failure
            out[name] = {"error": repr(e)}
        print(name, json.dumps(out[name], indent=1, default=float))
    d = REPO / "results" / "audit"
    d.mkdir(parents=True, exist_ok=True)
    (d / "audit.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
