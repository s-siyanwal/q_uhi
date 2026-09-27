import numpy as np

from quhi import BinaryPolynomial, ConstrainedBinaryProgram
from quhi.solvers import MILPSolver
from quhi.solvers.feasible import FeasibleSA
from quhi.solvers.feasible_sqa import FeasibleSQA
from quhi.solvers.feasible_tabu import TabuOnF
from quhi.uhi import DEFAULT_MIX, UHIPlanningProblem, generate_city


def _mix(seed=0):
    city = generate_city(6, 6, seed=seed, candidate_fraction=0.6)
    prob = UHIPlanningProblem(city, interventions=DEFAULT_MIX, budget=9, min_per_district=1)
    return prob.program(2)


def test_every_slice_feasible_and_energy_is_objective():
    cbp = _mix()
    ss = FeasibleSQA(num_sweeps=60, num_reads=4, trotter_slices=6).sample_program(cbp, seed=1)
    slices = ss.info["final_slices"].reshape(-1, cbp.n)
    assert cbp.is_feasible(slices).all()                      # every slice of every read in F
    assert cbp.is_feasible(ss.samples).all()
    assert ss.samples.shape[1] == cbp.n                       # no slack / auxiliary bits
    assert np.allclose(ss.energies, cbp.objective_values(ss.samples))   # f(x), never f + ΛP


def test_pimc_on_F_matches_exact_single_qubit():
    """F = {0,1}: the restricted PIMC reduces to plain PIMC and must reproduce
    <σz> = -(c/2)/E tanh(βE), E = sqrt((c/2)^2 + Γ^2), for f(x) = c x."""
    c, gamma, beta, P = 1.0, 0.7, 2.0, 16
    cbp = ConstrainedBinaryProgram(BinaryPolynomial({(0,): c}, n=1))
    ss = FeasibleSQA(num_sweeps=400, num_reads=200, trotter_slices=P,
                     fixed=(beta, gamma)).sample_program(cbp, seed=3)
    s = 2.0 * ss.info["final_slices"][:, :, 0] - 1.0
    E = np.hypot(c / 2, gamma)
    exact = -(c / 2) / E * np.tanh(beta * E)
    assert abs(s.mean() - exact) < 0.05


def test_fsqa_and_tabu_find_mix_optimum():
    cbp = _mix(1)
    f_opt = MILPSolver().solve(cbp).best_energy
    sq = FeasibleSQA(num_sweeps=400, num_reads=8).sample_program(cbp, seed=0)
    tb = TabuOnF(num_reads=8).sample_program(cbp, seed=0)
    assert cbp.is_feasible(tb.samples).all()
    assert tb.best_energy <= f_opt + 1e-6
    assert sq.best_energy <= f_opt + 0.02


def test_time_limited_feasible_sa_is_anytime():
    cbp = _mix()
    ss = FeasibleSA(num_sweeps=200, num_reads=4, time_limit_s=0.3).sample_program(cbp, seed=0)
    assert cbp.is_feasible(ss.samples).all() and ss.info["batches"] >= 1


def test_tabu_proposal_budget_is_reached():
    cbp = _mix()
    ss = TabuOnF(num_reads=1, max_proposals=20000).sample_program(cbp, seed=0)
    assert ss.info["proposals"] >= 20000 and cbp.is_feasible(ss.samples).all()
