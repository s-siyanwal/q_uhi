import numpy as np
import pytest

from quhi import quadratize
from quhi.postprocess import postprocess
from quhi.schedules import suggest_beta_range
from quhi.solvers import ExhaustiveSolver, MILPSolver, SimulatedAnnealing
from quhi.uhi import CANDIDATE, DEFAULT_MIX, UHIPlanningProblem, generate_city


@pytest.fixture(scope="module")
def city():
    return generate_city(8, 8, seed=3, candidate_fraction=0.4)


def test_city_reproducible_and_sane(city):
    c2 = generate_city(8, 8, seed=3, candidate_fraction=0.4)
    assert np.array_equal(city.land_use, c2.land_use)
    assert np.allclose(city.base_temp, c2.base_temp)
    assert (city.land_use == CANDIDATE).sum() > 5
    assert city.population[city.land_use != 0].sum() == 0     # people live in buildings only


def test_single_source_is_exact(city):
    prob = UHIPlanningProblem(city, prune_tol=0)
    f1 = prob.objective(1)
    for v in range(prob.n):
        x = np.zeros(prob.n, dtype=np.int8); x[v] = 1
        assert np.isclose(f1.energy(x), prob.true_objective(x)[0])


def test_bonferroni_bounds(city):
    """Truncated models bracket the exact saturating physics: f1 <= f_true <= f2, f3 <= f_true."""
    prob = UHIPlanningProblem(city, prune_tol=0)
    f1, f2, f3 = (prob.objective(k) for k in (1, 2, 3))
    X = np.random.default_rng(0).integers(0, 2, size=(200, prob.n)).astype(np.int8)
    ft = prob.true_objective(X)
    assert np.all(f1.energies(X) <= ft + 1e-9)
    assert np.all(f2.energies(X) >= ft - 1e-9)
    assert np.all(f3.energies(X) <= ft + 1e-9)
    # higher order is closer
    e2 = np.abs(f2.energies(X) - ft).mean()
    e3 = np.abs(f3.energies(X) - ft).mean()
    assert e3 < e2 < np.abs(f1.energies(X) - ft).mean()


def test_cooling_bounded_by_cap(city):
    prob = UHIPlanningProblem(city)
    dT = prob.cooling(np.ones(prob.n))
    assert dT.max() <= prob.delta_t_max + 1e-12 and dT.min() >= 0


def test_encoded_optimum_equals_milp_and_is_feasible():
    city = generate_city(6, 6, seed=5, candidate_fraction=0.5)
    prob = UHIPlanningProblem(city, budget=6)
    cbp = prob.program(2)
    enc = cbp.to_penalty_model()
    assert enc.model.n <= 24
    ex = ExhaustiveSolver().sample(enc.model)
    x = ex.best[: cbp.n]
    ms = MILPSolver().solve(cbp)
    assert cbp.is_feasible(x)[0]
    assert np.isclose(cbp.objective_values(x)[0], ms.best_energy)
    assert prob.costs @ x <= prob.budget


def test_intervention_mix_and_equity():
    city = generate_city(8, 8, seed=2, candidate_fraction=0.35)
    prob = UHIPlanningProblem(city, interventions=DEFAULT_MIX, budget=12, min_per_district=1)
    cbp = prob.program(2)
    assert cbp.at_most_one and len(cbp.constraints) >= 2
    ms = MILPSolver().solve(cbp)
    assert cbp.is_feasible(ms.best)[0]
    g = prob.decode(ms.best)
    assert (g >= 0).sum() == ms.best.sum()                    # one option per cell
    enc = cbp.to_penalty_model()
    br = suggest_beta_range(enc.model, resolution=1e-3)
    ss = SimulatedAnnealing(num_sweeps=1000, num_reads=16, beta_range=br).sample(enc.model, seed=0)
    xs = postprocess(cbp, ss.samples)
    feas = cbp.is_feasible(xs)
    assert feas.all()
    assert cbp.objective_values(xs[feas]).min() <= ms.best_energy + 1e-6


def test_cubic_and_cluster_models_quadratize_exactly():
    city = generate_city(5, 5, seed=1, candidate_fraction=0.6, road_spacing=0, coast=False)
    prob = UHIPlanningProblem(city, budget=9, cluster_bonus=0.3, prune_tol=1e-2)
    f = prob.objective(3)
    assert f.degree == 4                                      # cluster terms are quartic
    q = quadratize(f)
    assert q.qubo.is_quadratic
    # both are solved exactly by MILP; quadratisation must preserve the minimum
    assert np.isclose(MILPSolver().solve(q.qubo).best_energy, MILPSolver().solve(f).best_energy)


def test_fitted_quadratic_beats_truncation_under_strong_overlap():
    """With long cooling reach the Taylor QUBO is biased; the least-squares QUBO is not."""
    from quhi.uhi import Intervention
    city = generate_city(8, 8, seed=3, candidate_fraction=0.4)
    iv = Intervention("park", 1.5, 300.0, 3)
    prob = UHIPlanningProblem(city, interventions=(iv,), prune_tol=0)
    fit = prob.fitted_quadratic(n_samples=3000, seed=1)
    trunc = prob.objective(2)
    rng = np.random.default_rng(9)
    kmax = int(prob.budget // iv.cost)                      # region the budget allows
    X = np.zeros((300, prob.n), dtype=np.int8)
    for r in range(300):
        X[r, rng.choice(prob.n, size=rng.integers(0, kmax + 1), replace=False)] = 1
    ft = prob.true_objective(X)
    err_fit = np.abs(fit.energies(X) - ft).mean()
    err_trunc = np.abs(trunc.energies(X) - ft).mean()
    assert fit.is_quadratic and err_fit < 0.5 * err_trunc
