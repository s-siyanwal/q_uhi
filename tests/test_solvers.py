import numpy as np
import pytest

from quhi import BinaryPolynomial
from quhi.solvers import (QAOA, ExhaustiveSolver, SimulatedAnnealing, SimulatedQuantumAnnealing,
                          SteepestDescent, TabuSearch, annealing_spectrum, schrodinger_anneal)
from quhi.solvers import _kernels as K
from quhi.solvers.base import cp_args
from quhi.solvers.quantum import _apply_mixer, qaoa_state

from conftest import random_poly


@pytest.fixture(scope="module")
def hubo():
    return random_poly(14, 3, 60, np.random.default_rng(7))


@pytest.fixture(scope="module")
def ground(hubo):
    return ExhaustiveSolver().sample(hubo).best_energy


@pytest.mark.parametrize("solver", [
    SimulatedAnnealing(num_sweeps=300, num_reads=20),
    TabuSearch(max_iter=500, num_reads=20),
    SimulatedQuantumAnnealing(num_sweeps=200, num_reads=10, trotter_slices=8),
])
def test_heuristics_find_ground_state_of_hubo(solver, hubo, ground):
    ss = solver.sample(hubo, seed=1)
    assert np.isclose(ss.best_energy, ground)
    assert np.allclose(hubo.energies(ss.samples), ss.energies)     # reported energies are honest
    assert ss.success_probability(ground) > 0.3


def test_seed_reproducibility(hubo):
    a = SimulatedAnnealing(num_sweeps=50, num_reads=5).sample(hubo, seed=3)
    b = SimulatedAnnealing(num_sweeps=50, num_reads=5).sample(hubo, seed=3)
    assert np.array_equal(a.samples, b.samples)


def test_traces_monotone(hubo):
    ss = SimulatedAnnealing(num_sweeps=100, num_reads=4).sample(hubo, seed=0)
    tr = ss.info["traces"]
    assert np.all(np.diff(tr, axis=1) <= 1e-12)


def test_greedy_returns_local_minima(hubo):
    ss = SteepestDescent(num_reads=10).sample(hubo, seed=0)
    cp = hubo.compile()
    for x in ss.samples:
        zc = K.zero_counts(x, cp.term_ptr, cp.term_vars)
        assert all(K.delta_of(i, x, zc, cp.coeff, cp.var_ptr, cp.var_terms) >= -1e-12
                   for i in range(cp.n))


@pytest.mark.parametrize("h,gamma,beta", [(0.5, 0.7, 2.0), (-0.8, 0.3, 3.0), (0.2, 1.5, 1.0)])
def test_pimc_matches_exact_single_qubit(h, gamma, beta):
    """<σz> of H = h σz - Γ σx at inverse temperature β: -h/E tanh(βE), E = sqrt(h²+Γ²).

    Validates the Suzuki-Trotter action used by SQA (β/P on the problem term and
    J = ½ ln coth(βΓ/P)); the legacy implementation fails this test.
    """
    cp = BinaryPolynomial({(0,): 2 * h}).compile()      # f = 2h x = h s + h
    K.seed_rng(11)
    m = K.pimc_magnetization(*cp_args(cp), beta, gamma, 32, 60000, 1000)[0]
    E = np.hypot(h, gamma)
    assert abs(m - (-h / E * np.tanh(beta * E))) < 0.02


def test_sqa_requires_two_slices():
    with pytest.raises(ValueError):
        SimulatedQuantumAnnealing(trotter_slices=1)


def test_qaoa_state_normalised_and_mixer_unitary():
    rng = np.random.default_rng(0)
    cost = rng.normal(size=2 ** 6)
    psi = qaoa_state(cost, [0.3, 0.1], [0.5, 0.2])
    assert np.isclose(np.vdot(psi, psi).real, 1.0)
    v = rng.normal(size=2 ** 6) + 1j * rng.normal(size=2 ** 6)
    w = _apply_mixer(v.copy(), np.pi / 2, 6)             # exp(-i π/2 X)^⊗n = (-i)^n X^⊗n
    assert np.allclose(w, (-1j) ** 6 * v[::-1])


def test_qaoa_beats_uniform_sampling():
    p = random_poly(8, 3, 25, np.random.default_rng(2))
    ss = QAOA(p=3, shots=500).sample(p, seed=0)
    assert ss.info["p_ground"] > 2 * ss.info["p_ground_uniform"]
    assert ss.info["approx_ratio"] > 0.6


def test_spectrum_endpoints():
    p = random_poly(5, 3, 12, np.random.default_rng(3))
    spec = annealing_spectrum(p, [0.0, 1.0], k=4)
    assert np.isclose(spec["eigenvalues"][0, 0], -5.0)      # -sum X ground energy = -n
    E = np.sort(ExhaustiveSolver().sample(p, keep_all=True).info["all_energies"])
    diag = (E - E.min()) / (E.max() - E.min())
    assert np.allclose(spec["eigenvalues"][1, :4], diag[:4], atol=1e-8)


def test_slow_schrodinger_anneal_is_adiabatic():
    """Adiabatic theorem check: T >> 1/gap^2 gives the ground state, T ~ 1 does not."""
    p = random_poly(4, 2, 8, np.random.default_rng(4))
    gap = annealing_spectrum(p, np.linspace(0, 1, 101))["min_gap"]
    assert 0.05 < gap < 0.1
    fast = schrodinger_anneal(p, total_time=1.0, steps=100)["p_ground"]
    slow = schrodinger_anneal(p, total_time=20 / gap ** 2, steps=8000)["p_ground"]
    assert slow > 0.99 and fast < 0.5
