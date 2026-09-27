import numpy as np
import pytest

from quhi import BinaryPolynomial, quadratize
from quhi.solvers import ExhaustiveSolver

from conftest import all_states, random_poly


@pytest.mark.parametrize("seed", range(10))
def test_min_over_aux_reproduces_f(seed):
    rng = np.random.default_rng(seed)
    p = random_poly(6, 4, 18, rng)
    q = quadratize(p)
    assert q.qubo.is_quadratic
    m = q.qubo.n - p.n
    assert m <= 12
    Z = all_states(q.qubo.n)
    E = q.qubo.energies(Z)
    X = all_states(p.n)
    f = p.energies(X)
    for k, x in enumerate(X):
        rows = np.all(Z[:, : p.n] == x, axis=1)
        assert np.isclose(E[rows].min(), f[k])              # exact for every x
        assert np.isclose(q.qubo.energy(q.lift(x)), f[k])   # lift gives consistent aux


def test_quadratic_input_unchanged(rng):
    p = random_poly(5, 2, 10, rng)
    q = quadratize(p)
    assert q.aux == [] and q.qubo.terms == p.terms


def test_insufficient_penalty_breaks_exactness():
    # f = -3 x0 x1 x2: with a tiny M the aux can be set to 1 "for free"
    p = BinaryPolynomial({(0, 1, 2): -3.0})
    q = quadratize(p, penalty_weight=0.1)
    E_bad = ExhaustiveSolver().sample(q.qubo).best_energy
    E_good = ExhaustiveSolver().sample(quadratize(p).qubo).best_energy
    assert np.isclose(E_good, -3.0)
    assert E_bad < -3.0
