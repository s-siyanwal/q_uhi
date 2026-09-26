import numpy as np
import pytest

from quhi import BinaryPolynomial, ConstrainedBinaryProgram, LinearConstraint, slack_weights
from quhi.constraints import safe_penalty_weight
from quhi.solvers import ExhaustiveSolver, MILPSolver

from conftest import all_states, random_poly


@pytest.mark.parametrize("m", range(0, 40))
def test_slack_weights_cover_exactly(m):
    w = slack_weights(m)
    sums = {0}
    for a in w:
        sums |= {s + a for s in sums}
    assert sums == set(range(m + 1))


def test_non_integer_constraint_rejected():
    with pytest.raises(ValueError):
        LinearConstraint({0: 2500.5}, "<=", 10)


def _random_program(rng, n=7):
    f = random_poly(n, 3, 25, rng)
    a = {i: int(rng.integers(1, 5)) for i in range(n)}
    cons = [LinearConstraint(a, "<=", int(sum(a.values()) // 2), "budget"),
            LinearConstraint({0: 1, 1: 1, 2: 1}, ">=", 1, "cover"),
            LinearConstraint({3: 1, 4: 1}, "==", 1, "pick")]
    return ConstrainedBinaryProgram(f, cons, at_most_one=[[5, 6]])


def _constrained_opt(cbp):
    X = all_states(cbp.n)
    feas = cbp.is_feasible(X)
    f = cbp.objective_values(X)
    return f[feas].min()


@pytest.mark.parametrize("seed", range(8))
def test_penalty_is_integer_and_exact(seed):
    rng = np.random.default_rng(seed)
    cbp = _random_program(rng)
    n_tot = cbp.n + cbp.num_slack()
    P, _, _ = cbp.penalty_polynomial(n_tot)
    Z = all_states(n_tot)
    pen = P.energies(Z)
    assert np.allclose(pen, np.round(pen)) and pen.min() >= -1e-9
    feas = cbp.is_feasible(Z[:, : cbp.n])
    assert np.all(pen[~feas] >= 1 - 1e-9)                    # infeasible => P >= 1
    # feasible x => some slack gives zero penalty
    for x in all_states(cbp.n)[cbp.is_feasible(all_states(cbp.n))]:
        rows = np.all(Z[:, : cbp.n] == x, axis=1)
        assert np.isclose(pen[rows].min(), 0)


@pytest.mark.parametrize("seed", range(8))
def test_safe_penalty_ground_state_is_constrained_optimum(seed):
    rng = np.random.default_rng(seed)
    cbp = _random_program(rng)
    enc = cbp.to_penalty_model()
    ss = ExhaustiveSolver().sample(enc.model, keep_all=True)
    E = ss.info["all_energies"]
    n_tot = enc.model.n
    Z = ((np.arange(2 ** n_tot)[:, None] >> np.arange(n_tot)) & 1).astype(np.int8)
    gs = E <= E.min() + 1e-9
    assert np.all(cbp.is_feasible(Z[gs, : cbp.n]))            # every ground state feasible
    assert np.isclose(E.min(), _constrained_opt(cbp))        # and optimal
    # safe bound never exceeds the textbook range bound
    L, U = cbp.objective.coefficient_bounds()
    assert enc.penalty_weight <= (U - L) + 1e-2


def test_too_small_penalty_can_fail():
    # f rewards picking both, constraint allows one: with tiny Lambda the QUBO cheats
    f = BinaryPolynomial({(0,): -1.0, (1,): -1.0})
    cbp = ConstrainedBinaryProgram(f, [LinearConstraint({0: 1, 1: 1}, "<=", 1)])
    bad = cbp.to_penalty_model(penalty_weight=0.4)
    x = ExhaustiveSolver().sample(bad.model).best[:2]
    assert not cbp.is_feasible(x)[0]
    good = cbp.to_penalty_model()
    assert cbp.is_feasible(ExhaustiveSolver().sample(good.model).best[:2])[0]
    assert good.penalty_weight < 1.01


@pytest.mark.parametrize("seed", range(5))
def test_milp_matches_enumeration(seed):
    rng = np.random.default_rng(100 + seed)
    cbp = _random_program(rng, n=9)
    ms = MILPSolver().solve(cbp)
    assert ms.info["optimal"]
    assert cbp.is_feasible(ms.best)[0]
    assert np.isclose(ms.best_energy, _constrained_opt(cbp))


@pytest.mark.parametrize("form", ["unbalanced", "linear"])
def test_inexact_penalty_forms_match_their_formulas(form):
    rng = np.random.default_rng(3)
    cbp = _random_program(rng)
    lam, l1, l2 = 1.7, 0.9, 0.4
    enc = cbp.to_penalty_model(penalty_weight=lam, form=form, lambdas=(l1, l2))
    assert enc.model.n == cbp.n                                # no slack bits
    X = all_states(cbp.n)
    expect = cbp.objective_values(X).copy()
    for c in cbp.constraints:
        ax = c.lhs(X)
        if c.sense == "==":
            expect += lam * (ax - c.rhs) ** 2
            continue
        h = (c.rhs - ax) if c.sense == "<=" else (ax - c.rhs)
        expect += (-lam * h) if form == "linear" else (-l1 * h + l2 * h ** 2)
    for g in cbp.at_most_one:
        s = X[:, g].sum(1)
        expect += lam * s * (s - 1) / 2
    assert np.allclose(enc.model.energies(X), expect)
