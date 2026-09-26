import itertools
from functools import reduce

import numpy as np
import pytest

from quhi import BinaryPolynomial, ConstrainedBinaryProgram, LinearConstraint
from quhi.solvers import ConstrainedQAOA, FeasibleSA, MILPSolver, enumerate_feasible, move_graph
from quhi.uhi import DEFAULT_MIX, UHIPlanningProblem, generate_city

from conftest import all_states, random_poly

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])


def _op(n, which):
    """Kronecker product with qubit 0 as the least-significant bit (quhi's convention)."""
    mats = [which.get(q, I2) for q in reversed(range(n))]
    return reduce(np.kron, mats)


def xy_hamiltonian(n, pairs):
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i, j in pairs:
        H += 0.5 * (_op(n, {i: X, j: X}) + _op(n, {i: Y, j: Y}))
    return H


def weight(b):
    return bin(b).count("1")


@pytest.mark.parametrize("graph", ["complete", "ring"])
def test_xy_mixer_preserves_hamming_weight(graph):
    from scipy.linalg import expm
    n = 5
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)] if graph == "complete" else \
        [(i, (i + 1) % n) for i in range(n)]
    U = expm(-1j * 0.7 * xy_hamiltonian(n, pairs))
    for b in range(2 ** n):
        psi = U[:, b]
        leak = sum(abs(psi[c]) ** 2 for c in range(2 ** n) if weight(c) != weight(b))
        assert leak < 1e-10


@pytest.mark.parametrize("graph", ["complete", "ring"])
def test_projected_swap_graph_equals_xy_on_weight_sector(graph):
    n, k = 5, 2
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)] if graph == "complete" else \
        [(i, (i + 1) % n) for i in range(n)]
    H = xy_hamiltonian(n, pairs).real
    F = np.array([x for x in all_states(n) if x.sum() == k], dtype=np.int8)
    idx = F.astype(int) @ (1 << np.arange(n))
    A = move_graph(F, moves=("swap",), swap_graph=graph).toarray()
    assert np.allclose(A, H[np.ix_(idx, idx)])


def test_subspace_qaoa_equals_full_statevector_dicke_xy():
    """XY-QAOA from |D_5^2> simulated in the full 2^5 space equals the subspace simulation."""
    from scipy.linalg import expm
    n, k = 5, 2
    f = random_poly(n, 3, 12, np.random.default_rng(1))
    cbp = ConstrainedBinaryProgram(f, [LinearConstraint({i: 1 for i in range(n)}, "==", k)])
    q = ConstrainedQAOA(p=2, moves=("swap",), swap_graph="complete")
    prep = q.prepare(cbp)
    gam, bet = [0.4, 0.9], [0.3, 0.6]
    psi_sub = q.state(prep, gam, bet)
    # full space
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    H = xy_hamiltonian(n, pairs)
    Hs = H / np.abs(np.linalg.eigvalsh(H)).max()          # same unit-norm scaling
    Xall = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    c = f.energies(Xall)
    inF = Xall.sum(1) == k
    cs = np.where(inF, (c - c[inF].mean()) / c[inF].std(), 0.0)
    psi = np.where(inF, 1 / np.sqrt(inF.sum()), 0).astype(complex)
    for g, b in zip(gam, bet):
        psi = np.exp(-1j * g * cs) * psi
        psi = expm(-1j * b * Hs) @ psi
    assert np.sum(np.abs(psi[~inF]) ** 2) < 1e-12
    idx = prep["F"].astype(int) @ (1 << np.arange(n))
    assert np.allclose(np.abs(psi[idx]) ** 2, np.abs(psi_sub) ** 2, atol=1e-10)


def _small_program(seed):
    rng = np.random.default_rng(seed)
    f = random_poly(8, 3, 20, rng)
    cons = [LinearConstraint({i: int(rng.integers(1, 4)) for i in range(8)}, "<=", 6, "budget"),
            LinearConstraint({0: 1, 1: 1, 2: 1}, ">=", 1, "equity")]
    return ConstrainedBinaryProgram(f, cons, at_most_one=[[3, 4, 5]])


@pytest.mark.parametrize("seed", range(4))
def test_enumerate_feasible_matches_bruteforce(seed):
    cbp = _small_program(seed)
    Xs = all_states(8)
    brute = {tuple(x) for x in Xs[cbp.is_feasible(Xs)]}
    F = enumerate_feasible(cbp)
    assert {tuple(x) for x in F} == brute and len(F) == len(brute)


@pytest.fixture(scope="module")
def mix_program():
    city = generate_city(6, 6, seed=0, candidate_fraction=0.4)
    prob = UHIPlanningProblem(city, interventions=DEFAULT_MIX, budget=6, min_per_district=1)
    return prob.program(2)


def test_constrained_qaoa_on_mix_stays_feasible(mix_program):
    ss = ConstrainedQAOA(p=2, shots=300, restarts=1).sample_program(mix_program, seed=0)
    assert mix_program.is_feasible(ss.samples).all()          # budget ∩ one-option ∩ equity
    assert ss.info["mixer_components"] == 1
    assert ss.info["p_opt"] > ss.info["p_opt_uniform"]
    assert np.isclose(ss.info["prob"].sum(), 1.0)
    opt = MILPSolver().solve(mix_program).best_energy
    F = enumerate_feasible(mix_program)
    assert np.isclose(mix_program.objective_values(F).min(), opt)


def test_feasible_sa_is_feasible_and_honest(mix_program):
    ss = FeasibleSA(num_sweeps=300, num_reads=12).sample_program(mix_program, seed=2)
    assert mix_program.is_feasible(ss.samples).all()
    assert np.allclose(ss.energies, mix_program.objective_values(ss.samples))
    assert np.isclose(ss.best_energy, MILPSolver().solve(mix_program).best_energy)


def test_feasible_sa_every_step_feasible(mix_program):
    """Chain 50 one-sweep runs at near-infinite temperature (a near-random walk); the state
    handed back after every sweep must be feasible."""
    from quhi.solvers import _kernels as K
    from quhi.solvers.feasible import _fsa_run
    fsa = FeasibleSA(num_sweeps=1, num_reads=1)
    x = fsa.feasible_starts(mix_program, np.random.default_rng(0))
    cp = mix_program.objective.compile()
    A, lo, hi, group, ng = fsa.constraint_arrays(mix_program)
    for s in range(50):
        K.seed_rng(s)
        X, E, _, _ = _fsa_run(cp.n, cp.offset, cp.coeff, cp.term_ptr, cp.term_vars, cp.var_ptr,
                              cp.var_terms, A, lo, hi, group, ng, np.array([1e-9]), x, 1)
        assert mix_program.is_feasible(X).all()
        x = X
