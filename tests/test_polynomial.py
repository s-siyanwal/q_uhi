import numpy as np
import pytest

from quhi import BinaryPolynomial
from quhi.solvers import _kernels as K
from quhi.solvers.base import cp_args

from conftest import all_states, random_poly


def test_canonicalisation_merges_and_collapses():
    p = BinaryPolynomial({(1, 0): 2.0, (0, 1): 1.0, (2, 2): 4.0, (): 1.5})
    assert p.terms == {(0, 1): 3.0, (2,): 4.0}
    assert p.offset == 1.5 and p.n == 3


def test_energies_match_scalar(rng):
    p = random_poly(7, 4, 30, rng)
    X = all_states(7)
    assert np.allclose(p.energies(X), [p.energy(x) for x in X])


@pytest.mark.parametrize("sym", [True, False])
def test_qubo_matrix_roundtrip(rng, sym):
    Q = rng.normal(size=(6, 6))
    if sym:
        Q = (Q + Q.T) / 2
    p = BinaryPolynomial.from_qubo_matrix(Q, offset=0.3)
    X = all_states(6)
    direct = np.einsum("bi,ij,bj->b", X, Q, X) + 0.3
    assert np.allclose(p.energies(X), direct)
    U, off = p.to_qubo_matrix()
    assert np.allclose(np.triu(U), U)          # upper-triangular convention
    assert np.allclose(np.einsum("bi,ij,bj->b", X, U, X) + off, direct)


def test_spin_roundtrip_hubo(rng):
    p = random_poly(6, 4, 25, rng)
    g = p.to_spin()
    X = all_states(6)
    S = 2 * X - 1
    # evaluate g on spins directly
    gs = np.full(len(S), g.offset)
    for t, c in g.terms.items():
        gs += c * np.prod(S[:, list(t)], axis=1)
    assert np.allclose(gs, p.energies(X))
    back = BinaryPolynomial.from_spin(g)
    assert np.allclose(back.energies(X), p.energies(X))


def test_ising_matches_dimod(rng):
    dimod = pytest.importorskip("dimod")
    p = random_poly(6, 2, 20, rng)
    h, J, off = p.to_ising()
    bqm = p.to_dimod()
    X = all_states(6)
    S = 2 * X - 1
    ising = off + S @ h + sum(c * S[:, i] * S[:, j] for (i, j), c in J.items())
    assert np.allclose(ising, p.energies(X))
    assert np.allclose(bqm.energies((X, range(6))), p.energies(X))
    lin, quad, off2 = bqm.to_ising()
    assert np.isclose(off2, off)
    assert all(np.isclose(lin[i], h[i]) for i in range(6))


def test_arithmetic(rng):
    a, b = random_poly(5, 3, 10, rng), random_poly(5, 2, 10, rng)
    X = all_states(5)
    assert np.allclose((a + b).energies(X), a.energies(X) + b.energies(X))
    assert np.allclose((a * b).energies(X), a.energies(X) * b.energies(X))
    assert np.allclose((a - 2.0 * b).energies(X), a.energies(X) - 2 * b.energies(X))


def test_fix_substitution(rng):
    p = random_poly(6, 3, 20, rng)
    q = p.fix({0: 1, 3: 0})
    X = all_states(6)
    X = X[(X[:, 0] == 1) & (X[:, 3] == 0)]
    assert np.allclose(q.energies(X), p.energies(X))


def test_coefficient_bounds(rng):
    p = random_poly(8, 3, 30, rng)
    E = p.energies(all_states(8))
    L, U = p.coefficient_bounds()
    assert L <= E.min() + 1e-12 and E.max() <= U + 1e-12


def test_kernel_delta_and_energy(rng):
    p = random_poly(9, 4, 40, rng)
    cp = p.compile()
    args = cp_args(cp)
    for x in rng.integers(0, 2, size=(20, 9)).astype(np.int8):
        zc = K.zero_counts(x, cp.term_ptr, cp.term_vars)
        e = K.energy_of(x, cp.offset, cp.coeff, cp.term_ptr, cp.term_vars)
        assert np.isclose(e, p.energy(x))
        for i in range(9):
            d = K.delta_of(i, x, zc, cp.coeff, cp.var_ptr, cp.var_terms)
            y = x.copy(); y[i] ^= 1
            assert np.isclose(d, p.energy(y) - e)
        # flips keep zero counts consistent
        K.flip(3, x, zc, cp.var_ptr, cp.var_terms)
        assert np.array_equal(zc, K.zero_counts(x, cp.term_ptr, cp.term_vars))


def test_bruteforce_all_energies(rng):
    p = random_poly(10, 3, 40, rng)
    cp = p.compile()
    e, bits, ndeg, E = K.brute_force(*cp_args(cp), True)
    X = ((np.arange(2 ** 10)[:, None] >> np.arange(10)) & 1).astype(np.int8)
    assert np.allclose(E, p.energies(X))
    assert np.isclose(e, E.min()) and np.isclose(p.energy(bits), e)
