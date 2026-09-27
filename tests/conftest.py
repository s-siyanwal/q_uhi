import itertools

import numpy as np
import pytest

from quhi import BinaryPolynomial


def all_states(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int8)


def random_poly(n, degree, n_terms, rng, integer=False):
    terms = {}
    for _ in range(n_terms):
        k = int(rng.integers(1, degree + 1))
        t = tuple(sorted(rng.choice(n, size=k, replace=False)))
        c = float(rng.integers(-5, 6)) if integer else float(rng.normal())
        terms[t] = terms.get(t, 0.0) + c
    return BinaryPolynomial(terms, float(rng.normal()), n)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
