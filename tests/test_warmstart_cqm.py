import numpy as np
import pytest

from quhi.solvers.cqm import cqm_spec, spec_is_feasible, to_cqm
from quhi.solvers.warmstart import QaoaSeededFeasibleSA
from quhi.uhi import DEFAULT_MIX, UHIPlanningProblem, generate_city


def _mix():
    city = generate_city(6, 6, seed=0, candidate_fraction=0.6)
    prob = UHIPlanningProblem(city, interventions=DEFAULT_MIX, budget=9, min_per_district=1)
    return prob, prob.program(2)


def test_qaoa_seeded_sa_starts_and_ends_feasible():
    prob, cbp = _mix()
    ss = QaoaSeededFeasibleSA(time_limit_s=3.0, ps=(2,)).sample_program(cbp, seed=0)
    assert ss.info["quantum_seed"] and 0 < ss.info["frac_qaoa"] < 1
    assert cbp.is_feasible(ss.samples).all()


def test_cqm_spec_matches_program_feasibility():
    prob, cbp = _mix()
    X = np.random.default_rng(0).integers(0, 2, (3000, cbp.n)).astype(np.int8)
    X[:500] = 0
    assert np.array_equal(spec_is_feasible(cqm_spec(cbp), X), cbp.is_feasible(X))


def test_to_cqm_if_dimod_available():
    pytest.importorskip("dimod")
    prob, cbp = _mix()
    assert len(to_cqm(cbp).constraints) == len(cqm_spec(cbp)["constraints"])
