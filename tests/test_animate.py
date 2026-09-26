import numpy as np

from quhi.analysis import animate as an
from quhi.uhi import UHIPlanningProblem, generate_city


def test_frames_and_gif(tmp_path):
    city = generate_city(5, 5, seed=1, candidate_fraction=0.5)
    prob = UHIPlanningProblem(city, budget=6)
    cbp = prob.program(2)
    x, steps = prob.greedy_plan(return_steps=True)
    assert np.array_equal(steps[-1], x) and steps[0].sum() == 0
    T0 = prob.temperature_field(None)
    f = an.uhi_frame(prob, cbp, x, "greedy", "final", None, T0.min(), T0.max(), 1.0)
    assert f.ndim == 3 and f.shape[2] == 3 and f.dtype == np.uint8
    tr = np.linspace(1, 0, 20)
    g = an.trace_frame(prob, cbp, tr, 5, x, "SA", "it 5", "f(x)")
    h = an.distribution_frame(np.arange(6.0), np.full(6, 1 / 6), np.eye(6, dtype=bool)[0], "t", "h")
    info = an.write_gif([f, f], tmp_path / "a.gif", fps=5)
    assert info["frames"] == 2 and (tmp_path / "a.gif").stat().st_size > 0
    assert g.shape[2] == 3 and h.shape[2] == 3
    assert len(an.frame_indices(1000, 40)) <= 40 and an.frame_indices(1000, 40)[-1] == 999
