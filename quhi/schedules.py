"""Problem-aware annealing schedules.

Generic defaults (dwave-neal style) set the coldest temperature from the
smallest *model* coefficient.  In penalty-encoded models every coefficient is
penalty-dominated, so the anneal ends far too hot to resolve the objective
(docs/RESULTS.md, E3).  Here the cold end is set from a physically meaningful
*objective resolution* instead - e.g. 1e-3 °C of exposure temperature.
"""

from __future__ import annotations

import numpy as np

from .polynomial import BinaryPolynomial
from .solvers.base import energy_scale


def suggest_beta_range(model: BinaryPolynomial, resolution: float,
                       p_hot: float = 0.5, p_cold: float = 0.01) -> tuple:
    """(beta_hot, beta_cold): the hottest beta accepts the largest single-flip uphill
    move with prob. ``p_hot``; the coldest accepts an uphill move of size
    ``resolution`` with prob. ``p_cold``."""
    sc = energy_scale(model.compile())
    return (np.log(1 / p_hot) / sc["max_delta"], np.log(1 / p_cold) / resolution)
