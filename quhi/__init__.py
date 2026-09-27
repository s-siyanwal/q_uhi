"""quhi - simulate QUBO / HUBO formulations of urban-heat-island planning problems.

Layers
------
quhi.polynomial   BinaryPolynomial (QUBO + HUBO), Ising/spin conversion, compiled form
quhi.constraints  constrained programs -> exact penalty models (slack bits, safe weights)
quhi.quadratize   HUBO -> QUBO (Rosenberg substitution with provable penalty)
quhi.uhi          synthetic cities + saturating cooling surrogate -> QUBO/HUBO programs
quhi.solvers      exhaustive, MILP (HiGHS), SA, tabu, greedy, SQA (PIMC), QAOA, exact annealing
quhi.analysis     metrics (success prob., TTS99, residual energy), benchmarks, plots
"""

from .constraints import (ConstrainedBinaryProgram, LinearConstraint, PenaltyEncoding,
                          safe_penalty_weight, slack_weights)
from .polynomial import BinaryPolynomial, CompiledPolynomial
from .quadratize import Quadratization, quadratize

__version__ = "0.1.0"

__all__ = ["BinaryPolynomial", "CompiledPolynomial", "ConstrainedBinaryProgram",
           "LinearConstraint", "PenaltyEncoding", "safe_penalty_weight", "slack_weights",
           "quadratize", "Quadratization"]
