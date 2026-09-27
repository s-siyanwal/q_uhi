"""Common result container and solver interface."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..polynomial import BinaryPolynomial, CompiledPolynomial


@dataclass
class SampleSet:
    """Samples returned by a solver, sorted by energy (lowest first)."""

    samples: np.ndarray          # (m, n) int8
    energies: np.ndarray         # (m,)
    solver: str
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        order = np.argsort(self.energies, kind="stable")
        self.samples = np.asarray(self.samples, dtype=np.int8)[order]
        self.energies = np.asarray(self.energies, dtype=float)[order]
        if "traces" in self.info and self.info["traces"] is not None:
            self.info["traces"] = np.asarray(self.info["traces"])[order]

    @property
    def best(self) -> np.ndarray:
        return self.samples[0]

    @property
    def best_energy(self) -> float:
        return float(self.energies[0])

    @property
    def num_reads(self) -> int:
        return len(self.energies)

    @property
    def wall_time(self) -> float:
        return float(self.info.get("wall_time", np.nan))

    def success_probability(self, target: float, rtol: float = 1e-7) -> float:
        tol = rtol * max(1.0, abs(target))
        return float(np.mean(self.energies <= target + tol))

    def __repr__(self) -> str:
        return (f"SampleSet<{self.solver}, reads={self.num_reads}, best={self.best_energy:.6g}, "
                f"t={self.wall_time:.3g}s>")


class Solver:
    """Base class.  Subclasses implement ``_sample(cp, rng, **kw) -> SampleSet``."""

    name = "solver"

    def sample(self, model: BinaryPolynomial, seed: Optional[int] = None, **kw) -> SampleSet:
        cp = model.compile() if isinstance(model, BinaryPolynomial) else model
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        ss = self._sample(cp, rng, **kw)
        ss.info["wall_time"] = time.perf_counter() - t0
        ss.info.setdefault("seed", seed)
        return ss

    def _sample(self, cp: CompiledPolynomial, rng: np.random.Generator, **kw) -> SampleSet:
        raise NotImplementedError


def cp_args(cp: CompiledPolynomial):
    return (cp.n, cp.offset, cp.coeff, cp.term_ptr, cp.term_vars, cp.var_ptr, cp.var_terms)


def initial_states(cp: CompiledPolynomial, rng: np.random.Generator, num_reads: int,
                   x0: Optional[np.ndarray]) -> np.ndarray:
    if x0 is None:
        return rng.integers(0, 2, size=(num_reads, cp.n)).astype(np.int8)
    x0 = np.atleast_2d(np.asarray(x0, dtype=np.int8))
    return np.repeat(x0, num_reads, axis=0) if x0.shape[0] == 1 else x0


def energy_scale(cp: CompiledPolynomial) -> Dict[str, float]:
    """Characteristic single-flip energy scales used for automatic parameters."""
    md = cp.max_abs_delta()
    md = md[md > 0]
    nz = np.abs(cp.coeff[cp.coeff != 0])
    return {
        "max_delta": float(md.max()) if md.size else 1.0,
        "median_delta": float(np.median(md)) if md.size else 1.0,
        "min_coeff": float(nz.min()) if nz.size else 1.0,
    }
