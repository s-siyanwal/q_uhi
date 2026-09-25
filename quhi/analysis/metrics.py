"""Benchmark metrics (see docs/BACKGROUND.md sec. 5 for definitions and sources)."""

from __future__ import annotations

import numpy as np


def success_probability(energies: np.ndarray, e_opt: float, rtol: float = 1e-7) -> float:
    energies = np.asarray(energies, dtype=float)
    return float(np.mean(energies <= e_opt + rtol * max(1.0, abs(e_opt))))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, ctr - half), min(1.0, ctr + half))


def tts(p_success: float, t_run: float, target: float = 0.99) -> float:
    """Time-to-solution (Ronnow et al., Science 2014):
    TTS = t_run * ln(1 - target) / ln(1 - p), i.e. expected time to see the optimum
    at least once with probability ``target`` using independent repetitions."""
    if p_success <= 0:
        return float("inf")
    if p_success >= target:
        return float(t_run)
    return float(t_run * np.log(1 - target) / np.log(1 - p_success))


def benefit_ratio(f_x: np.ndarray, f_baseline: float, f_opt: float) -> np.ndarray:
    """Fraction of the optimal improvement achieved: (f0 - f(x)) / (f0 - f*).  1 = optimal."""
    den = f_baseline - f_opt
    return (f_baseline - np.asarray(f_x, dtype=float)) / (den if den != 0 else 1.0)
