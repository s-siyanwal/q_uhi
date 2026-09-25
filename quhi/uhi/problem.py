"""UHI mitigation planning as a constrained pseudo-Boolean program (QUBO / HUBO).

Physical surrogate
------------------
Variables x_v in {0,1}, v = (cell i, intervention o) for every allowed pair.
An intervention o at cell i cools target cell k by  A_o exp(-d_ik / L_o)
(exponential distance decay, the standard empirical form for the park cooling
effect; mean park cooling distances of ~100-350 m are reported in the
literature, see docs/BACKGROUND.md).  Cooling from several sources does not add
linearly - it saturates.  We use the multiplicative (probabilistic-OR) law

    dT_k(x) = dT_max * [ 1 - prod_v (1 - p_kv x_v) ],   p_kv = min(A_o e^{-d/L_o}, dT_max)/dT_max

which is additive for isolated sources, bounded by dT_max, and exhibits
diminishing returns when kernels overlap.  The planning objective is the
heat-exposure-weighted mean temperature

    f(x) = sum_k w_k (T0_k - dT_k(x))  [- cluster bonus]          (°C)

with w_k mixing population share and area share.

QUBO / HUBO hierarchy
---------------------
Inclusion-exclusion gives  1 - prod(1 - p x) = sum_{S != ∅} (-1)^{|S|+1} prod_{v∈S} p_v x_v,
so truncating at |S| <= K yields a degree-K polynomial:

* K = 1 : additive model (linear; ignores saturation)
* K = 2 : QUBO - pairwise diminishing returns  beta_uv = dT_max sum_k w_k p_ku p_kv > 0
* K = 3 : HUBO - third-order correction

By the Bonferroni inequalities the truncated cooling alternately over- (odd K)
and under-estimates (even K) the true cooling, so the K=2 QUBO objective is a
*conservative upper bound* on the true exposure temperature.  The optional
``cluster_bonus`` adds degree-4 terms rewarding complete 2x2 park blocks
(park-size threshold effect: larger contiguous parks form a stronger cool
island than the sum of small ones).

Constraints (integer data -> exact penalties, see quhi.constraints)
    budget      sum_v cost_o x_v <= B
    one option  at most one intervention per cell
    equity      each district receives >= m interventions (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..constraints import ConstrainedBinaryProgram, LinearConstraint
from ..polynomial import BinaryPolynomial
from .city import CANDIDATE, City


@dataclass(frozen=True)
class Intervention:
    name: str
    amplitude: float      # °C cooling at the source cell
    decay_length: float   # m, e-folding distance of the cooling effect
    cost: int             # integer cost units
    allowed: tuple = (CANDIDATE,)


# Illustrative, literature-informed defaults (see docs/BACKGROUND.md, sec. 1):
PARK = Intervention("park", amplitude=1.5, decay_length=100.0, cost=3)
WATER = Intervention("water", amplitude=2.0, decay_length=80.0, cost=5)
COOL_PAVEMENT = Intervention("cool_pavement", amplitude=0.6, decay_length=30.0, cost=1)
DEFAULT_MIX = (PARK, WATER, COOL_PAVEMENT)


@dataclass
class UHIPlanningProblem:
    city: City
    interventions: Sequence[Intervention] = (PARK,)
    budget: Optional[int] = None            # integer cost units; None = 30% of max spend
    delta_t_max: float = 3.0                # °C saturation cap per target cell
    exposure_weight: float = 0.7            # 1 = population-weighted, 0 = area-weighted
    min_per_district: int = 0               # equity constraint (0 = off)
    cluster_bonus: float = 0.0              # °C, degree-4 park-block synergy (0 = off)
    prune_tol: float = 1e-6                 # relative tolerance for dropping tiny terms
    variables: List[tuple] = field(init=False)

    def __post_init__(self) -> None:
        lu = self.city.land_use.ravel()
        self.variables = [(int(i), o) for i in range(lu.size)
                          for o, iv in enumerate(self.interventions) if lu[i] in iv.allowed]
        if not self.variables:
            raise ValueError("no candidate (cell, intervention) pairs")
        cen = self.city.centers()
        src = cen[[i for i, _ in self.variables]]
        d = np.linalg.norm(cen[:, None, :] - src[None, :, :], axis=2)       # (targets, vars)
        A = np.array([self.interventions[o].amplitude for _, o in self.variables])
        Lc = np.array([self.interventions[o].decay_length for _, o in self.variables])
        self.P = np.minimum(A[None, :] * np.exp(-d / Lc[None, :]), self.delta_t_max) / self.delta_t_max
        pop = self.city.population.ravel().astype(float)
        area = np.full(pop.size, 1.0 / pop.size)
        popw = pop / pop.sum() if pop.sum() > 0 else area
        self.w = self.exposure_weight * popw + (1 - self.exposure_weight) * area
        self.costs = np.array([self.interventions[o].cost for _, o in self.variables], dtype=int)
        if self.budget is None:
            per_cell_max = {}
            for (i, _), c in zip(self.variables, self.costs):
                per_cell_max[i] = max(per_cell_max.get(i, 0), c)
            self.budget = int(round(0.3 * sum(per_cell_max.values())))
        self._cell_groups: Dict[int, List[int]] = {}
        for v, (i, _) in enumerate(self.variables):
            self._cell_groups.setdefault(i, []).append(v)
        self._blocks = self._park_blocks() if self.cluster_bonus > 0 else []

    # ------------------------------------------------------------- helpers
    @property
    def n(self) -> int:
        return len(self.variables)

    @property
    def names(self) -> List[str]:
        R, C = self.city.shape
        return [f"{self.interventions[o].name}[{i // C},{i % C}]" for i, o in self.variables]

    def _park_blocks(self):
        """(vars of a 2x2 park block, bonus coefficient) for every all-candidate block."""
        R, C = self.city.shape
        park = [o for o, iv in enumerate(self.interventions) if iv.name == "park"]
        if not park:
            return []
        idx = {(i, o): v for v, (i, o) in enumerate(self.variables)}
        cen = self.city.centers()
        L = self.interventions[park[0]].decay_length
        out = []
        for r in range(R - 1):
            for c in range(C - 1):
                cells = [r * C + c, r * C + c + 1, (r + 1) * C + c, (r + 1) * C + c + 1]
                vs = [idx.get((i, park[0])) for i in cells]
                if None in vs:
                    continue
                ctr = cen[cells].mean(axis=0)
                reach = np.exp(-np.linalg.norm(cen - ctr, axis=1) / L)
                out.append((tuple(vs), self.cluster_bonus * float(self.w @ reach)))
        return out

    # ------------------------------------------------------------ objective
    def objective(self, order: int = 2) -> BinaryPolynomial:
        """Degree-``order`` truncation of the saturating cooling objective (°C)."""
        if order not in (1, 2, 3):
            raise ValueError("order must be 1, 2 or 3")
        P, w, D = self.P, self.w, self.delta_t_max
        n = self.n
        same_cell = np.array([i for i, _ in self.variables])
        terms: Dict[tuple, float] = {}
        lin = -D * (w @ P)
        for v in range(n):
            terms[(v,)] = lin[v]
        tol = self.prune_tol * np.abs(lin).max()
        self.pruned_mass = 0.0
        if order >= 2:
            WP = P * w[:, None]
            B = D * (WP.T @ P)
            for u in range(n):
                for v in range(u + 1, n):
                    if same_cell[u] == same_cell[v]:
                        continue          # mutually exclusive (at-most-one), term vanishes on feasible x
                    if abs(B[u, v]) > tol:
                        terms[(u, v)] = B[u, v]
                    else:
                        self.pruned_mass += abs(B[u, v])
        if order >= 3:
            for u in range(n):
                wu = w * P[:, u]
                for v in range(u + 1, n):
                    if same_cell[u] == same_cell[v]:
                        continue
                    wuv = wu * P[:, v]
                    if D * wuv.sum() <= tol:     # all triples containing (u,v) are smaller
                        continue
                    vals = -D * (wuv @ P[:, v + 1:])
                    for k, val in enumerate(vals):
                        l = v + 1 + k
                        if same_cell[l] in (same_cell[u], same_cell[v]):
                            continue
                        if abs(val) > tol:
                            terms[(u, v, l)] = val
                        else:
                            self.pruned_mass += abs(val)
        for vs, bonus in self._blocks:
            terms[vs] = terms.get(vs, 0.0) - bonus
        offset = float(w @ self.city.base_temp.ravel())
        return BinaryPolynomial(terms, offset, n, self.names)

    def fitted_quadratic(self, max_k: Optional[int] = None, n_samples: int = 4000,
                         ridge: float = 1e-6, seed: int = 0) -> BinaryPolynomial:
        """QUBO surrogate fitted by least squares to the exact objective on sampled plans.

        Unlike the inclusion-exclusion truncation (order=2), which is exact only near
        x = 0 and degrades when cooling kernels overlap strongly (docs/RESULTS.md,
        E2), this fits linear + pairwise coefficients on plans with 0..max_k
        interventions - the region the budget actually allows.
        """
        rng = np.random.default_rng(seed)
        n = self.n
        max_k = max_k or max(1, int(self.budget // max(1, self.costs.min())))
        X = np.zeros((n_samples, n), dtype=np.int8)
        for r in range(n_samples):
            k = rng.integers(0, min(max_k, n) + 1)
            X[r, rng.choice(n, size=k, replace=False)] = 1
        y = self.true_objective(X)
        iu, ju = np.triu_indices(n, 1)
        F = np.hstack([np.ones((n_samples, 1)), X, X[:, iu] * X[:, ju]]).astype(float)
        A = F.T @ F + ridge * np.eye(F.shape[1])
        c = np.linalg.solve(A, F.T @ y)
        terms = {(v,): c[1 + v] for v in range(n)}
        for k, (u, v) in enumerate(zip(iu, ju)):
            if abs(c[1 + n + k]) > 1e-12:
                terms[(int(u), int(v))] = c[1 + n + k]
        return BinaryPolynomial(terms, float(c[0]), n, self.names)

    def program(self, order: int = 2) -> ConstrainedBinaryProgram:
        cons = [LinearConstraint({v: int(c) for v, c in enumerate(self.costs)}, "<=",
                                 int(self.budget), "budget")]
        if self.min_per_district > 0:
            dist = self.city.district.ravel()
            for d in np.unique(dist):
                vs = [v for v, (i, _) in enumerate(self.variables) if dist[i] == d]
                if vs:
                    m = min(self.min_per_district, len({self.variables[v][0] for v in vs}))
                    cons.append(LinearConstraint({v: 1 for v in vs}, ">=", m, f"equity_d{d}"))
        groups = [g for g in self._cell_groups.values() if len(g) > 1]
        return ConstrainedBinaryProgram(self.objective(order), cons, groups, self.names)

    # ------------------------------------------------------ exact physics
    def cooling(self, X: np.ndarray) -> np.ndarray:
        """Exact saturating cooling dT_k(x) for a batch X (m, n) -> (m, targets)."""
        X = np.atleast_2d(X)[:, : self.n].astype(float)
        logq = np.log1p(-np.minimum(self.P, 1 - 1e-12))            # (targets, vars)
        return self.delta_t_max * (1.0 - np.exp(X @ logq.T))

    def true_objective(self, X: np.ndarray) -> np.ndarray:
        """Untruncated exposure-weighted temperature (°C) incl. cluster bonus."""
        X = np.atleast_2d(X)[:, : self.n]
        f = float(self.w @ self.city.base_temp.ravel()) - self.cooling(X) @ self.w
        for vs, bonus in self._blocks:
            f = f - bonus * np.all(X[:, list(vs)] == 1, axis=1)
        return f

    def temperature_field(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        T = self.city.base_temp.ravel().copy()
        if x is not None:
            T = T - self.cooling(x)[0]
        return T.reshape(self.city.shape)

    def decode(self, x: np.ndarray) -> np.ndarray:
        """Grid of chosen intervention index per cell (-1 = none)."""
        g = np.full(self.city.n_cells, -1, dtype=int)
        for v, (i, o) in enumerate(self.variables):
            if x[v]:
                g[i] = o
        return g.reshape(self.city.shape)

    def greedy_plan(self) -> np.ndarray:
        """Planner's baseline: repeatedly add the (cell, option) with the best exact
        marginal exposure reduction per unit cost that keeps the plan feasible
        (budget, one option per cell).  Equity constraints are not targeted."""
        x = np.zeros(self.n, dtype=np.int8)
        spent = 0
        used = set()
        f = self.true_objective(x)[0]
        while True:
            best, best_ratio, best_f = None, 0.0, f
            for v, (i, _) in enumerate(self.variables):
                if x[v] or i in used or spent + self.costs[v] > self.budget:
                    continue
                x[v] = 1
                fv = self.true_objective(x)[0]
                x[v] = 0
                ratio = (f - fv) / self.costs[v]
                if ratio > best_ratio:
                    best, best_ratio, best_f = v, ratio, fv
            if best is None:
                return x
            x[best] = 1
            spent += self.costs[best]
            used.add(self.variables[best][0])
            f = best_f

    def report(self, x: np.ndarray) -> dict:
        x = np.asarray(x)[: self.n]
        T0 = self.temperature_field(None)
        T1 = self.temperature_field(x)
        pop = self.city.population
        return {
            "n_interventions": int(x.sum()),
            "spend": int(self.costs @ x),
            "budget": int(self.budget),
            "exposure_temp_C": float(self.true_objective(x)[0]),
            "exposure_temp_baseline_C": float(self.true_objective(np.zeros(self.n))[0]),
            "mean_cooling_C": float((T0 - T1).mean()),
            "max_cooling_C": float((T0 - T1).max()),
            "pop_weighted_cooling_C": float(((T0 - T1) * pop).sum() / max(pop.sum(), 1)),
            "hot_cells_baseline": int((T0 > np.percentile(T0, 90)).sum()),
            "hot_cells_after": int((T1 > np.percentile(T0, 90)).sum()),
        }
