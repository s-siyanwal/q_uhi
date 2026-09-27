"""Synthetic urban grids for UHI planning experiments.

The generator follows the setting of the source material: a rectangular grid
of 50 m x 50 m cells (Zhou et al. 2023, Sanya) with fixed land uses and a subset
of *candidate* cells where interventions may be placed; plus the tropical
island features of the UHIPs notebook (dense core, coastal cooling).  All
quantities are synthetic but physically dimensioned so results read in °C.

Land-use codes
--------------
0 BUILDING  1 ROAD  2 CANDIDATE (vacant lot / parking / bare)  3 WATER  4 PARK
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

BUILDING, ROAD, CANDIDATE, WATER, PARK = 0, 1, 2, 3, 4
LAND_USE_NAMES = {BUILDING: "building", ROAD: "road", CANDIDATE: "candidate",
                  WATER: "water", PARK: "park"}
# impervious / heat-storing fraction by land use (illustrative)
IMPERVIOUS = {BUILDING: 0.95, ROAD: 1.0, CANDIDATE: 0.6, WATER: 0.0, PARK: 0.15}


@dataclass
class City:
    land_use: np.ndarray          # (R, C) int
    base_temp: np.ndarray         # (R, C) °C, baseline near-surface air temperature
    population: np.ndarray        # (R, C) residents per cell
    district: np.ndarray          # (R, C) int district id
    cell_size: float = 50.0       # m
    name: str = "city"

    @property
    def shape(self):
        return self.land_use.shape

    @property
    def n_cells(self) -> int:
        return self.land_use.size

    def centers(self) -> np.ndarray:
        R, C = self.shape
        rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
        return np.stack([rr.ravel(), cc.ravel()], axis=1) * self.cell_size + self.cell_size / 2

    def candidate_cells(self) -> np.ndarray:
        return np.flatnonzero(self.land_use.ravel() == CANDIDATE)

    def summary(self) -> dict:
        lu = self.land_use.ravel()
        return {"name": self.name, "shape": self.shape, "cell_size_m": self.cell_size,
                **{f"n_{v}": int(np.sum(lu == k)) for k, v in LAND_USE_NAMES.items()},
                "population": float(self.population.sum()),
                "base_temp_mean": float(self.base_temp.mean()),
                "base_temp_max": float(self.base_temp.max()),
                "n_districts": int(self.district.max() + 1)}


def generate_city(rows: int = 12, cols: int = 12, seed: int = 0, cell_size: float = 50.0,
                  candidate_fraction: float = 0.3, road_spacing: int = 4, coast: bool = True,
                  n_existing_parks: int = 1, t_rural: float = 29.0, uhi_max: float = 5.0,
                  coastal_cooling: float = 2.0, n_districts: int = 4,
                  name: Optional[str] = None) -> City:
    """Generate a reproducible synthetic city.

    Baseline temperature model (smoothed land-cover mixing, cf. the UHIPs
    notebook): T0 = t_rural + uhi_max * G_s[impervious] - coastal_cooling *
    exp(-d_coast / 150 m) + small noise, where G_s is a Gaussian smoother of
    width ~ 1.5 cells representing horizontal heat advection/mixing.
    """
    rng = np.random.default_rng(seed)
    R, C = rows, cols
    rr, cc = np.meshgrid(np.linspace(0, 1, R), np.linspace(0, 1, C), indexing="ij")
    # dense core + smooth random texture
    core = np.exp(-((rr - 0.45) ** 2 + (cc - 0.55) ** 2) / 0.18)
    tex = gaussian_filter(rng.normal(size=(R, C)), sigma=1.5)
    density = core + 0.35 * tex / (np.abs(tex).max() + 1e-12)
    density = (density - density.min()) / (np.ptp(density) + 1e-12)

    lu = np.full((R, C), BUILDING, dtype=int)
    if road_spacing:
        lu[::road_spacing, :] = ROAD
        lu[:, ::road_spacing] = ROAD
    if coast:
        lu[:, -1] = WATER                                   # sea along the east edge
    # candidates: prefer lower-density blocks (vacant lots, parking, bare land)
    free = np.flatnonzero(lu.ravel() == BUILDING)
    k = int(round(candidate_fraction * free.size))
    pref = (1.2 - density.ravel()[free])
    pref = pref / pref.sum()
    cand = rng.choice(free, size=k, replace=False, p=pref)
    lu.ravel()[cand] = CANDIDATE
    # existing parks from remaining building cells
    rest = np.flatnonzero(lu.ravel() == BUILDING)
    if n_existing_parks and rest.size:
        lu.ravel()[rng.choice(rest, size=min(n_existing_parks, rest.size), replace=False)] = PARK

    imperv = np.vectorize(IMPERVIOUS.get)(lu).astype(float)
    imperv = 0.6 * imperv + 0.4 * density           # built intensity also matters
    T0 = t_rural + uhi_max * gaussian_filter(imperv, sigma=1.5, mode="nearest")
    if coast:
        d_coast = (C - 1 - np.arange(C))[None, :] * cell_size
        T0 = T0 - coastal_cooling * np.exp(-d_coast / 150.0)
    T0 = T0 + 0.1 * rng.normal(size=(R, C))

    pop = np.where(lu == BUILDING, 50 + 450 * density, 0.0) * rng.uniform(0.7, 1.3, size=(R, C))
    pop = np.round(pop)

    # districts: vertical strips of roughly equal width
    nd = max(1, n_districts)
    district = np.minimum((np.arange(C) * nd) // C, nd - 1)[None, :].repeat(R, axis=0)
    return City(lu, T0, pop, district, cell_size, name or f"synthetic_{R}x{C}_s{seed}")
