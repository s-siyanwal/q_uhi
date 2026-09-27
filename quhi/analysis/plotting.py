"""Matplotlib figures: city maps, before/after plans, convergence and benchmark plots."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

LU_COLORS = ["#8c8c8c", "#3a3a3a", "#e8d9a8", "#4f8fd6", "#3f9b4f"]  # bldg road cand water park
LU_LABELS = ["building", "road", "candidate lot", "water", "existing park"]
IV_COLORS = {"park": "#1f8a3a", "water": "#1c6fd1", "cool_pavement": "#f2f2f2"}


def _grid(ax, shape):
    ax.set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.4)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)


def plot_city(city, path: Optional[str] = None):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    axs[0].imshow(city.land_use, cmap=ListedColormap(LU_COLORS), vmin=-0.5, vmax=4.5)
    axs[0].legend(handles=[Patch(color=c, label=l) for c, l in zip(LU_COLORS, LU_LABELS)],
                  loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=7, frameon=False)
    axs[0].set_title("Land use")
    im = axs[1].imshow(city.base_temp, cmap="RdYlBu_r")
    fig.colorbar(im, ax=axs[1], fraction=0.046, label="°C")
    axs[1].set_title("Baseline air temperature")
    im = axs[2].imshow(city.population, cmap="magma_r")
    fig.colorbar(im, ax=axs[2], fraction=0.046, label="residents / cell")
    axs[2].set_title("Population (exposure weight)")
    for a in axs:
        _grid(a, city.shape)
    fig.suptitle(f"{city.name}  ({city.shape[0]}x{city.shape[1]} cells of {city.cell_size:.0f} m)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def _overlay(ax, problem, x):
    g = problem.decode(x)
    for o, iv in enumerate(problem.interventions):
        rr, cc = np.nonzero(g == o)
        ax.scatter(cc, rr, marker="s", s=60, c=IV_COLORS.get(iv.name, "k"),
                   edgecolors="black", linewidths=0.6, label=iv.name)


def plot_plan_transition(problem, plans: Dict[str, np.ndarray], path: Optional[str] = None,
                         title: str = ""):
    """Row of maps: baseline temperature, then each plan's temperature with interventions."""
    k = 1 + len(plans)
    fig, axs = plt.subplots(1, k, figsize=(4.1 * k, 4.3))
    T0 = problem.temperature_field(None)
    fields = [T0] + [problem.temperature_field(x) for x in plans.values()]
    vmin = min(f.min() for f in fields)
    vmax = T0.max()
    im = axs[0].imshow(T0, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    base = problem.true_objective(np.zeros(problem.n))[0]
    axs[0].set_title(f"Baseline\nexposure T = {base:.3f} °C")
    for a, (label, x), T in zip(axs[1:], plans.items(), fields[1:]):
        a.imshow(T, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        _overlay(a, problem, x)
        r = problem.report(x)
        a.set_title(f"{label}\nexposure T = {r['exposure_temp_C']:.3f} °C, "
                    f"spend {r['spend']}/{r['budget']}", fontsize=9)
    for a in axs:
        _grid(a, problem.city.shape)
    handles = [Patch(color=IV_COLORS.get(iv.name, "k"), label=iv.name) for iv in problem.interventions]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.colorbar(im, ax=axs, fraction=0.02, label="air temperature °C")
    if title:
        fig.suptitle(title)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_cooling_map(problem, x, path: Optional[str] = None, title: str = "Cooling achieved"):
    fig, ax = plt.subplots(figsize=(5, 4.4))
    dT = problem.temperature_field(None) - problem.temperature_field(x)
    im = ax.imshow(dT, cmap="Blues")
    _overlay(ax, problem, x)
    fig.colorbar(im, ax=ax, fraction=0.046, label="ΔT (°C)")
    ax.set_title(title)
    _grid(ax, problem.city.shape)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_convergence(traces: Dict[str, np.ndarray], f_opt: Optional[float] = None,
                     path: Optional[str] = None, xlabel: str = "sweep / iteration",
                     title: str = "Best energy so far (median and IQR over reads)"):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, tr in traces.items():
        tr = np.asarray(tr, dtype=float)
        med = np.median(tr, axis=0)
        lo, hi = np.percentile(tr, [25, 75], axis=0)
        xs = np.arange(tr.shape[1])
        ax.plot(xs, med, label=label)
        ax.fill_between(xs, lo, hi, alpha=0.2)
    if f_opt is not None:
        ax.axhline(f_opt, color="k", ls="--", lw=1, label="certified optimum")
    ax.set_xscale("symlog")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("model energy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_metric_vs(df, x: str, y: str, hue: str = "solver", path: Optional[str] = None,
                   logy: bool = False, title: str = "", xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None, logx: bool = False):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, g in df.groupby(hue):
        s = g.groupby(x)[y]
        med = s.median()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        ax.plot(med.index, med.values, marker="o", label=str(label))
        ax.fill_between(med.index, q1.values, q3.values, alpha=0.15)
    vals = np.asarray(df[y], dtype=float)
    if logy and np.any(np.isfinite(vals) & (vals > 0)):
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_spectrum(spec: dict, path: Optional[str] = None, title: str = ""):
    fig, ax = plt.subplots(figsize=(6, 4))
    ev = spec["eigenvalues"]
    for j in range(ev.shape[1]):
        ax.plot(spec["s"], ev[:, j] - ev[:, 0], lw=1)
    ax.set_xlabel("annealing parameter s")
    ax.set_ylabel("E_j(s) - E_0(s)")
    ax.set_title(title or f"min gap {spec['min_gap']:.3g} at s={spec['s_min_gap']:.2f}")
    ax.set_ylim(bottom=-0.02)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
