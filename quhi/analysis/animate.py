"""Animated (GIF) views of how each solver moves in its own search space and what the
incumbent plan does to the city temperature field.

Frames are rendered with matplotlib (Agg) in the style of :mod:`quhi.analysis.plotting`
and written with Pillow.  Every frame of a GIF uses one fixed temperature scale
(``vmin``/``vmax``) and one fixed cooling scale, so colours are comparable across frames.
Frames always show a real plan x; temperatures are never interpolated between plans.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .plotting import IV_COLORS, LU_COLORS, _grid, _overlay, plt  # noqa: F401  (plt: Agg backend)
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DPI = 100


# ----------------------------------------------------------------- I/O
def fig_to_array(fig) -> np.ndarray:
    """Render a figure to an (H, W, 3) uint8 array and close it."""
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return a


def write_gif(frames: Sequence[np.ndarray], path, fps: float = 10, hold_last: int = 3) -> dict:
    """Write frames as a looping GIF (adaptive palette).  Returns frame count and bytes."""
    from PIL import Image

    imgs = [Image.fromarray(f).convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for f in frames]
    imgs += [imgs[-1]] * hold_last
    path = Path(path)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0,
                 optimize=True, disposal=1)
    return {"frames": len(frames), "bytes": path.stat().st_size, "width": frames[0].shape[1],
            "height": frames[0].shape[0], "fps": fps}


def frame_indices(n_steps: int, n_frames: int = 40) -> np.ndarray:
    """At most ``n_frames`` indices spread over range(n_steps), always keeping the last one."""
    if n_steps <= n_frames:
        return np.arange(n_steps)
    return np.unique(np.round(np.linspace(0, n_steps - 1, n_frames)).astype(int))


# -------------------------------------------------------------- headers
def header_text(problem, cbp, x, method: str, step: str, energy: Optional[float],
                energy_label: str = "model energy") -> str:
    r = problem.report(x)
    feas = bool(cbp.is_feasible(np.asarray(x)[: cbp.n])[0])
    e = "" if energy is None else f"   {energy_label} = {energy:.4f}"
    return (f"{method}  |  {step}{e}\nexposure T = {r['exposure_temp_C']:.3f} °C   "
            f"spend {r['spend']}/{r['budget']}   feasible: {'yes' if feas else 'NO'}")


def _land_use_panel(ax, problem, x):
    ax.imshow(problem.city.land_use, cmap=ListedColormap(LU_COLORS), vmin=-0.5, vmax=4.5)
    _overlay(ax, problem, x)
    _grid(ax, problem.city.shape)


def _legend(fig, problem):
    handles = [Patch(color=IV_COLORS.get(iv.name, "k"), label=iv.name) for iv in problem.interventions]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)


# ------------------------------------------------------- physical space
def uhi_frame(problem, cbp, x, method: str, step: str, energy: Optional[float], vmin: float, vmax: float,
              dtmax: float, energy_label: str = "model energy") -> np.ndarray:
    """Land use + plan | air temperature | cooling ΔT, on fixed colour scales."""
    x = np.asarray(x)[: problem.n]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.6), dpi=DPI)
    _land_use_panel(axs[0], problem, x)
    axs[0].set_title("Land use + interventions", fontsize=9)
    T0 = problem.temperature_field(None)
    T = problem.temperature_field(x)
    im = axs[1].imshow(T, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    _overlay(axs[1], problem, x)
    fig.colorbar(im, ax=axs[1], fraction=0.046, label="air temperature °C")
    axs[1].set_title("Air temperature", fontsize=9)
    im = axs[2].imshow(T0 - T, cmap="Blues", vmin=0, vmax=dtmax)
    _overlay(axs[2], problem, x)
    fig.colorbar(im, ax=axs[2], fraction=0.046, label="ΔT = T0 − T(x)  °C")
    r = problem.report(x)
    axs[2].set_title(f"Cooling (pop-weighted {r['pop_weighted_cooling_C']:.3f} °C, "
                     f"{r['n_interventions']} interventions)", fontsize=9)
    for a in axs[1:]:
        _grid(a, problem.city.shape)
    fig.suptitle(header_text(problem, cbp, x, method, step, energy, energy_label), fontsize=10)
    _legend(fig, problem)
    fig.subplots_adjust(top=0.82, bottom=0.1, wspace=0.25)
    return fig_to_array(fig)


def compare_frame(problem, cbp, cells: Dict[str, tuple], vmin: float, vmax: float, title: str = "") -> np.ndarray:
    """1 x k temperature maps; ``cells`` maps label -> (x, step text)."""
    k = len(cells)
    fig, axs = plt.subplots(1, k, figsize=(3.5 * k, 4.3), dpi=DPI)
    im = None
    for a, (label, (x, step)) in zip(np.atleast_1d(axs), cells.items()):
        x = np.asarray(x)[: problem.n]
        im = a.imshow(problem.temperature_field(x), cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        _overlay(a, problem, x)
        _grid(a, problem.city.shape)
        r = problem.report(x)
        feas = bool(cbp.is_feasible(x)[0])
        a.set_title(f"{label}\n{step}\nT = {r['exposure_temp_C']:.3f} °C, spend {r['spend']}/{r['budget']}"
                    f"{'' if feas else ', INFEASIBLE'}", fontsize=8)
    fig.colorbar(im, ax=axs, fraction=0.015, label="air temperature °C")
    if title:
        fig.suptitle(title, fontsize=10)
    _legend(fig, problem)
    fig.subplots_adjust(top=0.78, bottom=0.08, left=0.02, right=0.9, wspace=0.08)
    return fig_to_array(fig)


# ---------------------------------------------------------- search space
def trace_frame(problem, cbp, trace: np.ndarray, t: int, x, method: str, step: str, ylabel: str,
                f_opt: Optional[float] = None, note: str = "") -> np.ndarray:
    """Best-so-far energy vs sweep with a cursor at ``t`` | the incumbent decoded on the grid."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), dpi=DPI, gridspec_kw={"width_ratios": [1.4, 1]})
    xs = np.arange(len(trace))
    axs[0].plot(xs, trace, color="#999999", lw=1)
    axs[0].plot(xs[: t + 1], trace[: t + 1], color="#1f77b4", lw=1.8)
    axs[0].axvline(t, color="#d62728", lw=1)
    if f_opt is not None:
        axs[0].axhline(f_opt, color="k", ls="--", lw=1, label="certified constrained optimum")
        axs[0].legend(fontsize=8)
    axs[0].set_xscale("symlog")
    axs[0].set_xlim(left=0)
    axs[0].set_xlabel("sweep / iteration")
    axs[0].set_ylabel(ylabel)
    axs[0].grid(alpha=0.3)
    x = np.asarray(x)
    _land_use_panel(axs[1], problem, x[: problem.n])
    axs[1].set_title("incumbent plan" + (f"\n{note}" if note else ""), fontsize=9)
    fig.suptitle(header_text(problem, cbp, x, method, step, float(trace[t]), ylabel), fontsize=10)
    _legend(fig, problem)
    fig.subplots_adjust(top=0.82, bottom=0.14, wspace=0.15)
    return fig_to_array(fig)


def distribution_frame(cost: np.ndarray, prob: np.ndarray, opt: np.ndarray, title: str, header: str,
                       feasible: Optional[np.ndarray] = None, ylim: Optional[float] = None,
                       xlabel: str = "feasible plans, sorted by objective (cheapest → hottest)") -> np.ndarray:
    """|ψ|² over basis states sorted by cost; optimum marked; optional feasible/infeasible colouring."""
    order = np.argsort(cost, kind="stable")
    pr, op = prob[order], opt[order]
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=DPI)
    xs = np.arange(len(pr))
    if feasible is None:
        ax.bar(xs, pr, width=1.0, color="#1f77b4", label="feasible plan")
    else:
        fe = feasible[order]
        ax.bar(xs[fe], pr[fe], width=1.0, color="#1f77b4", label=f"feasible ({fe.sum()})")
        ax.bar(xs[~fe], pr[~fe], width=1.0, color="#d62728", alpha=0.7, label=f"infeasible ({(~fe).sum()})")
    ax.bar(xs[op], pr[op], width=max(1.0, len(pr) / 150), color="#2ca02c", label="certified optimum")
    ax.axhline(1 / len(pr), color="#8c564b", ls=":", lw=1, label="uniform")
    ax.set_xlim(-0.5, len(pr) - 0.5)
    if ylim:
        ax.set_ylim(0, ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability |ψ|²")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(header, fontsize=10)
    fig.subplots_adjust(top=0.8, bottom=0.13)
    return fig_to_array(fig)


def to_frames(render, items: List, n_frames: int = 40) -> List[np.ndarray]:
    """Render ``items[i]`` for at most ``n_frames`` evenly spaced i."""
    return [render(items[i]) for i in frame_indices(len(items), n_frames)]
