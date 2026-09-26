"""Quantum and quantum-inspired solvers.

* :class:`SimulatedQuantumAnnealing` - path-integral Monte Carlo of the
  transverse-field model (works directly on HUBOs).
* :class:`QAOA` - exact state-vector simulation of the Quantum Approximate
  Optimisation Algorithm (Farhi, Goldstone, Gutmann 2014) with a diagonal cost,
  so HUBO costs are native.  Pure NumPy, no Qiskit needed; n <= ~20.
* :func:`annealing_spectrum` / :func:`schrodinger_anneal` - exact
  diagonalisation / time evolution of H(s) = -(1-s) sum X_i + s H_P for tiny
  instances (n <= ~14): minimum gaps and adiabatic success probabilities.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..polynomial import BinaryPolynomial, CompiledPolynomial
from . import _kernels as K
from .base import SampleSet, Solver, cp_args, energy_scale, initial_states


# ---------------------------------------------------------------- SQA (PIMC)
class SimulatedQuantumAnnealing(Solver):
    """Path-integral Monte Carlo quantum annealing (Martonak, Santoro, Tosatti 2002).

    P Trotter replicas of the problem are coupled along imaginary time with
    J(Γ) = ½ ln coth(β Γ / P); the transverse field Γ is annealed to ~0.

    Two temperature modes:

    * ``mode="fixed"`` (textbook SQA): constant temperature T with P*T = ``pt``
      x E_s, where E_s is the median single-flip energy scale of the model.
    * ``mode="joint"`` (default): the per-replica inverse temperature β/P follows
      a geometric schedule ``beta_range`` (same meaning and default as
      :class:`SimulatedAnnealing`) while Γ decreases.  Penalty-encoded models
      have energy scales spanning 3-5 decades (penalties vs. objective), which a
      single fixed temperature cannot serve; see docs/RESULTS.md (E3).

    ``gamma_range`` is in units of the model's max single-flip energy scale unless
    ``absolute=True``.  Each sweep is one Metropolis pass over all P*n replica
    spins plus (``global_moves``) one pass flipping spin i in every replica.
    """

    name = "SQA"

    def __init__(self, num_sweeps: int = 1000, num_reads: int = 32, trotter_slices: int = 16,
                 mode: str = "joint", beta_range: Optional[tuple] = None, pt: float = 1.0,
                 gamma_range: tuple = (1.0, 1e-6), global_moves: bool = True,
                 absolute: bool = False, schedule: str = "linear"):
        if trotter_slices < 2:
            raise ValueError("trotter_slices must be >= 2 (P=1 has no quantum coupling)")
        if mode not in ("joint", "fixed"):
            raise ValueError("mode must be 'joint' or 'fixed'")
        self.num_sweeps = num_sweeps
        self.num_reads = num_reads
        self.P = trotter_slices
        self.mode = mode
        self.beta_range = beta_range
        self.pt = pt
        self.gamma_range = gamma_range
        self.global_moves = global_moves
        self.absolute = absolute
        self.schedule = schedule

    def _sample(self, cp, rng, x0: Optional[np.ndarray] = None, **_) -> SampleSet:
        sc = energy_scale(cp)
        P = self.P
        if self.mode == "fixed":
            T = self.pt * (1.0 if self.absolute else sc["median_delta"]) / P
            betas = np.full(self.num_sweeps, 1.0 / T)
        else:
            b0, b1 = self.beta_range or (np.log(2.0) / sc["max_delta"], np.log(100.0) / sc["min_coeff"])
            betas = P * np.geomspace(b0, b1, self.num_sweeps)       # β/P follows the SA schedule
        gscale = 1.0 if self.absolute else sc["max_delta"]
        g0, g1 = self.gamma_range[0] * gscale, self.gamma_range[1] * gscale
        if self.schedule == "linear":
            gammas = np.linspace(g0, g1, self.num_sweeps)
        elif self.schedule == "geometric":
            gammas = np.geomspace(g0, g1, self.num_sweeps)
        else:
            raise ValueError(self.schedule)
        K.seed_rng(int(rng.integers(2 ** 31)))
        X0 = initial_states(cp, rng, self.num_reads, x0)
        X, E, tr = K.sqa_run(*cp_args(cp), betas, gammas, P, X0, self.num_reads,
                             self.global_moves)
        return SampleSet(X, E, self.name, {
            "traces": tr, "P": P, "mode": self.mode,
            "beta_per_replica": (float(betas[0] / P), float(betas[-1] / P)),
            "gamma_range": (g0, g1),
            "flips_attempted": self.num_reads * self.num_sweeps * cp.n * (P + int(self.global_moves)),
        })


# ------------------------------------------------------------------- QAOA
def _apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(n):
        v = psi.reshape(-1, 2, 1 << q)
        a0 = v[:, 0, :].copy()
        a1 = v[:, 1, :]
        v[:, 0, :] = c * a0 + s * a1
        v[:, 1, :] = c * a1 + s * a0
    return psi


def qaoa_state(cost: np.ndarray, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    n = int(np.log2(cost.size))
    psi = np.full(cost.size, 1.0 / np.sqrt(cost.size), dtype=np.complex128)
    for g, b in zip(gammas, betas):
        psi *= np.exp(-1j * g * cost)
        psi = _apply_mixer(psi, b, n)
    return psi


class QAOA(Solver):
    """State-vector QAOA with X mixer and linear-ramp (TQA-style) initialisation.

    The cost is standardised, C' = (C - mean C)/std C over basis states, so
    angles are comparable across instances (this uses only the cost function,
    not the optimum).  Angles are then optimised with L-BFGS-B on <C'>.
    """

    name = "QAOA"

    def __init__(self, p: int = 3, shots: int = 256, max_n: int = 20, optimize: bool = True,
                 ramp: float = 0.75, maxiter: int = 300):
        self.p = p
        self.shots = shots
        self.max_n = max_n
        self.optimize = optimize
        self.ramp = ramp
        self.maxiter = maxiter

    def _sample(self, cp, rng, **_) -> SampleSet:
        from scipy.optimize import minimize

        if cp.n > self.max_n:
            raise ValueError(f"QAOA state vector limited to n <= {self.max_n} (got {cp.n})")
        cost = K.brute_force(*cp_args(cp), True)[3]
        mu, sd = cost.mean(), cost.std() or 1.0
        c = (cost - mu) / sd
        p = self.p
        k = (np.arange(p) + 0.5) / p
        x_init = np.concatenate([self.ramp * k, self.ramp * (1 - k)])

        def f(th):
            psi = qaoa_state(c, th[:p], th[p:])
            return float(np.real(np.vdot(psi, c * psi)))

        nfev = 0
        th = x_init
        if self.optimize:
            res = minimize(f, x_init, method="L-BFGS-B", options={"maxiter": self.maxiter})
            th, nfev = res.x, res.nfev
        psi = qaoa_state(c, th[:p], th[p:])
        prob = np.abs(psi) ** 2
        prob /= prob.sum()
        e_min = cost.min()
        gs = cost <= e_min + 1e-7 * max(1.0, abs(e_min))
        idx = rng.choice(cost.size, size=self.shots, p=prob)
        X = ((idx[:, None] >> np.arange(cp.n)) & 1).astype(np.int8)
        exp_c = float(prob @ cost)
        return SampleSet(X, cost[idx], self.name, {
            "angles": th, "p": p, "nfev": nfev, "p_ground": float(prob[gs].sum()),
            "expected_energy": exp_c, "p_ground_uniform": float(gs.mean()),
            "approx_ratio": float((cost.max() - exp_c) / (cost.max() - e_min)) if cost.max() > e_min else 1.0,
        })


# ------------------------------------------------ exact annealing dynamics
def _hamiltonian_parts(model: BinaryPolynomial, normalise: bool = True):
    from .exact import all_energies
    diag = all_energies(model)
    if normalise:
        span = diag.max() - diag.min()
        diag = (diag - diag.min()) / (span if span > 0 else 1.0)
    return diag


def _driver_matvec(n: int):
    def mv(v):
        v = np.asarray(v).ravel()
        out = np.zeros_like(v)
        for q in range(n):
            out -= v.reshape(-1, 2, 1 << q)[:, ::-1, :].reshape(-1)
        return out
    return mv


def annealing_spectrum(model: BinaryPolynomial, s_values: Sequence[float], k: int = 6,
                       normalise: bool = True, dense_max_n: int = 10) -> dict:
    """Lowest-k eigenvalues of H(s) = -(1-s) sum_i X_i + s H_P for each s.

    H_P is the diagonal problem Hamiltonian, rescaled to [0, 1] when
    ``normalise`` (so gaps are comparable across penalty weights).  The final
    ground space may be degenerate (degeneracy d); the relevant gap is
    E_d(s) - E_0(s), reported as ``gap``.  Dense diagonalisation for
    n <= ``dense_max_n``, Lanczos (eigsh) above.
    """
    from scipy.sparse.linalg import eigsh

    diag = _hamiltonian_parts(model, normalise)
    n = model.n
    N = diag.size
    d = int(np.sum(diag <= diag.min() + 1e-9))
    kk = min(max(k, d + 2), N - 2)
    drv = _driver_matvec(n)
    Xsum = None
    Xsp = None
    if n <= dense_max_n:
        Xsum = np.column_stack([drv(e) for e in np.eye(N)])
    else:                                  # compiled sparse matvec: ~10x faster than drv()
        import scipy.sparse as sparse
        idx = np.arange(N)
        rows = np.tile(idx, n)
        cols = np.concatenate([idx ^ (1 << q) for q in range(n)])
        Xsp = sparse.csr_matrix((-np.ones(rows.size), (rows, cols)), shape=(N, N))
        Dsp = sparse.diags(diag)
    evs = []
    for s in s_values:
        if s >= 1.0 - 1e-12:              # H(1) = H_P is diagonal: exact, and avoids slow
            evs.append(np.sort(diag)[:kk])  # Lanczos convergence on its near-degenerate bottom
            continue
        if Xsum is not None:
            w = np.linalg.eigvalsh((1 - s) * Xsum + s * np.diag(diag))[:kk]
        else:
            # larger Krylov space copes with the binomially degenerate driver spectrum
            w = np.sort(eigsh((1 - s) * Xsp + s * Dsp, k=kk, which="SA", return_eigenvectors=False,
                              tol=1e-10, ncv=min(N - 1, max(10 * kk, 64))))
        evs.append(np.sort(w))
    evs = np.array(evs)
    gap = evs[:, d] - evs[:, 0]
    i = int(np.argmin(gap))
    return {"s": np.asarray(s_values), "eigenvalues": evs, "degeneracy": d, "gap": gap,
            "min_gap": float(gap[i]), "s_min_gap": float(np.asarray(s_values)[i])}


def schrodinger_anneal(model: BinaryPolynomial, total_time: float, steps: int = 400,
                       normalise: bool = True) -> dict:
    """Evolve i d/dt psi = H(t/T) psi from |+>^n with 2nd-order split-operator steps.

    H(s) = -(1-s) sum X_i + s H_P; the driver factor is applied exactly per qubit
    and the diagonal factor exactly in the computational basis (error O(dt^3)/step).
    Returns the final probability of the (possibly degenerate) ground space.
    """
    diag = _hamiltonian_parts(model, normalise)
    n = model.n
    N = diag.size
    psi = np.full(N, 1 / np.sqrt(N), dtype=np.complex128)
    dt = total_time / steps
    for j in range(steps):
        s = (j + 0.5) / steps
        psi *= np.exp(-0.5j * dt * s * diag)
        psi = _apply_mixer(psi, -dt * (1 - s), n)      # exp(+i dt (1-s) sum X)
        psi *= np.exp(-0.5j * dt * s * diag)
    prob = np.abs(psi) ** 2
    gs = diag <= diag.min() + 1e-9
    return {"total_time": total_time, "p_ground": float(prob[gs].sum()), "prob": prob}
