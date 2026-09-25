"""Numba kernels shared by all heuristic solvers.

Data structure: for the current assignment x we keep ``zc[t]`` = number of
variables of term t that are 0.  Term t is "on" iff zc[t] == 0.  Then

    delta_i(x) = f(x ^ e_i) - f(x) = (1 - 2 x_i) * sum_{t ∋ i, others all 1} c_t
               = (1 - 2 x_i) * sum_{t ∋ i} c_t [zc[t] - (1 - x_i) == 0]

costs O(deg_i) for QUBO *and* HUBO, and a flip updates zc in O(deg_i).
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def zero_counts(x, term_ptr, term_vars):
    T = term_ptr.shape[0] - 1
    zc = np.zeros(T, dtype=np.int64)
    for t in range(T):
        c = 0
        for p in range(term_ptr[t], term_ptr[t + 1]):
            if x[term_vars[p]] == 0:
                c += 1
        zc[t] = c
    return zc


@njit(cache=True)
def energy_of(x, offset, coeff, term_ptr, term_vars):
    e = offset
    for t in range(coeff.shape[0]):
        on = True
        for p in range(term_ptr[t], term_ptr[t + 1]):
            if x[term_vars[p]] == 0:
                on = False
                break
        if on:
            e += coeff[t]
    return e


@njit(cache=True)
def delta_of(i, x, zc, coeff, var_ptr, var_terms):
    xi0 = 1 - x[i]
    s = 0.0
    for p in range(var_ptr[i], var_ptr[i + 1]):
        t = var_terms[p]
        if zc[t] - xi0 == 0:
            s += coeff[t]
    return s if x[i] == 0 else -s


@njit(cache=True)
def flip(i, x, zc, var_ptr, var_terms):
    if x[i] == 0:
        x[i] = 1
        for p in range(var_ptr[i], var_ptr[i + 1]):
            zc[var_terms[p]] -= 1
    else:
        x[i] = 0
        for p in range(var_ptr[i], var_ptr[i + 1]):
            zc[var_terms[p]] += 1


# --------------------------------------------------------------------- exact
@njit(cache=True)
def brute_force(n, offset, coeff, term_ptr, term_vars, var_ptr, var_terms, keep_all):
    """Gray-code enumeration of all 2^n states.

    Returns (e_min, argmin_bits, n_degenerate, energies_or_empty).  ``energies``
    is indexed little-endian: state index b = sum_i x_i 2^i.
    """
    x = np.zeros(n, dtype=np.int8)
    zc = zero_counts(x, term_ptr, term_vars)
    e = offset
    N = 1 << n
    energies = np.empty(N if keep_all else 0, dtype=np.float64)
    best = e
    best_idx = 0
    if keep_all:
        energies[0] = e
    g = 0
    for k in range(1, N):
        # bit to flip = number of trailing zeros of k
        i = 0
        kk = k
        while (kk & 1) == 0:
            kk >>= 1
            i += 1
        e += delta_of(i, x, zc, coeff, var_ptr, var_terms)
        flip(i, x, zc, var_ptr, var_terms)
        g ^= (1 << i)
        if keep_all:
            energies[g] = e
        if e < best - 1e-12:
            best = e
            best_idx = g
    # count degeneracy in a second pass (cheap relative to enumeration)
    tol = 1e-7 * max(1.0, abs(best))
    ndeg = 0
    if keep_all:
        for k in range(N):
            if energies[k] <= best + tol:
                ndeg += 1
    else:
        x[:] = 0
        zc = zero_counts(x, term_ptr, term_vars)
        e = offset
        if e <= best + tol:
            ndeg += 1
        for k in range(1, N):
            i = 0
            kk = k
            while (kk & 1) == 0:
                kk >>= 1
                i += 1
            e += delta_of(i, x, zc, coeff, var_ptr, var_terms)
            flip(i, x, zc, var_ptr, var_terms)
            if e <= best + tol:
                ndeg += 1
    bits = np.zeros(n, dtype=np.int8)
    for i in range(n):
        bits[i] = (best_idx >> i) & 1
    return best, bits, ndeg, energies


# ------------------------------------------------------- simulated annealing
@njit(cache=True)
def sa_run(n, offset, coeff, term_ptr, term_vars, var_ptr, var_terms,
           betas, x0, num_reads, trace_every):
    """Metropolis single-flip SA, sequential sweeps, one beta per sweep."""
    S = betas.shape[0]
    out = np.empty((num_reads, n), dtype=np.int8)
    out_e = np.empty(num_reads)
    ntr = (S + trace_every - 1) // trace_every
    trace = np.empty((num_reads, ntr))
    for r in range(num_reads):
        x = x0[r].copy()
        zc = zero_counts(x, term_ptr, term_vars)
        e = energy_of(x, offset, coeff, term_ptr, term_vars)
        best_e = e
        best_x = x.copy()
        for s in range(S):
            b = betas[s]
            for i in range(n):
                d = delta_of(i, x, zc, coeff, var_ptr, var_terms)
                if d <= 0.0 or np.random.random() < np.exp(-b * d):
                    flip(i, x, zc, var_ptr, var_terms)
                    e += d
            if e < best_e - 1e-12:
                best_e = e
                best_x[:] = x
            if s % trace_every == 0:
                trace[r, s // trace_every] = best_e
        out[r] = best_x
        out_e[r] = best_e
    return out, out_e, trace


# --------------------------------------------------------------- tabu search
@njit(cache=True)
def tabu_run(n, offset, coeff, term_ptr, term_vars, var_ptr, var_terms,
             x0, num_reads, max_iter, tenure, stall_limit):
    """1-flip tabu search with aspiration (Glover 1989; Palubeckis 2004 style)."""
    out = np.empty((num_reads, n), dtype=np.int8)
    out_e = np.empty(num_reads)
    trace = np.empty((num_reads, max_iter))
    for r in range(num_reads):
        x = x0[r].copy()
        zc = zero_counts(x, term_ptr, term_vars)
        e = energy_of(x, offset, coeff, term_ptr, term_vars)
        d = np.empty(n)
        for i in range(n):
            d[i] = delta_of(i, x, zc, coeff, var_ptr, var_terms)
        tabu_until = np.zeros(n, dtype=np.int64)
        best_e = e
        best_x = x.copy()
        stall = 0
        for it in range(max_iter):
            pick = -1
            pick_d = np.inf
            nties = 0
            for i in range(n):
                allowed = tabu_until[i] <= it or e + d[i] < best_e - 1e-12
                if allowed:
                    if d[i] < pick_d - 1e-12:
                        pick_d = d[i]
                        pick = i
                        nties = 1
                    elif abs(d[i] - pick_d) <= 1e-12:
                        nties += 1          # reservoir-sample among ties
                        if np.random.random() * nties < 1.0:
                            pick = i
            if pick < 0:
                pick = np.random.randint(n)
                pick_d = d[pick]
            flip(pick, x, zc, var_ptr, var_terms)
            e += pick_d
            tabu_until[pick] = it + 1 + tenure + np.random.randint(max(1, tenure // 2 + 1))
            for p in range(var_ptr[pick], var_ptr[pick + 1]):
                t = var_terms[p]
                for q in range(term_ptr[t], term_ptr[t + 1]):
                    j = term_vars[q]
                    d[j] = delta_of(j, x, zc, coeff, var_ptr, var_terms)
            d[pick] = delta_of(pick, x, zc, coeff, var_ptr, var_terms)
            if e < best_e - 1e-12:
                best_e = e
                best_x[:] = x
                stall = 0
            else:
                stall += 1
            trace[r, it] = best_e
            if stall >= stall_limit:
                for k in range(it + 1, max_iter):
                    trace[r, k] = best_e
                break
        out[r] = best_x
        out_e[r] = best_e
    return out, out_e, trace


# -------------------------------------------- simulated quantum annealing
@njit(cache=True)
def sqa_run(n, offset, coeff, term_ptr, term_vars, var_ptr, var_terms,
            betas, gammas, P, x0, num_reads, global_moves):
    """Path-integral Monte Carlo quantum annealing (Martonak, Santoro, Tosatti 2002).

    Target distribution over P coupled replicas x^1..x^P (periodic in k):

        pi(x) ∝ exp( -(beta/P) sum_k E(x^k)  +  J(Γ) sum_{k,i} s_i^k s_i^{k+1} ),
        J(Γ) = 1/2 ln coth(beta Γ / P),       s = 2x - 1,

    which is the Suzuki-Trotter approximation of Tr exp(-beta (E - Γ sum σ^x)).
    Note the beta/P on the problem term; the legacy code used beta, i.e. it
    simulated each replica at temperature T instead of P*T (docs/AUDIT.md).
    The problem Hamiltonian E may be any diagonal HUBO - no quadratisation needed.
    ``betas`` (one per sweep) may be constant (pure quantum annealing at fixed
    temperature) or increasing (joint thermal + quantum annealing).
    """
    S = gammas.shape[0]
    T = coeff.shape[0]
    out = np.empty((num_reads, n), dtype=np.int8)
    out_e = np.empty(num_reads)
    trace = np.empty((num_reads, S))
    for r in range(num_reads):
        X = np.empty((P, n), dtype=np.int8)
        ZC = np.empty((P, T), dtype=np.int64)
        E = np.empty(P)
        for k in range(P):
            X[k] = x0[r]
            ZC[k] = zero_counts(X[k], term_ptr, term_vars)
            E[k] = energy_of(X[k], offset, coeff, term_ptr, term_vars)
        best_e = E[0]
        best_x = X[0].copy()
        for s in range(S):
            g = gammas[s]
            beta = betas[s]
            bp = beta / P
            arg = beta * g / P
            if arg > 20.0:
                J = 0.0
            elif arg < 1e-12:
                J = 0.5 * np.log(1e12)          # Γ -> 0: replicas locked together
            else:
                J = 0.5 * np.log(1.0 / np.tanh(arg))
            for k in range(P):
                km = (k - 1) % P
                kp = (k + 1) % P
                xk = X[k]
                zk = ZC[k]
                for i in range(n):
                    dE = delta_of(i, xk, zk, coeff, var_ptr, var_terms)
                    si = 2 * xk[i] - 1
                    nb = (2 * X[km, i] - 1) + (2 * X[kp, i] - 1)
                    dS = bp * dE + 2.0 * J * si * nb
                    if dS <= 0.0 or np.random.random() < np.exp(-dS):
                        flip(i, xk, zk, var_ptr, var_terms)
                        E[k] += dE
            if global_moves:
                for i in range(n):
                    dtot = 0.0
                    for k in range(P):
                        dtot += delta_of(i, X[k], ZC[k], coeff, var_ptr, var_terms)
                    dS = bp * dtot
                    if dS <= 0.0 or np.random.random() < np.exp(-dS):
                        for k in range(P):
                            d = delta_of(i, X[k], ZC[k], coeff, var_ptr, var_terms)
                            flip(i, X[k], ZC[k], var_ptr, var_terms)
                            E[k] += d
            for k in range(P):
                if E[k] < best_e - 1e-12:
                    best_e = E[k]
                    best_x[:] = X[k]
            trace[r, s] = best_e
        out[r] = best_x
        out_e[r] = best_e
    return out, out_e, trace


@njit(cache=True)
def steepest_descent(n, offset, coeff, term_ptr, term_vars, var_ptr, var_terms, X):
    """Greedy 1-flip descent to a local minimum, applied to each row of X in place."""
    out_e = np.empty(X.shape[0])
    for r in range(X.shape[0]):
        x = X[r]
        zc = zero_counts(x, term_ptr, term_vars)
        e = energy_of(x, offset, coeff, term_ptr, term_vars)
        while True:
            bi = -1
            bd = -1e-12
            for i in range(n):
                d = delta_of(i, x, zc, coeff, var_ptr, var_terms)
                if d < bd:
                    bd = d
                    bi = i
            if bi < 0:
                break
            flip(bi, x, zc, var_ptr, var_terms)
            e += bd
        out_e[r] = e
    return out_e


@njit(cache=True)
def pimc_magnetization(n, offset, coeff, term_ptr, term_vars, var_ptr, var_terms,
                       beta, gamma, P, num_sweeps, burn_in):
    """Equilibrium PIMC at fixed (beta, Gamma): time-average of <s_i> over slices.

    Used to validate the Suzuki-Trotter action against exact quantum thermal
    expectations (tests/test_solvers.py::test_pimc_matches_exact_single_qubit).
    """
    T = coeff.shape[0]
    X = np.zeros((P, n), dtype=np.int8)
    for k in range(P):
        for i in range(n):
            X[k, i] = np.random.randint(2)
    ZC = np.empty((P, T), dtype=np.int64)
    for k in range(P):
        ZC[k] = zero_counts(X[k], term_ptr, term_vars)
    bp = beta / P
    J = 0.5 * np.log(1.0 / np.tanh(beta * gamma / P))
    acc = np.zeros(n)
    cnt = 0
    for s in range(num_sweeps):
        for k in range(P):
            km = (k - 1) % P
            kp = (k + 1) % P
            for i in range(n):
                dE = delta_of(i, X[k], ZC[k], coeff, var_ptr, var_terms)
                si = 2 * X[k, i] - 1
                nb = (2 * X[km, i] - 1) + (2 * X[kp, i] - 1)
                dS = bp * dE + 2.0 * J * si * nb
                if dS <= 0.0 or np.random.random() < np.exp(-dS):
                    flip(i, X[k], ZC[k], var_ptr, var_terms)
        # global moves keep the chain ergodic at large J
        for i in range(n):
            dtot = 0.0
            for k in range(P):
                dtot += delta_of(i, X[k], ZC[k], coeff, var_ptr, var_terms)
            if bp * dtot <= 0.0 or np.random.random() < np.exp(-bp * dtot):
                for k in range(P):
                    flip(i, X[k], ZC[k], var_ptr, var_terms)
        if s >= burn_in:
            for k in range(P):
                for i in range(n):
                    acc[i] += 2 * X[k, i] - 1
            cnt += P
    return acc / cnt
