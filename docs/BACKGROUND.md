# Background, theory and design decisions

This document covers three things:

* the physics surrogate behind `quhi.uhi`;
* the corrected theory behind `quhi.constraints` and `quhi.quadratize`;
* the algorithms in `quhi.solvers`.

It also records where each modelling choice comes from. The references are listed at the end.

## 1. UHI planning as binary optimisation

### 1.1 Setting
Following the Sanya study that the legacy material builds on [Zhou et al. 2023], the city is a
grid of 50 m × 50 m cells with fixed land uses. A subset of *candidate* cells (vacant lots,
parking, bare land) can receive an intervention o ∈ {park, water body, cool pavement}.
Each decision variable is x_v ∈ {0,1} with v = (cell i, option o).

### 1.2 Cooling from one intervention
Field studies consistently find that park cooling decays with distance from the park edge and
vanishes beyond a *park cooling distance* (PCD). Most PCDs lie within 200 m (mean ≈ 161 m,
maxima > 400 m). Large parks (> 10 ha) give 1–2 °C of cooling up to about 350 m away
[Sci. Rep. 2024 size-threshold study; Heliyon 2019 review; PLOS One 2025]. Cool roofs are
more local: raising roof albedo from 0.3 to 0.8 lowers UHI intensity by up to about 0.64 °C
[PMC11496832]. `quhi` therefore models the cooling at target cell k from option o at cell i as

  e_ik = A_o · exp(−d_ik / L_o)

The default values are illustrative and inside the reported ranges: park A = 1.5 °C, L = 100 m;
water A = 2.0 °C, L = 80 m; cool pavement A = 0.6 °C, L = 30 m. They are all parameters of
`Intervention`, and experiment E2 sweeps L from 50 to 300 m.

### 1.3 Combining several interventions: saturation
Cooling effects do not add without limit, because overlapping cool islands saturate. We use the
multiplicative (probabilistic-OR) law

  ΔT_k(x) = ΔT_max · [1 − ∏_v (1 − p_kv x_v)],  p_kv = min(e_kv, ΔT_max) / ΔT_max

This law is exact for a single source, additive when sources do not overlap, bounded by
ΔT_max (3 °C by default), and has **diminishing returns**. The planning objective is the
heat-exposure-weighted mean temperature

  f(x) = Σ_k w_k (T0_k − ΔT_k(x)),  w_k = λ·pop_k/Σpop + (1 − λ)/N_cells  (λ = 0.7)

So f is measured in °C and reads as "the temperature the average (weighted) person experiences".

### 1.4 Where the QUBO and HUBO come from
Inclusion–exclusion gives 1 − ∏(1 − p_v x_v) = Σ_{∅≠S} (−1)^{|S|+1} ∏_{v∈S} p_v x_v.
Truncating at |S| ≤ K gives a degree-K pseudo-Boolean polynomial:

| K | model | term signs |
|---|---|---|
| 1 | additive (linear) | −(cooling) |
| 2 | **QUBO** | pairs +ΔT_max Σ_k w_k p_ku p_kv (diminishing returns) |
| 3 | **HUBO** | triples −ΔT_max Σ_k w_k p_ku p_kv p_kl |

**Bonferroni property (tested in `tests/test_uhi.py::test_bonferroni_bounds`).** The partial sums of
inclusion–exclusion alternately over- and under-estimate the union. Hence
f₁ ≤ f_true ≤ f₂ and f₃ ≤ f_true. The QUBO objective is therefore a *conservative* (pessimistic)
estimate of the exposure temperature, and each extra order tightens the bracket.

A second, *genuinely* higher-order effect is optional (`cluster_bonus`). Park cooling grows
non-linearly with park area; there is a size threshold below which small parks form weak cool islands
[Sci. Rep. 2024]. We reward each complete 2×2 block of parks (a 1 ha park) with a quartic term
−κ·Σ_k w_k e^{−d_k/L} x_a x_b x_c x_d.

### 1.5 Constraints
* **Budget** Σ_v cost_o x_v ≤ B (integer cost units).
* **One option per cell** Σ_o x_(i,o) ≤ 1.
* **Equity (optional).** Every district receives at least m interventions.

### 1.6 A structural fact that matters for any "quantum advantage" claim
With a single option per cell and no equity constraint, f is a constant minus a weighted
*coverage* function, so −f is **monotone submodular** (a sum over k of w_k·ΔT_max·(1 − ∏(1 − p x)),
each term a probabilistic coverage function). Maximising a monotone submodular function
under a cardinality constraint with the greedy algorithm is guaranteed to reach within a factor
(1 − 1/e) of optimal [Nemhauser, Wolsey & Fisher 1978]; knapsack budgets admit similar guarantees
[Sviridenko 2004]. In practice greedy is often optimal on such instances: experiments E3 and E7 show
the cost-benefit greedy planner hitting the certified optimum on most instances. Any claim of a
quantum or quantum-inspired advantage on this problem therefore has to be made against greedy and
MILP baselines, not only against SA. `quhi` always reports them.

## 2. Exact penalty encodings (corrected Theorem IV.1)

**Setting.** Minimise f(x) subject to linear constraints aₖ·x (≤, ≥, =) bₖ with **integer**
aₖ, bₖ, and "at most one" groups G.

**Encoding.**
* Equality: (a·x − b)².
* `≤`: (a·x + Σ_j w_j s_j − b)², with slack weights w = (1, 2, 4, …, 2^{m−1}, r) whose subset
  sums are exactly {0, …, b − min a·x}. This is the *bounded* binary encoding; plain binary
  would admit slack values that are too large.
* `≥`: (a·x − Σ w_j s_j − b)², with range [0, max a·x − b].
* At-most-one: Σ_{i<j∈G} x_i x_j (no slack needed).

Let P(x,s) be the sum of these terms.

**Lemma.** P is integer-valued and non-negative. If x is infeasible then P(x,s) ≥ 1 for every s.
If x is feasible then min_s P(x,s) = 0.

*Proof.* Each squared term is the square of an integer. For infeasible x, some `≤` constraint has
a·x > b, so a·x + slack − b ≥ 1; the same argument works for `≥` and `=`. A violated
at-most-one group contains a pair (i,j) with x_i x_j = 1. For feasible x, the required slack
b − a·x lies in the representable range, so each term can be made 0. ∎

**Theorem (sufficient penalty).** Let f* be the constrained optimum and L ≤ min_x f(x) any lower
bound (for example L = c₀ + Σ_{c_T<0} c_T). If Λ > f* − L, every global minimiser of
F = f + ΛP is feasible, and its x-part is optimal.

*Proof.* If (x,s) has infeasible x, then F ≥ L + Λ > f*. The feasible optimum x* with its matching
slack achieves F = f*. Among feasible points, min_s F = f. ∎

`safe_penalty_weight` bounds f* above using a greedy-feasible point improved by feasibility-preserving
descent. On the E1 instances the result is about 2× smaller than the textbook range bound U − L
(median 2.06×). E1 checks by exhaustive enumeration that the resulting ground states equal the
MILP optimum. `tests/test_constraints.py` checks the lemma and the theorem on random programs, and
shows that too small a Λ fails.

**Why not simply make Λ huge?** It stays exact, but (i) it compresses the objective relative to
the energy range, which is fatal for analog hardware precision and for the annealing gap (E6), and
(ii) it makes the landscape rugged for local-move samplers (E4).

## 3. Quadratisation (HUBO → QUBO)

Rosenberg substitution [Rosenberg 1975; Boros & Hammer 2002] repeatedly replaces the most frequent
pair u·v in the terms of degree ≥ 3 by a new variable y. It adds
M·(uv − 2uy − 2vy + 3y), which is ≥ 0 and equals 0 exactly when y = uv.

**Sufficient M (proof of the bound used in `quhi.quadratize`).** Let G be the reduced objective
without penalties, and S = Σ|c_T| over the terms of G that contain an auxiliary variable. Fix x.
Suppose some auxiliary is inconsistent, and take the earliest-created such y. Its inputs are
original variables or earlier (consistent) auxiliaries, so its penalty is ≥ M. Meanwhile
G(x,y) ≥ G(x,y*) − S, because every term changes by at most |c_T|. Hence M > S implies
min_y F(x,y) = G(x,y*) = f(x). ∎
Tests: `tests/test_quadratize.py` (the minimum over the auxiliaries reproduces f for every x, and too small an M fails).

## 4. Algorithms

All heuristics share a numba kernel that keeps, for each term, the number of its variables that are
currently 0. A single-flip energy change and its update then cost O(deg_i) for QUBO **and** HUBO
alike, so no quadratisation is needed to *solve* a HUBO heuristically.

| Solver | Notes |
|---|---|
| `ExhaustiveSolver` | Gray-code enumeration, n ≤ 30; certifies optimality and degeneracy |
| `MILPSolver` | Linearises each term (y_T ≤ x_i, y_T ≥ Σx − \|T\| + 1) and solves the **constrained** program natively with HiGHS (no penalties or slack). This gives the ground truth |
| `SimulatedAnnealing` | Metropolis single-flip with geometric β [Kirkpatrick et al. 1983]; neal-style default range |
| `TabuSearch` | 1-flip, aspiration criterion, randomised tenure [Glover 1989] |
| `SimulatedQuantumAnnealing` | Path-integral MC [Martoňák, Santoro & Tosatti 2002]; see below |
| `QAOA` | Exact state vector, diagonal cost (HUBO native), linear-ramp initialisation, then L-BFGS-B [Farhi, Goldstone & Gutmann 2014] |
| `annealing_spectrum`, `schrodinger_anneal` | Exact spectrum / split-operator dynamics of H(s) = −(1−s)ΣX + s·H_P for n ≲ 16 |
| `postprocess` | Repair plus feasible 1-flip/swap descent on decision variables |

### SQA derivation (what the legacy code got wrong)
For H = H_P − Γ Σ σˣ, Suzuki–Trotter with P slices gives
Z ≈ Σ exp[ −(β/P) Σ_k E_P(s^k) + J⊥ Σ_{k,i} s_i^k s_i^{k+1} ], with J⊥ = ½ ln coth(βΓ/P).
The **β/P** on the problem term is essential; the legacy code used β (AUDIT F3). We validate the
action against the exact ⟨σz⟩ of a single qubit (unit test and audit A3).

`quhi` also adds global moves that flip a spin in every slice, which keeps the chain ergodic
when J⊥ is large. It supports two schedules: fixed temperature (textbook) and **joint**
annealing of β and Γ (the default). The joint schedule is needed because penalty-encoded
models span 3–5 decades of energy scale (E3). Note that discrete-time PIMC dynamics is not a
faithful model of physical QA dynamics [Heim et al. 2015]. SQA results here are a
quantum-*inspired* heuristic, not a hardware prediction.

### Adiabatic scale
The run time needed for adiabatic success scales roughly as 1/Δ_min² [Albash & Lidar 2018].
`annealing_spectrum` reports Δ_min, split into the interior avoided-crossing gap and the final
gap, which is set by how well the problem's energy levels can be resolved after penalty
scaling. `tests/test_solvers.py::test_slow_schrodinger_anneal_is_adiabatic` checks that
T = 20/Δ² reaches > 99% ground-state probability.

## 5. Metrics
* **Success**: the read is feasible **and** attains the MILP-certified constrained optimum (relative tolerance 1e-7).
* **TTS99** = t_read · ln(0.01) / ln(1 − p) [Rønnow et al. 2014]; single-core wall time.
* **Benefit ratio** = (f₀ − f(x)) / (f₀ − f*): the fraction of the optimal cooling achieved (infeasible reads count as 0).
* **Wilson 95% intervals** for success probabilities.
* **Compute-matched comparison.** SQA does (P+1)·n flip attempts per sweep, so "SA-matched" runs SA with (P+1)× more sweeps.

## References
* Zhou et al., *Optimization Methods of Urban Green Space Layout on Tropical Islands to Control Heat Island Effects*, Energies 16(1):368 (2023). https://doi.org/10.3390/en16010368
* *A study of size threshold for cooling effect in urban parks and their cooling accessibility and equity*, Sci. Rep. (2024). https://www.nature.com/articles/s41598-024-67277-2
* *Urban green space cooling effect in cities*, Heliyon (2019). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6458494/
* *The impact of urban parks on the thermal environment of built-up areas and an optimization method*, PLOS One (2025). https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0318633
* *Nonlinear changes in urban heat island intensity … with roof albedo* (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11496832/
* *Optimal Placement of Nature-Based Solutions for Urban Challenges*, arXiv:2502.11065 (classical optimisation; related problem framing).
* Lucas, *Ising formulations of many NP problems*, Front. Phys. 2:5 (2014).
* Glover, Kochenberger & Du, *Quantum Bridge Analytics I: a tutorial on formulating and using QUBO models*, 4OR 17 (2019).
* Rosenberg, *Reduction of bivalent maximization to the quadratic case*, Cahiers du CERO 17:71–74 (1975); Boros & Hammer, *Pseudo-Boolean optimization*, Discrete Appl. Math. 123 (2002).
* Nemhauser, Wolsey & Fisher, *An analysis of approximations for maximizing submodular set functions I*, Math. Prog. 14 (1978); Sviridenko, *A note on maximizing a submodular set function subject to a knapsack constraint*, Oper. Res. Lett. 32 (2004).
* Kirkpatrick, Gelatt & Vecchi, *Optimization by simulated annealing*, Science 220 (1983); Glover, *Tabu Search—Part I*, ORSA J. Comput. 1 (1989).
* Martoňák, Santoro & Tosatti, *Quantum annealing by the path-integral Monte Carlo method: the two-dimensional random Ising model*, Phys. Rev. B 66, 094203 (2002); Santoro et al., Science 295, 2427 (2002).
* Heim, Rønnow, Isakov & Troyer, *Quantum versus classical annealing of Ising spin glasses*, Science 348, 215 (2015).
* Farhi, Goldstone & Gutmann, *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028 (2014).
* Rønnow et al., *Defining and detecting quantum speedup*, Science 345, 420 (2014). https://arxiv.org/abs/1401.2910
* Albash & Lidar, *Adiabatic quantum computation*, Rev. Mod. Phys. 90, 015002 (2018).

Literature note: in a targeted search we did not find prior work that applies quantum annealing
or QAOA specifically to urban green/blue-space placement for heat mitigation. The closest work is
classical (GA / heuristic) green-space layout optimisation, plus QUBO work on other urban
location problems. We treat this as a gap, with the caveat of §1.6.
