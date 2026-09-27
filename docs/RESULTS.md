# Initial results

All numbers come from `python scripts/run_experiments.py` (seeded; environment recorded in
`results/env.json`: Python 3.11, numpy 2.4, scipy 1.17 / HiGHS, numba 0.67, 4-core container,
all solvers single-threaded). Raw rows are in `results/E*/…_raw.csv`, summaries in `…_summary.csv`,
and figures are next to them. The instances are **synthetic** cities (see `docs/BACKGROUND.md` §1).
Treat the numbers as a first characterisation of the toolkit and the problem class, not as
field-validated UHI predictions.

**Headline findings**

1. **The encoding is certified exact.** On all 11 instances, the ground state of the penalty QUBO,
   built with the corrected-theorem weight, equals the MILP constrained optimum. That weight is
   about 2× smaller than the textbook bound (E1).
2. **The Taylor-truncated QUBO is a poor surrogate once cooling reach is long.** At L = 300 m it
   loses 10.3% of the achievable cooling. The **cubic HUBO** (≤ 0.02% loss) and a
   **least-squares-fitted QUBO** of the same size (≤ 0.42% loss) fix this (E2).
3. **HUBOs should be solved natively.** Tabu finds the certified optimum on 99% of reads on the
   native quartic HUBO and on 0% after Rosenberg quadratisation, which needs 6.4× as many
   variables (E5).
4. **Penalty weight is a real trade-off.** The certified-safe weight gives 100% feasibility but a
   rugged landscape. On the constrained multi-intervention problem, 0.01× that weight plus
   classical repair raises the optimal rate from ≤ 5% to 42–47% for SA and Tabu (E4). The same
   penalties shrink the final annealing gap as 1/Λ, so the adiabatic time grows as Λ² (E6).
5. **No quantum-inspired advantage once compute is matched.** SQA beats SA at equal sweeps, but SA
   given the same number of spin-flip attempts is comparable on the park family (17 variables:
   P(opt) 0.54 ± 0.07 for SQA vs 0.50 ± 0.11 for SA-matched, over 12 instance-seed runs).
   On the harder mix + equity family, SQA reaches a slightly higher mean benefit ratio
   (0.83 vs 0.81, 0.79 vs 0.77, 0.71 vs 0.68) at about 2.4× the wall time. Tabu dominates the
   single-intervention family (E3).
6. **Problem structure favours classical greedy.** The objective is submodular (BACKGROUND §1.6).
   The cost-benefit greedy planner is optimal on 8 of 15 park instances and on the E7 showcase,
   and within 0.04 °C elsewhere. It does, however, violate equity on 5 of 9 mix instances. Any
   quantum claim for this application must beat greedy and MILP, and on these sizes nothing does.

---

## E1: encoding certification
`results/E1_encoding/encoding_certification.csv`.
Exhaustive enumeration of the full penalty QUBO (decision + slack variables) is compared with
HiGHS on the constrained program.

| instance | decision | slack | total | MILP optimum (°C) | QUBO ground (°C) | match | Λ_safe | Λ_range | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 6x6_s0_park | 5 | 3 | 8 | 30.818 | 30.818 | ✓ | 1.91 | 3.59 | 1.88 |
| 6x6_s1_park | 5 | 3 | 8 | 30.755 | 30.755 | ✓ | 1.83 | 3.36 | 1.84 |
| 6x6_s2_park | 5 | 3 | 8 | 30.720 | 30.720 | ✓ | 1.85 | 3.42 | 1.85 |
| 6x6_s3_park | 5 | 3 | 8 | 30.823 | 30.823 | ✓ | 2.05 | 3.71 | 1.81 |
| 8x8_s0_park | 12 | 4 | 16 | 30.584 | 30.584 | ✓ | 2.96 | 6.29 | 2.12 |
| 8x8_s1_park | 12 | 4 | 16 | 30.507 | 30.507 | ✓ | 2.33 | 5.15 | 2.21 |
| 8x8_s2_park | 12 | 4 | 16 | 30.738 | 30.738 | ✓ | 2.42 | 5.30 | 2.19 |
| 8x8_s3_park | 12 | 4 | 16 | 30.757 | 30.757 | ✓ | 2.71 | 5.81 | 2.15 |
| 6x6_s0_mix | 15 | 3 | 18 | 30.369 | 30.369 | ✓ | 4.85 | 10.01 | 2.06 |
| 6x6_s1_mix | 15 | 3 | 18 | 30.339 | 30.339 | ✓ | 4.54 | 9.16 | 2.02 |
| 8x8_s1_park + equity | 12 | 11 | 23 | 30.552 | 30.552 | ✓ | 2.40 | 5.15 | 2.14 |

One configuration (8x8_s0 with equity) was skipped because its equity demands exceed the budget,
so it is infeasible. Some 6×6 instances show ground-state degeneracy 2. This is an artefact of the
bounded slack encoding: when the last weight repeats an earlier one, two slack patterns give the same
sum. It is harmless for exactness.

## E2: surrogate fidelity (additive vs QUBO vs HUBO vs fitted QUBO)
`results/E2_fidelity/`. Setup: 12×12 cities, 22 candidate cells, park budget ≤ 6, 4 seeds per
decay length. Each surrogate's own optimum is found by enumerating all feasible plans. That plan
is then scored on the **exact** saturating objective and compared with the exact optimum.

| L (m) | additive K=1 | QUBO K=2 (Taylor) | QUBO (least-squares fit) | HUBO K=3 |
|---|---|---|---|---|
| 50 | 0.20% | 0.00% | 0.00% | 0.00% |
| 100 | 0.00% | 0.20% | 0.02% | 0.00% |
| 200 | 0.29% | 2.81% | 0.21% | 0.02% |
| 300 | 0.00% | **10.30%** | 0.42% | 0.00% |

*(Mean share of the achievable cooling benefit lost. Surrogate error at the chosen plan grows with L
for every order: see `fidelity.png`, right panel.)*

**Interpretation.** When kernels overlap strongly (Σ_v p_kv ≳ 1), the second-order truncation
over-counts diminishing returns. The QUBO then believes clustered parks are much worse than they are
and spreads them out (`plans_L200_seed0.png`). The additive model is badly wrong in *value*
(errors up to 1.9 °C) but ranks plans reasonably well, because budgets are small. The cubic term
restores the ranking. Fitting the same 253 QUBO coefficients by least squares on budget-feasible
plans keeps the loss within 0.42% of optimal at every L, which makes it the practical choice when
hardware or a solver requires a QUBO. The number of terms grows from 22 (K=1) to 253 (K=2) to
1,793 (K=3).

## E3: solver scaling (certified optima, safe penalty)
`results/E3_scaling/`. Two families × 3 instances per size × 4 seeds × 32 reads. Budgets:
1,000 sweeps for SA/SQA, SQA with P = 16 slices. "SA-matched" gets (P+1)× more sweeps, so it makes
as many flip attempts as SQA. QAOA is only run where the state vector fits (≤ 16 qubits).

**Raw success probability** (a read is feasible **and** optimal; median over instances × seeds):

| family / QUBO vars | Greedy | SA | SA-matched | SQA | SQA-fixedT | Tabu | QAOA p=4 |
|---|---|---|---|---|---|---|---|
| park / 13 | 0.20 | 0.48 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| park / 17 | 0.06 | 0.03 | 0.48 | 0.56 | 0.84 | 1.00 | – |
| park / 27 | 0 | 0 | 0 | 0 | 0 | **1.00** | – |
| park / 32 | 0 | 0 | 0 | 0 | 0 | **0.94** | – |
| park / 46 | 0 | 0 | 0 | 0 | 0 | **0.66** | – |
| mix+equity / 42–89 | 0 | 0 | 0 | 0 | 0 | 0 | – |

**Mean benefit ratio per read** (fraction of optimal cooling achieved; infeasible = 0), with time per read:

| family / vars | SA | SA-matched | SQA | SQA-fixedT | Tabu | ms/read (SA, SA-m, SQA, Tabu) |
|---|---|---|---|---|---|---|
| park / 27 | 0.909 | 0.963 | 0.961 | 0.965 | 1.000 | 2.5, 42, 85, 1.9 |
| park / 46 | 0.905 | 0.946 | 0.945 | 0.944 | 1.000 | 5.6, 96, 242, 8.8 |
| mix+eq / 42 | 0.617 | 0.805 | **0.832** | 0.512 | 0.703 | 4.4, 74, 174, 4.1 |
| mix+eq / 56 | 0.592 | 0.767 | **0.787** | 0.357 | 0.764 | 7.0, 120, 295, 12.9 |
| mix+eq / 89 | 0.514 | 0.683 | **0.712** | 0.201 | 0.564 | 14.6, 247, 615, 36.7 |

After post-processing (repair plus feasible swap descent, `*_pp` columns) every sampler, even
`Random`, solves most park instances. Post-processed success therefore does *not* discriminate
between samplers on the park family. On mix + equity, the best post-processed rates are SA-matched
0.27/0.19/0.03 and SQA 0.20/0.12/0.02.

**Readings.**
* The problem-aware schedule matters. The fixed-temperature SQA is best on the easy family but
  collapses in feasibility (0.42–0.78) on mix + equity, where the energy scales span decades.
* The SQA-over-SA advantage at equal sweeps largely disappears at equal flips on the park family
  (17 variables: 0.54 ± 0.07 vs 0.50 ± 0.11). On mix
  + equity, SQA's +0.02–0.03 benefit ratio over SA-matched costs about 2.4× wall time.
* Tabu is the strongest sampler on single-intervention QUBOs and weaker on the constrained
  multi-choice family.
* QAOA at p = 4 on 13 qubits concentrates only 1% probability on the optimum. That is above the
  uniform 1/2¹³, but far from useful (see E6).
* Greedy planner (`greedy_planner.csv`): optimal on 8 of 15 park instances, with worst gap
  0.037 °C. On mix + equity it is optimal on 4 of 9 and **infeasible on 5 of 9**, because it
  ignores equity. Its objective values there are below the optimum only *because* it violates equity.

## E4: penalty-weight sweep
`results/E4_penalty/penalty_sweep.png`. The weight is Λ/Λ_safe ∈ {0.01, …, 3}. 3 seeds × 32 reads.

| instance | solver | Λ/Λ_safe | P(feasible) raw | P(opt) raw | P(opt) after pp | benefit after pp |
|---|---|---|---|---|---|---|
| mix 8×8 + equity | SA | 0.01 | 0.10 | 0.01 | **0.42** | 0.78 |
| | SA | 1 | 1.00 | 0 | 0.03 | 0.77 |
| | SQA | 0.01 | 0.00 | 0 | 0.29 | **0.94** |
| | SQA | 1 | 1.00 | 0 | 0.01 | 0.89 |
| | Tabu | 0.01 | 0.20 | 0.01 | **0.47** | 0.83 |
| | Tabu | 1 | 1.00 | 0 | 0 | 0.71 |
| park 12×12 | all | ≤ 0.03 | ≤ 0.15 | ≈ 0 | 1.00 | 1.00 |
| | Tabu | ≥ 0.1 | 1.00 | 0.97–0.99 | 1.00 | 1.00 |

Below about 0.1 Λ_safe, raw feasibility collapses, as the theory predicts (the ground state is no
longer guaranteed feasible). The infeasible low-penalty samples are nonetheless *better starting
points* for repair on the constrained problem. The recommended recipe for hard constrained
instances is therefore: a weak penalty, then post-processing. That recipe is neither certified nor
end-to-end quantum; it is hybrid.

## E5: native HUBO vs quadratised QUBO
`results/E5_hubo/`. Setup: 10×10 cities, 30 decision variables, cubic saturation plus quartic
park-block synergy (degree 4). The optimum of the constrained HUBO is certified by MILP on all
three instances.

| formulation | vars | solver | P(opt) raw | P(opt) pp | mean benefit | ms/read |
|---|---|---|---|---|---|---|
| native HUBO | 30 | **Tabu** | **0.990** | 1.000 | **0.999** | 179 |
| | | SA | 0 | 0.997 | 0.920 | 14 |
| | | SQA | 0 | 0.993 | 0.960 | 573 |
| | | Greedy descent | 0.003 | 0.976 | 0.921 | 0.2 |
| QUBO (Rosenberg) | 187–197 | Tabu | 0 | 1.000 | 0.847 | 29 |
| | | SA | 0 | 1.000 | 0.837 | 24 |
| | | SQA | 0 | 0.997 | 0.930 | 1,115 |

Quadratisation adds 157–167 auxiliary variables, each tied to its pair by penalties of size M > S.
This creates deep narrow valleys that single-flip dynamics cannot cross, because changing an
original variable requires coordinated auxiliary flips. The numba kernels' support for native HUBOs
(O(deg) delta updates for any degree) is the enabling feature here. For gate-model hardware, the
same argument favours phase-separator circuits that implement k-local Z terms directly over
quadratisation.

## E6: small-scale quantum simulation (14 qubits)
`results/E6_quantum/`. Setup: 7×7 city, 10 decision + 4 slack = 14 qubits.
H(s) = −(1−s)ΣX + s·H_P, with H_P normalised to [0,1].

| Λ/Λ_safe | ground state = optimum | final gap Δ(s=1) | 1/Δ² | P(ground), anneal T=10 / 100 / 1000 | QAOA P(ground) p=1 / 2 / 4 / 6 | QAOA approx. ratio p=6 |
|---|---|---|---|---|---|---|
| 0.1 | ✓ | 7.1e-5 | 2.0e8 | 1.0e-4 / 2.3e-4 / 5.8e-4 | 0.00085 / 0.0011 / 0.0017 / 0.0019 | 0.9963 |
| 0.3 | ✓ | 2.4e-5 | 1.8e9 | 1.0e-4 / 2.3e-4 / 5.6e-4 | 0.00085 / 0.0011 / 0.0016 / 0.0018 | 0.9977 |
| 1 | ✓ | 7.1e-6 | 2.0e10 | 1.0e-4 / 2.3e-4 / 5.5e-4 | 0.00085 / 0.0010 / 0.0016 / 0.0017 | 0.9981 |
| 3 | ✓ | 2.4e-6 | 1.8e11 | 1.0e-4 / 2.3e-4 / 5.5e-4 | 0.00075 / 0.0010 / 0.0016 / 0.0017 | 0.9982 |
| 10 | ✓ | 7.1e-7 | 2.0e12 | 1.0e-4 / 2.3e-4 / 5.5e-4 | 0.00075 / 0.0010 / 0.0016 / 0.0017 | 0.9980 |

Uniform random sampling finds the optimum with probability 1/2¹⁴ = 6.1e-5.

**Readings.**
* **The gap is set by the penalty.** The final gap scales exactly as 1/Λ: the objective
  differences shrink relative to the spectrum width, which the penalty dominates. The minimum gap
  sits at s = 1. On the grid (s ≤ 0.95 by Lanczos, plus s = 1 exactly), the lowest gap decreases
  monotonically toward the end, with no earlier avoided crossing; the "interior" value 0.10 is just
  2(1−s) at s = 0.95. The adiabatic time scale 1/Δ² therefore grows as Λ², from 2×10⁸ to 2×10¹²
  (in units of the normalised H_P). It is not the classical hardness of the instance that makes
  analog QA hard here; it is the energy resolution the penalty encoding demands.
* **Practical anneal times are far from adiabatic.** At T ≤ 1000 the ground-state probability is
  about 9× uniform and essentially independent of Λ.
* **QAOA's approximation ratio is misleading.** It is ≈ 0.998, but it is measured on a
  penalty-dominated scale, so it mostly means "feasible and low-penalty". The probability of the
  actual optimum only grows from 0.08% (p = 1) to 0.17–0.19% (p = 6), about 30× uniform, and
  falls slightly as Λ grows.
* **Implications.** A useful quantum approach for this problem class needs penalty-light encodings:
  constraint-preserving mixers (XY/Dicke-state QAOA for budget/cardinality), native k-local terms
  (E5), or weak penalties plus classical repair (E4).

## E7: end-to-end showcase
`results/E7_showcase/`: `city.png`, `transition.png`, `cooling_milp.png`, `convergence.png`.
Setup: a 14×14 city (700 m × 700 m) with parks, water and cool pavement, a budget of 24 units and
at least one intervention per district. That gives 81 decision variables and a 105-variable QUBO
at 0.1 Λ_safe.

| plan | feasible | exposure T (°C, exact physics) | pop.-weighted cooling | interventions | time |
|---|---|---|---|---|---|
| baseline | – | 32.311 | – | 0 | – |
| **MILP (certified optimal)** | ✓ | **31.141** | **1.231 °C** | 8 parks | 10.8 s |
| Greedy planner | ✓ | 31.141 (same plan) | 1.231 °C | 8 parks | < 1 s |
| SA + pp | ✓ | 31.230 | 1.140 °C | 10 | 12.4 s |
| SQA + pp | ✓ | 31.230 (same plan as SA) | 1.140 °C | 10 | 32.1 s |
| Tabu + pp | ✓ | 31.459 | 0.891 °C | 14 | 10.8 s |

The optimal plan lowers the exposure-weighted temperature by 1.17 °C. All 20 baseline hot-spot
cells (top 10% of baseline temperatures) drop below that threshold, and so they do under every
heuristic plan. Water bodies are never selected:
at 5 cost units for 2.0 °C they are dominated by parks (3 units for 1.5 °C) under these illustrative
costs. The hottest core receives no intervention because the synthetic generator places few
candidate lots in dense blocks, which is realistic, but it shows that the siting constraint, not
the optimiser, limits what can be achieved.

---

## Limitations and next steps
* **Data.** Replace synthetic cities with real land-use, LST (e.g. Landsat) and census layers, and
  calibrate A_o and L_o per intervention against local measurements.
* **Physics.** The saturation law is a surrogate. The fitted-QUBO machinery (E2) can equally be
  trained on CFD or ENVI-met runs, which is the path to a data-driven QUBO.
* **Scale.** MILP certification becomes slow above about 60 dense decision variables (the 20×20
  pilot was not certified in 120 s). Larger studies should report best-known solutions and gaps
  instead.
* **Quantum.** SQA is quantum-*inspired*; discrete-time PIMC dynamics does not model QA hardware
  [Heim et al. 2015]. A hardware study should use native k-local encodings or fitted QUBOs with
  weak penalties plus classical repair, and must benchmark against greedy and MILP.
* **Statistics.** 3 instances × 4 seeds per cell are enough to see the trends reported here, but
  not for fine-grained scaling exponents.

---

# Next slice: E8–E11 (see docs/NEXT.md)

Success criterion: match the **greedy planner and HiGHS MILP** at matched wall-clock on both
families. Matching SA is not enough.

## E8: constraint-preserving mixers
`results/E8_mixers/`. Setup:
- **Park instances (8):** the E6 instance, its "exactly 3 parks" twin (a pure Dicke/XY case),
  and E3 park 8×8 and 10×10 (3 seeds each).
- **Mix+equity instances (6):** 6×6 and 7×7 cities with three intervention types, equity, and
  |F| from 720 to 2,243.
- **QAOA settings:** p ∈ {1, 2}; one L-BFGS-B run from a linear ramp plus one random restart.
- **Heuristics:** 32 reads × 3 seeds.

P(opt) is exact for QAOA and per read for heuristics. Benefit is measured on the exact
saturating physics relative to the MILP plan; infeasible counts as 0.

| family | method | P(opt) | P(feasible) | benefit (exact) | wall s |
|---|---|---|---|---|---|
| park | MILP | 1 | 1 | 1 | 0.09 |
| park | Greedy planner | 0.86 | 1 | 1.00 | 0.001 |
| park | XY-QAOA (swap+add/remove) p=1 / p=2 | 0.063 / 0.197 | **1** | 0.87 / 0.94 | 1.6 / 7.1 |
| park | X-mixer penalty QAOA p=1 / p=2 (≤16 qubits) | 0.008 / 0.011 | 0.70 / 0.80 | 0.40 / 0.51 | 0.6 / 1.1 |
| park | FeasibleSA | **1** | 1 | 1 | 0.26 |
| park | SA (safe-Λ QUBO) | 0.25 | 1 | 0.96 | 0.07 |
| park | Tabu (safe-Λ QUBO) | 1 | 1 | 1 | 0.02 |
| exactly-3 | Dicke-XY complete p=1 / p=2 | 0.18 / 0.37 | **1** | 0.94 / 0.97 | 3 / 15 |
| exactly-3 | Dicke-XY ring p=1 / p=2 | 0.04 / 0.14 | 1 | 0.88 / 0.91 | 3 / 16 |
| exactly-3 | X-mixer penalty QAOA p=1 / p=2 | 0.004 / 0.005 | 0.45 / 0.62 | 0.35 / 0.49 | 0.2 / 0.3 |
| mix+eq | MILP | 1 | 1 | 1 | 0.18 |
| mix+eq | Greedy planner | 0.33 | 0.33 | 0.33 | 0.002 |
| mix+eq | XY-QAOA p=1 / p=2 | 0.002 / 0.003 | **1** | 0.60 / 0.61 | 3.6 / 25 |
| mix+eq | FeasibleSA | **0.39** | 1 | **0.92** | 0.79 |
| mix+eq | SA / Tabu (safe-Λ QUBO) | 0.002 / 0 | 1.0 | 0.61 / 0.61 | 0.2 / 0.14 |

Two further notes:
- **Benefit slightly above 1.** The greedy planner scores 1.0001 on park instances because it
  optimises the exact physics, whereas MILP optimises the pessimistic QUBO surrogate.
- **Skipped.** X-mixer QAOA is omitted wherever the penalty QUBO needs more than 16 qubits
  (all 10×10 park and all mix instances).

**Readings.**
- **Constraint-preserving mixers remove the penalty failure mode.** P(feasible) = 1 by
  construction. At equal depth, P(opt) is **10–70× higher** than X-mixer penalty QAOA on the
  same instances, e.g. 0.37 vs 0.005 for Dicke-XY vs X-mixer at p=2 on "exactly 3".
- **They do not reach the classical bar.** On park instances greedy, Tabu and FeasibleSA find
  the optimum essentially every time, in milliseconds. On mix+equity, XY-QAOA at p ≤ 2 barely
  beats uniform sampling of F (P(opt) ≈ 0.002), while FeasibleSA reaches 0.39 per read and a
  benefit of 0.92.
- **Complete beats ring.** Complete-graph XY mixing outperforms the ring XY mixer.
- **Negative result.** Constraint-preserving QAOA closes the gap to penalty QAOA, not to
  greedy or MILP.

## FeasibleSA (fair classical annealer)
FeasibleSA is Metropolis on feasible decision vectors only: add/remove and swap moves,
energy = f(x), no Λ. It is the strongest heuristic on mix+equity in E8 (0.39 vs ≤ 0.002 for
penalty SA and Tabu) and in E9 (P(opt) 0.46 per read, benefit 0.96).

## E9: weak penalty + repair as the declared hybrid recipe
`results/E9_hybrid/`. Setup: E3 mix+equity 8×8, 10×10 and 12×12 cities (3 seeds each, all
certified by MILP), 2 sampler seeds, α·Λ_safe with α ∈ {0.01, 0.03, 0.1, 1}. Raw and
post-processed (pp) columns are reported separately; pp means repair plus feasible local search.

| solver | α | P(feas) raw | P(opt) raw | P(opt) pp | benefit pp | s/read (pp) |
|---|---|---|---|---|---|---|
| SA | 0.01 / 1 | 0.61 / 0.98 | 0.02 / 0 | 0.50 / 0.03 | 0.94 / 0.81 | 0.07 |
| SA-matched | 0.01 / 1 | 0.72 / 1 | 0.17 / 0.003 | **0.80** / 0.18 | **0.98** / 0.90 | 0.25 |
| SQA (PIMC) | 0.01 / 1 | 0.83 / 1 | 0.16 / 0.003 | 0.77 / 0.11 | 0.98 / 0.88 | 0.53 |
| Tabu | 0.01 / 1 | 0.79 / 0.98 | 0.08 / 0 | 0.32 / 0.01 | 0.90 / 0.75 | 0.07 |
| FeasibleSA | – | 1 | 0.46 | – | 0.96 (raw) | 0.13 |
| Greedy planner | – | 0.44 | 0.44 | – | 0.44 | 0.007 |
| MILP | – | 1 | 1 | – | 1 | 3.8 |

**Readings.**
- **The hybrid recipe works.** α = 0.01 plus repair raises P(opt) from ≤ 0.18 (safe Λ) to
  0.77–0.80 for SA-matched and SQA.
- **SQA gains nothing over compute-matched SA** (0.77 vs 0.80), at about 2× the time per read.
- **Greedy is fast but unreliable here.** It violates equity on 5 of 9 instances. MILP is
  certified and takes 3.8 s on average.
- **This is hybrid, not end-to-end quantum.** Nothing here meets MILP's P(opt) = 1.

## E10: penalty form (quick run: 2 instances)
`results/E10_penalty_form/`. Unbalanced and linear penalties need no slack bits (10 vs 13 variables here).

| form | dynamic range | ground state feasible | ground state optimal | SA P(opt) raw | P(opt) pp |
|---|---|---|---|---|---|
| quadratic slack, Λ_safe (exact) | 163 | 1 | 1 | 0.34 | 0.72 |
| unbalanced 0.1·Λ_safe | **16.5** | 1 | 1 | **0.53** | **0.86** |
| unbalanced 1·Λ_safe | 163 | 1 | 1 | 0.56 | 0.81 |
| linear 0.5·μ / 1·μ / 2·μ | 0.4–1.0 | 0 / 0.5 / 1 | 0 / 0.5 / 0 | 0 / 0.5 / 0 | 1 / 0.5 / 1 |

- **Unbalanced penalty at 0.1·Λ_safe helps** on these two instances: 10× smaller dynamic range
  and better samples, with the ground state still optimal.
- **But it is not certified in general.** Its exactness is instance-dependent.
- **The linear penalty is a Lagrangian relaxation.** Its ground states are feasible or optimal
  only for lucky multipliers.
- **Small sample.** This needs the full run before any claim is made.

## E11: native HUBO vs Rosenberg QUBO spectra (quick run: 2 instances)
`results/E11_pubo_gap/`. Setup: cubic (K=3) objective, 6 decision variables.

| model | qubits | final gap | P(ground), T=10 |
|---|---|---|---|
| A: native cubic HUBO + slack penalty | 9 | 9.4e-5 / 1.1e-5 | 3.1e-3 |
| B: Rosenberg QUBO of A | 16 | 9.4e-5 / 1.1e-5 | 2.4e-5 |
| C: native HUBO on F (infeasible padded, diagonal only) | 6 | 2.4e-2 / 3.7e-3 | 4.5e-2 / 4.1e-2 |

- **Quadratisation does not change the bottleneck gap.** The final gap is set by the slack
  penalty, and A and B are identical to three digits.
- **It still costs a lot.** B needs 7 more qubits, and the short-anneal P(ground) falls about
  100×.
- **Removing the penalty helps most.** C's gap is about 250× larger, consistent with E6: the
  penalty, not the degree, controls the gap. C is a diagonal-only simulation and is not
  hardware-realisable.

## Verdict of this slice
No quantum or quantum-inspired method reached the success criterion:
- **Park family:** greedy, Tabu and FeasibleSA find the certified optimum every time in
  milliseconds; XY-QAOA peaks at P(opt) ≈ 0.2–0.4.
- **Mix+equity family:** the best non-MILP methods are the hybrid recipe (SA-matched or SQA with
  weak Λ plus repair, P(opt) ≈ 0.8) and FeasibleSA (≈ 0.4–0.5 per read, benefit 0.92–0.96).
  SQA shows no advantage over compute-matched SA.
- **Mixers help, but not enough.** Constraint-preserving mixers are a real improvement over
  penalty QAOA (10–70×), yet remain well below classical at p ≤ 2.

## E8b: Dicke-XY depth curve (is p ≤ 2 too shallow?)
`results/E8b_depth/`. Setup:
- **Instances:** three "exactly 3 parks" cities, each a 7×7 city built like the E6/E8 twin
  (seeds 4, 5, 6; seed 4 is the E8 instance). Each has n = 10, k = 3, |F| = C(10,3) = 120 and a
  single certified optimum. No slack.
- **QAOA:** L-BFGS-B from the linear ramp plus one random restart, maxiter 300. P(opt) is the
  exact state-vector probability.
- **FeasibleSA:** as in E8, 1,000 sweeps × 32 reads × 3 seeds.
- **X-mixer penalty QAOA:** runs at every p, since the penalty QUBO has 10 qubits.

Values are the mean over the 3 cities, with min–max in brackets.

| method | p=1 | p=2 | p=3 | p=4 | p=6 |
|---|---|---|---|---|---|
| Dicke-XY complete | 0.134 [0.05–0.18] | 0.242 [0.08–0.37] | 0.309 [0.11–0.47] | 0.376 [0.13–0.56] | **0.427** [0.17–0.65] |
| Dicke-XY ring | 0.034 | 0.067 | 0.057 | 0.066 | 0.111 [0.02–0.26] |
| X-mixer penalty QAOA | 0.004 | 0.005 | 0.006 | 0.008 | 0.009 |

Classical references on the same instances:

| method | P(opt) | benefit (exact) | wall s |
|---|---|---|---|
| uniform on F | 1/120 = 0.0083 | 0.85 | – |
| FeasibleSA | **1.000** | 1.000 | 0.21 |
| Greedy planner | 0.67 (2 of 3) | 1.000 | 0.001 |
| MILP (HiGHS) | 1 | 1 | 0.32 |

Expected benefit for Dicke-XY complete rises from 0.955 (p=1) to 0.993 (p=6). At p=6, wall time is
8–15 s per city (2,200–4,200 function evaluations), all in the classical angle optimiser.

- **Depth helps, then flattens.** P(opt) grows roughly logarithmically in p (0.13 → 0.43 from p=1
  to p=6) and is still below 0.5 on average at p=6. The worst city reaches only 0.17. None of the
  curves approach FeasibleSA (1.0) or MILP. The p ≤ 2 verdict of E8 stands: going deeper does not
  close the gap on the smallest instance family we have.
- **Mixer connectivity matters more than depth.** Ring-XY at p=6 (0.11) is below complete-XY at
  p=1 (0.13). The ring curve is not monotone in p (0.067 at p=2 against 0.057 at p=3; one city drops to 1e-5 at p=3), a sign that
  one random restart does not reliably find the global angle optimum. The restart count was not
  reduced from E8.
- **The penalty X-mixer barely beats uniform.** At p=6 it puts 0.9991 of its mass on feasible
  strings, but P(opt) is 0.0086, essentially uniform over F (1/120 = 0.0083). Its dynamics learn
  the constraint, not the objective. The E6 finding (penalty gap ∝ 1/Λ) is the reason.
- **Greedy misses the surrogate optimum on one city (s6), yet its exact-physics benefit is 1.001.**
  Greedy optimises the exact saturating objective, while the certified optimum is for the K=2
  surrogate. This shows the surrogate's truncation error (E2), not a greedy failure.
- **ConstrainedQAOA is still a simulation.** It runs exactly in span(F); it is not a compiled gate
  circuit (docs/BACKGROUND.md §6).

## Animations
`results/ANIM/` holds the GIFs and `manifest.json` (instance, n, frame count, fps and the fixed
colour scales). Regenerate with `python scripts/run_experiments.py --only ANIM`, which takes about
5 minutes. Settings: at most 40 frames per GIF at 10 fps, width ≤ 1,400 px, 8.1 MB in total.

There are two showcases:
- **A:** the E8b seed-4 "exactly 3 parks" city (n = 10, |F| = 120).
- **B:** the E9 10×10 mix+equity city, seed 0 (n = 39, three intervention types, budget 18,
  equity on).

Every frame's header gives the method, the step, the model energy, the exact-physics exposure
temperature, spend/budget and whether the plan is feasible. Each temperature scale is fixed to the
instance's baseline T_min–T_max for all of its GIFs, and each cooling scale is fixed to the largest
ΔT any recorded plan reaches. Every map is computed from a real plan x.

**Each method lives in a different space, and the GIFs do not pretend otherwise.**
- **ConstrainedQAOA** is a probability distribution over the feasible set F. Its
  `search_dicke_xy_complete_*` GIFs show |ψ|² over F, sorted cheapest → hottest, with the certified
  optimum in green. They start from the Dicke (uniform-on-F) state and step through the L-BFGS-B
  iterates of the p=6 circuit, reaching P(opt) = 0.65. Most of the remaining mass (0.31) sits on the
  second-best plan. A companion `search_dicke_xy_complete_layers_*` GIF shows the state after each
  of the 6 layers, using the optimised angles.
- **X-mixer penalty QAOA** is a distribution over all 2¹⁰ bitstrings, sorted by f+ΛP.
  `search_xmixer_penalty_qaoa_*` shows mass leaving infeasible strings (red) until P(feasible)
  reaches 0.999, while P(opt) stays at uniform-over-F level.
- **SA, Tabu and SQA** are single trajectories on the penalty QUBO, so their y-axes read f+ΛP.
  FeasibleSA walks on feasible plans only, so its y-axis reads constrained f(x). The two energy
  scales are not comparable.
- **Recording method.** FeasibleSA and SA incumbents are snapshotted by running one geometric β
  schedule as 40 consecutive pieces, each continuing from the previous piece's state. The solver API
  is unchanged.
- **Tabu and SQA** show their real energy traces, but the grid panel holds the final plan, because
  intermediate states are not recorded. SQA Trotter slices are never drawn as maps.

**Physical space.** In the `uhi_*` GIFs, each frame has three panels: land use with the plan,
the air-temperature field, and the cooling ΔT. The first frame is always the MILP reference. The
Dicke-XY UHI GIFs show successive draws from the final |ψ|², labelled as such, not as annealing
time. The E9 hybrid GIF ends with the plan after repair and feasible local search.

**Comparison.** `compare_temp_*` is the figure to watch: greedy | FeasibleSA | Dicke-XY p=6 |
MILP, each column on the same locked temperature scale and resampled to 40 frames.
- **Showcase A:** all four columns end on the same plan (T = 29.571 °C). The QAOA column keeps
  flickering to other plans, because its draws still miss the optimum 35% of the time.
- **Showcase B:** |F| > 4,000, so the QAOA column is replaced by the E9 hybrid (SA-matched
  α=0.01 + repair), and the frame title says so. The greedy column ends **infeasible**, because
  greedy does not target equity.

Stills for a paper: `compare_final.png` (A; the QAOA panel shows the most probable plan) and
`compare_final_B_mix_10x10_s0.png`.

## E12: comparable-performance bakeoff
*(The definition and conventions below were committed before E12 was run.)*

**Families.**
- **H (hard):** 8×8, 10×10 and 12×12 cities (seeds 100–102; candidate fraction 0.3) with park, water and cool pavement, budget = R (grid side), and at least one intervention per district. The 10×10 and 12×12 cities add a 0.3 °C park-block synergy (quartic terms wherever a 2×2 candidate block exists). With budget = R, the greedy planner is infeasible on 6 of 9 seeds.
- **C (control):** three park-only 8×8 cities (seeds 100–102).
- **MILP:** HiGHS with an 8 s cap; the gap is recorded if it does not certify.
- **Feasible set:** |F| is enumerated up to 20,000 and reported as ">20k" above that.

**COMPARABLE.** A method is comparable on a family if, at matched wall-clock T\* per instance, all three hold:
1. P(feasible) = 1.
2. Its family-mean benefit_true is at least 0.95 × the best classical family-mean benefit_true.
3. No run exceeds 1.5 T\*.

The pieces of that definition:
- **benefit_true** is exact saturating (plus synergy) cooling of the returned plan, divided by the cooling of the MILP plan. If MILP is not certified, the divisor is the best feasible plan any method returned. Infeasible counts as 0. benefit_true can exceed 1, because MILP certifies the K=2 surrogate, not the exact physics.
- **T\*** = max(median wall time of default FeasibleSA over two seeds [1,000 sweeps, 32 reads], HiGHS wall time capped at 8 s).
- **Best classical** is the best of: HiGHS at the cap, FeasibleSA, the greedy planner, Tabu-on-F, and SA-matched (α = 0.01) + repair.

**Convention (applied to every heuristic).**
- Each (instance, sampler seed) run gets T\* of wall-clock, as repeated batches of reads.
- A batch that would end after T\* is not started, or is discarded if it runs over.
- The method returns its best feasible plan, and that plan is scored.
- There are two sampler seeds per instance.
- QAOA returns the best of 256 shots drawn from the final state.
- P(opt) is the fraction of runs whose returned plan attains the certified optimum. It is reported as secondary, and only on certified instances.

**Contestants.**
- **FeasibleSQA:** PIMC with P = 8 slices, each slice feasible, 500 sweeps, 2 reads per batch.
- **QAOA-seeded FeasibleSA:** quantum seeds only when |F| ≤ 4,000.
- **ConstrainedQAOA p = 2 and p = 4:** only when |F| ≤ 4,000.
- **X-mixer penalty QAOA:** a negative control, run only when the penalty QUBO has ≤ 14 qubits.

The safe-Λ slack QUBO and the Rosenberg QUBO are not contestants.

### E12 results
`results/E12_comparable/` holds `raw.csv`, `summary.csv`, `e12.md`, `benefit_vs_time.png` and the returned-plan grids `plans_H_10x10_s100.png` and `plans_C_park_8x8_s100.png`. The full run took 16 min on an otherwise idle machine; a first run that overlapped with pytest was discarded.

**Instances.**
- **T\*:** 1.4–1.6 s on H 8×8, 4.0–4.7 s on H 10×10, 18–20 s on H 12×12 (set by FeasibleSA's default run on the quartic synergy objective with 66 bits), and 0.13–0.15 s on C.
- **MILP:** certified all 12 instances, in 0.81 s on average for H.
- **|F|:** 822–2,484 on H 8×8 and >20k on H 10×10 and 12×12.

**Family H**, 9 instances × 2 sampler seeds. benefit_true is the mean, with the minimum in brackets.

| method | kind | P(feas) | P(opt) | benefit 8×8 / 10×10 / 12×12 | benefit_true | wall s | comparable |
|---|---|---|---|---|---|---|---|
| MILP (HiGHS, 8 s cap) | classical | 1 | 1 | 1 / 1 / 1 | **1.000** | 0.81 | bar |
| Greedy planner | classical | 0.33 | 0.33 | 0 / 0.67 / 0.33 | 0.333 [0] | 0.009 | no |
| FeasibleSA | classical | 1 | 0.78 | 1 / 1 / 0.920 | 0.973 [0.81] | 7.7 | yes |
| Tabu-on-F | classical | 1 | 0.78 | 1 / 0.978 / 0.951 | 0.976 [0.71] | 8.1 | yes |
| SA-matched α=0.01 + repair | classical/hybrid | 0.83 | 0.72 | 0.493 / 1 / 1 | 0.831 [0] | 7.8 | no |
| **FeasibleSQA** | quantum-inspired (PIMC on F) | 1 | 0.72 | 1 / 1 / 0.943 | **0.981** [0.85] | 8.0 | **yes** |
| QAOA-seeded FeasibleSA | hybrid | 1 | 0.83 | 0.667 / 1 / 0.951 | 0.873 [0] | 8.1 | no |
| ConstrainedQAOA p=2 (8×8 only) | quantum (subspace sim) | 1 | 0.50 | 0.658 / – / – | 0.658 [0] | 2.0 | no |
| ConstrainedQAOA p=4 (8×8 only) | quantum (subspace sim) | 1 | 0.17 | 0.665 / – / – | 0.665 [0] | 2.2 | no |

**Verdict: FeasibleSQA is COMPARABLE on H by the pre-committed definition, but MILP is still better.**
- **Against the bar.** FeasibleSQA's family-mean benefit_true is 0.981, against a best classical of 1.000 (HiGHS). The threshold is 0.95, P(feasible) = 1, and no run went over time.
- **MILP is faster.** It reaches 1.000 on every instance, in 0.8 s on average against T\* = 8.4 s. FeasibleSQA reaches the certified optimum on 72% of runs.
- **Against FeasibleSA and Tabu-on-F.** FeasibleSQA is slightly above both (0.981 vs 0.973 and 0.976) and has the best worst case (0.85 vs 0.81 and 0.71). The whole difference comes from the 12×12 synergy cities (0.943 vs 0.920 and 0.951), where no heuristic reaches the optimum reliably within 20 s. That is 3 instances × 2 seeds, so the difference is within seed noise. The fair statement is that FeasibleSQA **ties** the feasible-space classical annealers; it does not beat them.

**The QAOA contestants and the hybrid do not qualify:**
- **ConstrainedQAOA** is only simulable on H 8×8 (|F| ≤ 2,484). There, building and diagonalising the |F|×|F| mixer alone takes longer than T\* ≈ 1.5 s. The |F| = 2,484 city runs over 1.5 T\* and scores 0, which is the pre-committed rule.
- **Constrained QAOA on the other two 8×8 cities** (|F| = 822 and 911) stays within T\*. Its best of 256 shots averages 0.988 with p=2 and 0.997 with p=4. QAOA-seeded FeasibleSA scores 1.000 there, with 75–85% of its wall-clock spent in QAOA.
- **QAOA-seeded FeasibleSA** fails for the same reason on the |F| = 2,484 city: 99% of its wall-clock goes to QAOA there. On 10×10 and 12×12 there is no quantum seed (|F| > 20k), and it is FeasibleSA seeded with greedy plus random feasible plans (0.951 on 12×12).
- **The E9 hybrid** (α = 0.01 + repair) is the best method on 10×10 and 12×12 (1.000). It fails on all three 8×8 cities (0.49 in total), because its greedy repair cannot restore feasibility when the budget is tight and equity binds. This limits the declared recipe, not the sampler.

**Family C (control).** All nine methods except the X-mixer control reach benefit 1.000 with P(opt) = 1, and so does the greedy planner (C is the submodular park case where greedy ≈ MILP). The X-mixer penalty QAOA scores 0.17 because 5 of its 6 runs exceed 1.5 T\* = 0.2 s. FeasibleSQA's H result is not a C-only win: on C every method is at 1.000.

**E13 (real-shaped 16×16 tile)** was skipped. T\* already reaches 20 s per run on the 12×12 synergy cities. A 16×16 tile would need several times that for each of 7 methods × 2 seeds, which is more than a short extra run.

## What is still not hardware
- **FeasibleSQA** is path-integral Monte Carlo on the feasible set: a classical sampler of a Trotterised transverse field whose slices are restricted to F. It is validated against the exact single-qubit ⟨σz⟩ (`tests/test_feasible_sqa.py`). It is not a model of an annealer, and no annealer implements a transverse field restricted to F.
- **ConstrainedQAOA** is an exact state-vector simulation on span(F), built from a dense |F|×|F| eigendecomposition. It is not a compiled circuit. The cost that makes it miss T\* on H 8×8 is a cost of the simulation, not of the circuit.
- **No CQM or hybrid-cloud sampler ran.** dimod is not installed and there is no API token. `quhi.solvers.cqm` builds a dependency-free constraint spec, tested against `ConstrainedBinaryProgram.is_feasible`, plus a `to_cqm` builder for when dimod exists. No D-Wave result is reported.
