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
   penalties also shrink the annealing gap in proportion to the weight (E6).
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

E6_TABLE_PLACEHOLDER

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
