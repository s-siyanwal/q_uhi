# Constraint encodings, not annealers, dominate QUBO/HUBO siting for urban heat islands

*Paper skeleton. Every number below is copied from `docs/RESULTS.md`, `docs/AUDIT.md`,
`results/*/summary.csv` or `results/*/e*.md`; numbers not found there are omitted. Methods
are frozen after E12b/E13.*

**PDFs.** LaTeX sources are in `latex/` (see `latex/README.md`). The compiled IEEE conference paper, Springer LNCS paper and full report are `results/papers/ieee_quhi.pdf`, `results/papers/springer_quhi.pdf` and `results/papers/quhi_full_report.pdf`. All three use the same facts as this skeleton.

## Abstract
We site parks, water and cool pavement on synthetic 50 m city grids. Each instance is a
constrained binary program, encoded as a QUBO or HUBO and scored on exact cooling physics
against a HiGHS-certified optimum.
- A slack-bit penalty QUBO with a corrected weight is exact on all 11 enumerated instances (E1).
- The second-order Taylor QUBO loses 10.3% of the achievable cooling at 300 m reach, where the
  cubic HUBO loses ≤ 0.02% (E2).
- The penalty sets the final annealing gap: Δ(s=1) ∝ 1/Λ (E6).
- Constraint-preserving Dicke-XY QAOA reaches P(opt) 0.43 at p = 6, against 1.00 for
  feasible-space SA (E8b).
- On a hard mix-plus-equity family, path-integral Monte Carlo restricted to the feasible set
  meets a pre-committed 0.95-of-best-classical bar (benefit 0.981, E12). It is a
  Trotter-replica annealer that made 3.6× more proposals than feasible-space SA. At equal proposals it scores 0.983 against 0.988 (E12b).
- HiGHS certifies every family-H and E13 instance in ≤ 2.3 s.

Encodings and constraint handling, not the choice of annealer, decide the outcome. No quantum
or quantum-inspired advantage is observed.

## 1 Problem and success criterion
- **Physics.** Cooling saturates, ΔT = ΔT_max[1 − ∏(1 − p x)]. Expanded to order K = 1, 2 or 3
  it gives an additive model, a QUBO or a HUBO. A 0.3 °C park-block synergy adds quartic terms.
- **Constraints.** An integer budget, at most one option per cell, and optionally at least one
  intervention per district (equity). See `docs/BACKGROUND.md` §1–2.
- **Scoring.** benefit_true is the exact-physics cooling of a returned plan divided by that of
  the MILP plan. Infeasible plans score 0.
- **Success criterion (`docs/NEXT.md`).** A quantum or quantum-inspired method is "at par" only
  if, at matched wall-clock on certified instances, its benefit and P(opt) meet the **greedy
  planner and HiGHS MILP** on **both** the park family and the mix+equity family. Matching SA is
  not enough.

## 2 Legacy audit (F1–F7)
Reproduced by `python scripts/audit_legacy.py` → `results/audit/audit.json`; details in
`docs/AUDIT.md`. Energies from the legacy results are not cited here; they are penalty artefacts
(F2).

| | finding | evidence |
|---|---|---|
| F1 | Inequalities encoded as equalities, in raw m² | coefficient ratio 1.72×10⁹ vs 2.0; the global minimum is infeasible when the budget is not a multiple of the cell area |
| F2 | Shipped "research results" are artefacts | QUBO energies are penalty magnitudes; SQA returns the all-zero plan |
| F3 | SQA uses β instead of β/P | single-qubit ⟨σz⟩: exact −0.545, `quhi` PIMC −0.546, legacy −0.997 |
| F4 | Penalty theorem IV.1 false for non-integer data | counter-example in `docs/AUDIT.md`; the corrected bound is about 2× smaller (median ratio 2.06) |
| F5 | Automatic penalty weight is not a bound | can be too small by a factor of n |
| F6 | QUBO↔Ising round trip doubles couplings | `ising_to_qubo(qubo_to_ising(Q)) ≠ Q` |
| F7 | Worked examples and "known optima" are wrong | e.g. a claimed optimum of −19.25 where f = −5 and the true optimum is −6 |

The legacy test suite has 19 failures out of 163 tests. Legacy files are provenance only; see
`docs/LEGACY.md`.

## 3 Encodings (E1, E4, E10, E11)
- **E1 (exactness).** On all 11 instances, the ground state of the penalty QUBO (decision plus
  bounded slack bits, corrected Λ_safe) equals the MILP optimum. Λ_safe is 1.81–2.21× smaller
  than the range bound.
- **E4 (weak penalty).** On mix 8×8 + equity, 0.01·Λ_safe followed by repair raises P(opt)
  after post-processing from ≤ 0.03 to 0.42 (SA) and 0.47 (Tabu). Raw feasibility collapses
  below about 0.1·Λ_safe.
- **E10 (penalty form; quick run, 2 instances).** An unbalanced penalty at 0.1·Λ_safe cuts
  dynamic range from 163 to 16.5 while keeping the ground state optimal. This is not
  certified in general. The linear (Lagrangian) form is feasible and optimal only for lucky
  multipliers.
- **E11 (quadratisation; quick run, 2 instances).** Rosenberg quadratisation of a cubic HUBO
  leaves the slack-penalty final gap unchanged (9.4e-5 / 1.1e-5). It adds 7 qubits (9 → 16) and
  cuts short-anneal P(ground) from 3.1e-3 to 2.4e-5.

## 4 Surrogates (E2, E5)
**E2.** Mean share of the achievable cooling lost when a surrogate's optimum is scored on the
exact physics (12×12, 4 seeds per decay length L):

| L (m) | additive K=1 | QUBO K=2 (Taylor) | QUBO (least-squares fit) | HUBO K=3 |
|---|---|---|---|---|
| 100 | 0.00% | 0.20% | 0.02% | 0.00% |
| 200 | 0.29% | 2.81% | 0.21% | 0.02% |
| 300 | 0.00% | **10.30%** | 0.42% | 0.00% |

**E5.** On a quartic HUBO with 30 variables, Tabu finds the certified optimum on 99.0% of raw
reads. After Rosenberg quadratisation (187–197 variables) it finds it on 0%. HUBOs should be
solved natively.

## 5 Classical bar (E3, E7, E9, E12 MILP times)
- **Greedy.** The objective is submodular. The cost-benefit greedy planner is optimal on 8 of 15
  park instances (worst gap 0.037 °C). On mix+equity it is infeasible on 5 of 9 instances in E3
  and 6 of 9 family-H instances in E12, because it ignores equity.
- **Tabu on the safe-Λ QUBO** solves park instances up to 27 variables on every read (P(opt)
  1.00 in E3). On mix+equity, no penalty-QUBO sampler reaches the optimum on raw reads (E3).
- **E7 showcase** (14×14, 81 decision variables). MILP certifies in 10.8 s. The greedy planner
  returns the same plan (exposure temperature 31.141 °C against 32.311 °C at baseline).
- **HiGHS wall-clock:**

  | study | instances | HiGHS wall |
  |---|---|---|
  | E9 (mix+equity 8–12²) | 9 | 3.8 s mean |
  | E12 family H | 9 | 0.81 s mean, 2.29 s max |
  | E12b family H | 9 | 0.77 s mean |
  | E12b family C | 3 | 0.013 s mean |
  | E13 tile | 1 | 1.67 s |

- **Where HiGHS slows down.** Certification becomes slow above about 60 dense decision
  variables: a 20×20 pilot was not certified in 120 s.

## 6 Quantum-inspired and QAOA (E6, E8, E8b, E12 QAOA rows)
**E6 (14 qubits, exact spectra).** The final gap scales as 1/Λ: Δ(s=1) = 7.1e-5 at 0.1·Λ_safe
and 7.1e-7 at 10·Λ_safe, so 1/Δ² goes from 2.0e8 to 2.0e12. X-mixer QAOA at p = 6 puts
0.17–0.19% of its probability on the optimum. Its approximation ratio of ≈ 0.998 is measured on
a penalty-dominated scale and is misleading.

**E8 (constraint-preserving mixers, p ≤ 2).**
- **Against penalty QAOA.** P(feasible) = 1 by construction, and P(opt) is 10–70× higher than
  X-mixer penalty QAOA (exactly-3, p = 2: Dicke-XY 0.37 vs X-mixer 0.005).
- **Against classical, on mix+equity.** XY-QAOA p = 2 reaches P(opt) 0.003 and benefit 0.61,
  against FeasibleSA at 0.39 and 0.92.

**E8b (Dicke-XY depth curve; `results/E8b_depth/summary.csv`).** Three exactly-3 park cities
(n = 10, |F| = 120). P(opt) is the mean [min–max] over the 3 cities.

| method | p=1 | p=2 | p=3 | p=4 | p=6 |
|---|---|---|---|---|---|
| Dicke-XY complete | 0.134 [0.05–0.18] | 0.242 [0.08–0.37] | 0.309 [0.11–0.47] | 0.376 [0.13–0.56] | **0.427** [0.17–0.65] |
| Dicke-XY ring | 0.034 | 0.067 | 0.057 | 0.066 | 0.111 [0.02–0.26] |
| X-mixer penalty QAOA | 0.004 | 0.005 | 0.006 | 0.008 | 0.009 |

References on the same cities:
- Uniform on F: 1/120 = 0.0083.
- FeasibleSA: 1.000 in 0.21 s.
- Greedy planner: 0.67.
- MILP: 1, in 0.32 s.

Expected benefit for Dicke-XY complete rises from 0.955 (p = 1) to 0.993 (p = 6).

**E12 QAOA rows (family H, 8×8 only, |F| ≤ 2,484).**
- **ConstrainedQAOA:** benefit 0.658 at p = 2 and 0.665 at p = 4. The dense |F|×|F| subspace
  simulation exceeds T\* on the |F| = 2,484 city.
- **QAOA-seeded FeasibleSA:** 0.873 over all of family H.

## 7 Feasible-space PIMC (E12, E12b)
**Method.** FeasibleSQA is PIMC with P = 8 Trotter slices, where every slice is a feasible plan.
Its moves are add/remove and swap. The problem term carries β/P and the replica coupling is
J = ½ ln coth(βΓ/P). It is validated against the exact single-qubit ⟨σz⟩.

**E12 (family H: 9 instances × 2 seeds, matched wall-clock T\*).** The pre-committed rule is
benefit ≥ 0.95 × best classical, P(feas) = 1 and no run over time. FeasibleSQA passes it, but so
do the two feasible-space classical heuristics.

| method | P(feas) | P(opt) | benefit mean [min] | comparable |
|---|---|---|---|---|
| MILP (HiGHS) | 1 | 1 | 1.000 | bar |
| FeasibleSA | 1 | 0.78 | 0.973 [0.81] | yes |
| Tabu-on-F | 1 | 0.78 | 0.976 [0.71] | yes |
| FeasibleSQA | 1 | 0.72 | 0.981 [0.85] | yes |
| Greedy planner | 0.33 | 0.33 | 0.333 [0] | no |

**E12b ablation (`results/E12b_ablation/summary.csv`).**
- **The clock was not padded.** HiGHS was faster than default FeasibleSA on every instance, so
  E12's T\* was FeasibleSA's own wall-clock, and the tight clock reproduces E12.
- **The difference is work.** In the same time, FeasibleSQA made **3.6× more proposals**
  (4.66M vs 1.30M per run), because moves that would leave F are rejected before the objective
  is evaluated.

Equal-work rows, family H, W\* = 4.53M proposals per run:

| method | clock | P(feas) | P(opt) | benefit mean [min] | wall s | proposals | ≥ 0.95 FSA | beats FSA | w/t/l vs FSA |
|---|---|---|---|---|---|---|---|---|---|
| FeasibleSA | equal_work | 1 | 0.89 | **0.988** [0.85] | 25.0 | 4.53M | – | – | – |
| FeasibleSQA | equal_work | 1 | 0.78 | 0.983 [0.81] | 7.1 | 4.53M | yes | no | 1/6/2 |
| Tabu-on-F | equal_work | 1 | 0.83 | 0.977 [0.71] | 11.3 | 4.65M | yes | no | 1/6/2 |
| MILP (HiGHS) | – | 1 | 1 | 1.000 | 0.77 | – | – | – | – |

**Reading.** FeasibleSQA is a feasible-space annealer that ties FeasibleSA. It is within 5% at
equal proposals and slightly behind. Its E12 lead was an extra-work artefact. It is not a QPU
result and it does not beat HiGHS.

## 8 E13: an LST-shaped tile (no classical gap)
`results/E13_tile/`: one 16×16 tile.
- **City:** a compact hot core with a cool edge, population on the core, and only 2 candidate
  lots left in the core.
- **Problem:** DEFAULT_MIX, equity, budget 16. That gives n = 108 and |F| > 4,000, so
  ConstrainedQAOA is not run.
- **Clock:** tight T\* = 68.59 s.

Means over two sampler seeds (`results/E13_tile/e13.md`):

| method | P(feas) | P(opt) | benefit mean | wall s |
|---|---|---|---|---|
| MILP (HiGHS, 8 s cap) | 1 | 1 | 1.0000 | 1.67 |
| Greedy planner | 0 | 0 | 0 | 0.06 |
| FeasibleSA | 1 | 0.5 | 0.9569 | 66.56 |
| Tabu-on-F | 1 | 0 | 0.6412 | 67.35 |
| FeasibleSQA | 1 | 0.5 | 0.9998 | 63.72 |
| SA-matched α=0.01 + repair | 1 | 1 | 1.0000 | 64.68 |

- **HiGHS certifies the tile in 1.67 s.** The greedy planner is infeasible because it ignores
  equity, the same failure as on family H. The mask created no classical gap.
- **FeasibleSQA vs FeasibleSA.** FeasibleSQA's higher mean comes from the tight clock with 2
  seeds. By E12b it therefore rests on extra proposals, not on better sampling.

## 9 What hardware could still test
- **What would be fair.** Submit the constrained model natively, as a CQM to Leap's hybrid
  solver, so that constraints are not turned into penalties.
- **What must never be submitted.** A safe-Λ slack QUBO must never be sent to a QPU: E6 and E11
  show that its gap is set by Λ.
- **The skip-hook.** `quhi.solvers.cqm.sample_cqm` does nothing unless `DWAVE_API_TOKEN` is set
  and dwave-system is installed (`pip install -e .[dwave]`). It refuses objectives of degree
  > 2 and re-scores returned plans with the constrained program. No D-Wave result exists in this
  repository.

To run (not run here):
```bash
DWAVE_API_TOKEN=... python -c "import sys; sys.path.insert(0, 'scripts'); import run_experiments as r; \
from quhi.solvers.cqm import sample_cqm; _, _, cbp, _ = r._cardinality_twin(4); print(sample_cqm(cbp))"
```

Target instances, if someone later runs it:
- **E8b exactly-3, seed 4** (`_cardinality_twin(4)`, n = 10).
- **Family C, seed 100** (`UHIPlanningProblem(generate_city(8, 8, seed=100, candidate_fraction=0.3)).program(2)`).

Forbidden: the E12 12×12 slack QUBO. Its quartic objective is also refused by `sample_cqm`.

## 10 Limitations
- **Synthetic data.** The cities (and the E13 tile) are synthetic. No Landsat LST or census data
  were used, and the intervention amplitudes and decay lengths are illustrative, not calibrated.
- **Surrogate, not simulation.** The saturation law is a surrogate for CFD or ENVI-met, and MILP
  certifies the K = 2 surrogate, not the exact physics. This is why benefit can slightly
  exceed 1.
- **PIMC ≠ QPU.** FeasibleSQA and SQA are classical path-integral Monte Carlo samplers. No
  annealer implements a transverse field restricted to F.
- **ConstrainedQAOA = subspace simulation.** It is an exact state-vector simulation on span(F),
  built from a dense eigendecomposition, not a compiled circuit. Its T\* misses are simulation
  costs.
- **Small samples.** There are 3 instances × 2 seeds per E12/E12b cell, and E10 and E11 are
  quick runs on 2 instances.

## Figures
- **Plans and cooling:** `results/E7_showcase/transition.png`, `results/E2_fidelity/fidelity.png`.
- **Depth and bakeoff:** `results/E8b_depth/p_opt_vs_p.png`, `results/E12_comparable/benefit_vs_time.png`.
- **E13 tile:** `results/E13_tile/city.png` and `results/E13_tile/plans.png`.
- **Animations, showcase A (exactly-3):** `results/ANIM/compare_temp_A_park_7x7_s4_exactly3.gif`,
  `results/ANIM/search_dicke_xy_complete_p6_A_park_7x7_s4_exactly3.gif`,
  `results/ANIM/search_xmixer_penalty_qaoa_p6_A_park_7x7_s4_exactly3.gif`.
- **Animations, showcase B (mix 10×10):** `results/ANIM/compare_temp_B_mix_10x10_s0.gif`,
  `results/ANIM/search_feasiblesa_B_mix_10x10_s0.gif`.
- **Stills:** `results/ANIM/compare_final.png` and `results/ANIM/compare_final_B_mix_10x10_s0.png`.
