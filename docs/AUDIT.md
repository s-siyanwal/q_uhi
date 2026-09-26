# Audit of the existing UHI-QUBO material

Scope: every file in this repository before `quhi/` was added. That is the six
`*.md` summaries, the 14 `.docx` design/maths documents, `UHIPs.ipynb`, the PDF,
and the code in `UHIP_Quantum-main.zip` (~20k lines: `core/`, `validation/`,
`scripts/`, `notebooks/`). The zip's copies of the root files are identical duplicates.

Every numbered finding can be reproduced. `python scripts/audit_legacy.py`
imports the **unmodified** legacy code from the zip and writes the evidence to
`results/audit/audit.json`.

## What the material is

| Layer | Content | Assessment |
|---|---|---|
| `UHI_QUBO_maths3.docx` | Main theory: UHI problems from the literature (Sanya green-space GA, cool-pavement GWP), QUBO/Ising definitions, penalty theorem IV.1, resource counts, TFIM, Suzuki–Trotter SQA | Sound in outline, but has the errors listed below |
| `UHI_QUBO_maths2.docx` | Classical baselines: MILP/McCormick, B&B, local search, tabu, SA, GA | Correct and standard |
| `UHI_QUBO_maths.docx` + `File-0x` / `Part-x` docx | AI-generated code plans for each module | Several formulas and "known optima" are wrong (see below) |
| `UHIP_Quantum-main.zip` | Implementation of those plans, with tests, notebooks and "research results" | Runs, but **19 of 163 tests fail**, and the results are invalid (see F1–F3) |
| `UHIPs.ipynb` | Continuous tropical-island green-space model (differential evolution) | Useful physical intuition: dense core, coastal cooling, kernels |

## Findings

Severity: **critical** means the results are wrong; **major** means a theorem or
algorithm is wrong; **minor** means inconsistency or dead code.

### F1 (critical): inequality constraints are encoded as equalities, in raw m²
`core/qubo_formulation/constraint_penalty.py:65-89` implements `Σ a_i x_i ≤ A_max`
as `P·(Σ a_i x_i − A_max)²`. That is an *equality* penalty: it punishes plans
that use less than the budget, and it is never zero unless the budget is hit
exactly. Areas are in m² (`a_i = 2500`), so the penalty coefficients are about
P·2500² while the cooling coefficients are about 2 °C.

* Coefficient ratio in the legacy `UHIGreenLayoutProblem` (4×4 grid): **|Q|max = 1.72×10⁹
  against |Q_physical|max = 2.0**, a ratio of 8.6×10⁸. At double precision the physics is
  numerical noise next to the penalty.
* When the budget is not a multiple of the cell area (area fraction 0.33 gives 3.63 cells),
  the legacy QUBO's global minimum selects **4 cells = 10,000 m² > 9,075 m², which is infeasible**.
  (4 − 3.63)² < (3 − 3.63)². The true constrained optimum uses 3 cells.
* **Fix in `quhi`:** integer constraint data (validated), slack bits with a
  *bounded* binary encoding (every slack value in [0, b − min a·x] is representable,
  none larger), and `≥` constraints with negative slack.

### F2 (critical): the shipped "research results" are artefacts
`notebooks/research_results/research_results.json` reports QUBO energies of
**−6.89×10¹⁰** for SA, Tabu and GA; these are penalty magnitudes, not temperatures.
It reports **SQA energy 0.0 with an all-zero plan** (no green space), yet
names "GA" the best algorithm on differences in the 8th significant digit.
Nothing in these results says anything about UHI mitigation or about
quantum versus classical methods.

### F3 (critical): SQA simulates the wrong temperature
`core/solvers/quantum/trotter_expansion.py:202-262` and `metropolis_sampler.py`
use the effective energy `Σ_k E(s^k) − J_τ Σ s s` with acceptance `exp(−β ΔE)`.
The Suzuki–Trotter action has **β/P** on the problem term of every slice (the
same error appears in `UHI_QUBO_maths3.docx` §VII.1, whose `Z ≈ Σ exp(−β H_eff)` drops the 1/M).
Tested against the exact single-qubit thermal expectation
`⟨σz⟩ = −h/E·tanh(βE)` (h=0.5, Γ=0.7, β=2, P=16):

| | ⟨σz⟩ |
|---|---|
| exact quantum | −0.545 |
| `quhi` PIMC | −0.546 |
| legacy action | **−0.997** (the exact value at P·β with Γ/P, i.e. a nearly classical, P× colder system) |

The legacy sampler is P times too cold and its transverse field is effectively
P times too weak. It is essentially classical annealing with a lot of overhead, which
probably explains the frozen all-zero SQA output in F2. `quhi` also runs this check
as a unit test (`tests/test_solvers.py::test_pimc_matches_exact_single_qubit`).

### F4 (major): Theorem IV.1 (penalty exactness) is false as stated
The claim is that Λ > f_max − f_min makes every minimiser feasible "because the penalty is ≥ Λ at
infeasible points". That needs every violation to satisfy g(x)² ≥ 1, which only
holds for **integer** constraint data. Counter-example: minimise −x₀ − x₁ subject to
0.6x₀ + 0.6x₁ ≤ 1 with Λ = 2.02 > range. The penalised minimiser is (1,1), which is
infeasible because the violation² is 0.04. The theorem also uses `max(0,h)²`, which is
not a polynomial and so not a QUBO; the maths text says the squares are "quadratic
polynomials", which is wrong for the `max`.

**Corrected statement (proved in `docs/BACKGROUND.md` §2):** with integer data and
slack bits, Λ > f* − L(f) suffices, where f* is the constrained optimum (bounded
above by any feasible point) and L(f) is any lower bound on f. This is usually
**about 2× smaller** than the range bound (median ratio 2.06 across the E1 instances),
and experiment E1 certifies it exactly on 11 instances.

### F5 (major): the automatic penalty weight is not a bound
`ConstraintPenaltyEncoder.calculate_sufficient_penalty_weight` estimates the range of
xᵀQx as Σ|λᵢ(Q)|. Over {0,1}ⁿ one only has xᵀQx ≤ ‖x‖²·Σλ⁺ with ‖x‖² ≤ n, so the
estimate can be too small by a factor of n. Example: Q = all-ones (n = 10) gives a
legacy weight of 10 while max xᵀQx = 100 (audit item A7).

### F6 (major): the QUBO↔Ising round trip is broken
`core/qubo_formulation/ising_converter.py:230` builds off-diagonals as `−4J_ij` on
both triangles of a symmetric Q, which doubles every coupling. `qubo_to_ising` is
correct, but `ising_to_qubo(qubo_to_ising(Q)) ≠ Q`:
`[[1,-2,0],[-2,3,-1],[0,-1,2]]` comes back as `[[1,-4,0],[-4,3,-2],[0,-2,2]]`
(also caught by the legacy suite's own failing test `test_ising_to_qubo`). The root cause
is that the code base mixes the symmetric and upper-triangular conventions for Q.
**Fix:** `quhi` uses one canonical sparse polynomial and exports upper-triangular Q only.

### F7 (major): worked examples and "known optima" are wrong
* `UHI_QUBO_maths3.docx` Example IV.1 prints a *symmetric* Q with the full pair coefficient
  (3.9) in both triangles. Read as xᵀQx, as the document defines QUBO, it doubles
  the couplings: its minimiser is `[1,0,0,0]`, one cell, although the constraint asks for
  exactly two. Read as upper-triangular, it matches the intended penalised model,
  whose optimum is `[1,1,0,0]` with f = −0.8.
* Theorem II.3's proof sets `Q_ii = c_i/2`. It must be `Q_ii = c_i`, since x_i² = x_i.
* The research plan's 4-cell benchmark (α = [−3,−2,−3,−2], β₀₁ = β₂₃ = 0.5) claims
  optimum `[0,1,1,0]` with energy −19.25. In fact f([0,1,1,0]) = −5, while `[1,0,1,0]`
  gives −6, and the −19.25 value corresponds to neither. Tests built on such "ground truths" certify nothing.

### F8 (minor/major): other issues
* **Sign of the pairwise physics.** The plan uses β_ij < 0 (synergy) in one place and
  says "adjacent greens have diminishing returns" (β > 0) in another. `quhi` derives the sign
  from a saturation law: pairwise terms are positive (diminishing returns) and cubic terms negative.
* **Legacy tests.** 19 fail / 141 pass / 3 skipped (qiskit-dependent tests excluded, because
  qiskit is not installed): GA API mismatches, `SolutionResult.__repr__`, the
  Ising round trip, energy consistency and others. Several passing tests only check
  shapes or run without asserting correctness against an independent oracle.
* **QAOA/VQE modules** require qiskit, and are memory-stress tested rather than
  checked for correctness.
* **Reproducibility.** Results embed timestamps but no seeds or environment
  details. The zip contains `__pycache__` and `.coverage` artefacts.
* **Scaling claims.** The "expected quantum advantage" section cites asymptotic
  QA-vs-SA residual-energy results for 1D disordered chains (Santoro et al. 2002). These do not
  transfer to dense, penalty-dominated QUBOs. Experiment E6 shows that, with the certified
  penalty, the objective's level spacing is 7×10⁻⁶ of the spectrum width (14 qubits), and that
  this gap shrinks as 1/Λ. That is the opposite of what analog hardware needs.

## What was kept

* **Problem framing:** the Sanya-style 50 m grid with candidate cells (Zhou et al., 2023),
  green/blue/cool-material interventions, budget and equity constraints, and exposure weighting.
* **The solver line-up** from `maths2.docx` (exact MILP, tabu, SA) plus SQA, now implemented
  with correct physics and checked against oracles.
* **The idea of HUBO terms from higher-order interactions.** `quhi` derives them rather
  than postulating them (inclusion–exclusion expansion of cooling saturation; park-size synergy).

The legacy code is left untouched in `UHIP_Quantum-main.zip` for provenance; nothing
in `quhi/` imports it.
