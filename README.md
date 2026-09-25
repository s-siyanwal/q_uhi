# UHIP_Quantum
A repo for the research paper for quantum UHIP.

## `quhi`: a QUBO/HUBO simulation toolkit for urban heat island planning

`quhi` turns urban-heat-island (UHI) mitigation planning into **QUBO** and **HUBO** problems.
It then solves them with exact, classical, quantum-inspired and simulated quantum solvers,
and scores every result against a certified optimum and against the exact (non-polynomial) physics.

```
city grid ──► saturating cooling physics ──► constrained binary program ──► penalty QUBO / HUBO ──► solvers ──► plans, maps, metrics
 (50 m cells)   ΔT = ΔTmax[1-∏(1-p x)]        budget, one-option,          exact Λ, slack bits,        MILP · exhaustive · SA · tabu
                 expanded to order K=1,2,3     equity                       Rosenberg quadratisation     SQA (PIMC) · QAOA · exact annealing
```

This repository also contains the earlier design documents and code (`*.docx`, `*.md`,
`UHIP_Quantum-main.zip`). They were audited, and the audit found critical errors, including an
SQA sampler at the wrong temperature and results dominated by penalty artefacts.
See **[docs/AUDIT.md](docs/AUDIT.md)**. `quhi` is a clean re-implementation that does not import the legacy code.

## Install and test

```bash
pip install -e .[dev]          # numpy, scipy (HiGHS MILP), numba, matplotlib, pandas, pytest
pytest                         # 108 tests, ~20 s
python examples/quickstart.py  # end-to-end demo, ~1 min, writes examples/output/*.png
python scripts/run_experiments.py [--only E3] [--quick]   # regenerate results/
python scripts/audit_legacy.py                           # reproduce the audit of the legacy code
```

Optional: `pip install -e .[dwave]` adds `dimod`/`neal` interop (`BinaryPolynomial.to_dimod()`).
The legacy `requirements.txt` (qiskit, pyqubo, …) is only needed for the legacy code in the zip.

## Using it

```python
from quhi.uhi import generate_city, UHIPlanningProblem, DEFAULT_MIX
from quhi.solvers import MILPSolver, SimulatedQuantumAnnealing
from quhi.schedules import suggest_beta_range
from quhi.postprocess import postprocess

city = generate_city(12, 12, seed=0)                          # land use, baseline °C, population
prob = UHIPlanningProblem(city, interventions=DEFAULT_MIX,    # park / water / cool pavement
                          budget=18, min_per_district=1)
program = prob.program(order=2)            # 1 = additive, 2 = QUBO, 3 = HUBO (cubic saturation)
opt = MILPSolver().solve(program)          # certified constrained optimum (no penalties)

enc = program.to_penalty_model()           # provably exact penalty weight + slack bits
br = suggest_beta_range(enc.model, resolution=1e-3)   # resolve 0.001 °C at the cold end
ss = SimulatedQuantumAnnealing(num_sweeps=1000, beta_range=br).sample(enc.model, seed=0)
plans = postprocess(program, ss.samples)   # optional repair + feasible local search
print(prob.report(plans[0]))               # exposure temperature, cooling, spend, ...
```

Generic QUBO/HUBO use, with no UHI involved:

```python
from quhi import BinaryPolynomial, quadratize
f = BinaryPolynomial({(0,): -1, (0, 1): 2, (0, 1, 2): -3})   # a cubic HUBO
f.energies(X); f.to_spin()                                   # evaluate, map to spins
q = quadratize(f)                                            # exact QUBO with auxiliaries
Q, offset = q.qubo.to_qubo_matrix()                          # upper-triangular Q
```

## Package layout

| Module | Contents |
|---|---|
| `quhi.polynomial` | `BinaryPolynomial` (QUBO + HUBO, one canonical form), spin/Ising maps, dimod export, compiled CSR form |
| `quhi.constraints` | `ConstrainedBinaryProgram`, integer linear constraints, bounded-binary slack, **certified penalty weight** (corrected Theorem IV.1) |
| `quhi.quadratize` | Rosenberg substitution with a proven-sufficient penalty |
| `quhi.uhi` | Synthetic cities; saturating cooling surrogate; K-order truncations; least-squares fitted QUBO; greedy planner; reports |
| `quhi.solvers` | `ExhaustiveSolver`, `MILPSolver` (HiGHS), `SimulatedAnnealing`, `TabuSearch`, `SteepestDescent`, `SimulatedQuantumAnnealing` (PIMC, native HUBO), `QAOA` (state vector), `annealing_spectrum`, `schrodinger_anneal` |
| `quhi.postprocess`, `quhi.schedules` | Constraint repair / feasible local search; objective-resolution annealing schedules |
| `quhi.analysis` | Metrics (success probability, Wilson CI, TTS99, benefit ratio), benchmark harness, plots |

## Documentation
* [docs/BACKGROUND.md](docs/BACKGROUND.md): physics surrogate, literature sources, the corrected penalty theorem and quadratisation proofs, solver derivations, metrics.
* [docs/RESULTS.md](docs/RESULTS.md): initial results from experiments E1–E7, with figures in `results/`.
* [docs/AUDIT.md](docs/AUDIT.md): findings on the pre-existing documents and code.

## Caveats
City data are synthetic, and intervention parameters are literature-informed but illustrative.
SQA is a classical, quantum-*inspired* heuristic, and QAOA/annealing dynamics are simulated
exactly only for small n. Neither predicts hardware performance.
