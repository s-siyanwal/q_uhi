# Next slice: can quantum / quantum-inspired methods come at par with classical?

## Success criterion
A quantum or quantum-inspired method is "at par" only if, **at matched wall-clock on
certified instances**, its benefit ratio (exact physics) and P(opt) meet the **greedy planner
and HiGHS MILP** on **both** the park and the mix+equity families. Matching SA or
SA-matched is not enough. SQA is quantum-inspired PIMC, not hardware QA.

## Experiments
| ID | Question | Code |
|---|---|---|
| E8 | Do constraint-preserving mixers (Dicke/XY, per-cell XY, feasibility-projected moves) fix the penalty QAOA failure? | `quhi.solvers.mixers.ConstrainedQAOA` |
| — | Fair classical annealer: feasible-space SA (no penalty, no slack) | `quhi.solvers.feasible.FeasibleSA` |
| E8b | Is p ≤ 2 too shallow? Dicke-XY complete/ring at p = 1…6 on exactly-3 instances | `scripts/run_experiments.py --only E8b` |
| E9 | Weak penalty (α·Λ_safe) + repair as the declared **hybrid** recipe | `scripts/run_experiments.py --only E9` |
| E10 | Unbalanced / linear penalties vs quadratic slack: dynamic range and feasibility | `ConstrainedBinaryProgram.to_penalty_model(form=...)` |
| E11 | Native cubic HUBO vs Rosenberg QUBO: annealing gap | `annealing_spectrum` on both models |
| E12 (executed) | Comparable-performance bakeoff at matched wall-clock T* on a hard mix+equity family H and a park control C; FeasibleSQA, QAOA-seeded FeasibleSA, Tabu-on-F added | `scripts/run_experiments.py --only E12` |
| E12b (executed) | Ablation of E12: tight clock (FeasibleSA's own wall) and equal proposal count W\* | `--only E12b` |
| E13 (executed) | One LST-shaped 16×16 tile (hot core, scarce core lots, equity, tight budget) | `--only E13` |
| ANIM | GIFs of each method in its own search space and of the incumbent's temperature field | `quhi.analysis.animate`, `--only ANIM` |

Results: `results/E8_mixers`, `results/E8b_depth` … `results/E11_pubo_gap`, `results/ANIM`, `results/E12_comparable`, `results/E12b_ablation`, `results/E13_tile`; write-up in `docs/RESULTS.md`.

**Methods are frozen after E12b/E13; no E14 solver.** FeasibleSQA ties FeasibleSA at equal proposals and does not beat HiGHS; gate-model QAOA stays below the bar.

Legacy material (`UHIP_Quantum-main.zip`, root `*.docx`, `RESEARCH_EXECUTION_SUMMARY.md`,
the other root summaries) must not be used as a scientific source; see `docs/AUDIT.md`.
