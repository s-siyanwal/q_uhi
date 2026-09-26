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
| E9 | Weak penalty (α·Λ_safe) + repair as the declared **hybrid** recipe | `scripts/run_experiments.py --only E9` |
| E10 | Unbalanced / linear penalties vs quadratic slack: dynamic range and feasibility | `ConstrainedBinaryProgram.to_penalty_model(form=...)` |
| E11 | Native cubic HUBO vs Rosenberg QUBO: annealing gap | `annealing_spectrum` on both models |

Results: `results/E8_mixers` … `results/E11_pubo_gap`; write-up in `docs/RESULTS.md`.

Legacy material (`UHIP_Quantum-main.zip`, root `*.docx`, `RESEARCH_EXECUTION_SUMMARY.md`,
the other root summaries) must not be used as a scientific source; see `docs/AUDIT.md`.
