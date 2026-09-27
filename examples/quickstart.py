"""End-to-end quickstart: city -> QUBO/HUBO -> classical + quantum-inspired solvers -> maps.

    python examples/quickstart.py        # ~1 minute, writes examples/output/*.png
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quhi import quadratize
from quhi.analysis import build_instance, score
from quhi.analysis import plotting as qp
from quhi.postprocess import postprocess
from quhi.schedules import suggest_beta_range
from quhi.solvers import (QAOA, MILPSolver, SimulatedAnnealing, SimulatedQuantumAnnealing,
                          TabuSearch)
from quhi.uhi import DEFAULT_MIX, UHIPlanningProblem, generate_city

out = Path(__file__).parent / "output"
out.mkdir(exist_ok=True)

# 1. A synthetic 12x12 city of 50 m cells (land use, baseline temperature, population)
city = generate_city(12, 12, seed=0, candidate_fraction=0.3)
print(city.summary())
qp.plot_city(city, out / "city.png")

# 2. Planning problem: parks / water / cool pavement, budget, >= 1 intervention per district
prob = UHIPlanningProblem(city, interventions=DEFAULT_MIX, budget=18, min_per_district=1)
program = prob.program(order=2)                   # order=2 -> QUBO, order=3 -> HUBO
print("objective:", program.objective, "| constraints:", [c.name for c in program.constraints])

# 3. Ground truth: MILP solves the constrained program exactly (no penalties)
milp = MILPSolver().solve(program)
print("MILP optimum:", milp.best_energy, "°C  optimal =", milp.info["optimal"])

# 4. Penalty encoding (certified-exact weight), then heuristic and quantum-inspired sampling
inst = build_instance("quickstart", program, penalty_mult=0.1, f_opt=milp.best_energy)
print("QUBO:", inst.model, "| penalty weight", round(inst.encoding.penalty_weight, 3),
      "(safe:", round(inst.meta["safe_penalty_weight"], 3), ")")
br = suggest_beta_range(inst.model, resolution=1e-3)       # resolve 0.001 °C at the cold end
plans = {"MILP (optimal)": milp.best}
for name, solver in {"SA": SimulatedAnnealing(num_sweeps=1500, num_reads=32, beta_range=br),
                     "SQA": SimulatedQuantumAnnealing(num_sweeps=1500, num_reads=16, beta_range=br,
                                                      gamma_range=(0.3, 1e-6)),
                     "Tabu": TabuSearch(max_iter=4000, num_reads=32)}.items():
    ss = solver.sample(inst.model, seed=0)
    m = score(inst, ss, postprocess=True)
    print(f"{name:5s} raw P(opt)={m['p_success']:.2f} P(feas)={m['p_feasible']:.2f} | "
          f"after post-processing P(opt)={m['p_success_pp']:.2f} | {ss.wall_time:.1f}s")
    X = postprocess(program, ss.samples)
    f = np.where(program.is_feasible(X), program.objective_values(X), np.inf)
    plans[f"{name} + pp"] = X[int(np.argmin(f))]

plans["Greedy planner"] = prob.greedy_plan()
for k, x in plans.items():
    r = prob.report(x)
    print(f"{k:16s} exposure {r['exposure_temp_C']:.3f} °C (baseline {r['exposure_temp_baseline_C']:.3f}),"
          f" spend {r['spend']}/{r['budget']}, feasible={program.is_feasible(x)[0]}")
qp.plot_plan_transition(prob, {k: plans[k] for k in ("MILP (optimal)", "SQA + pp", "Greedy planner")},
                        out / "transition.png", "Baseline -> optimised plans")

# 5. HUBO: cubic saturation terms, solved natively vs after quadratisation
hubo = UHIPlanningProblem(city, budget=18, prune_tol=1e-3).program(order=3)
enc = hubo.to_penalty_model()
q = quadratize(enc.model)
print("HUBO:", enc.model, "-> quadratised:", q.qubo, f"({len(q.aux)} auxiliaries)")

# 6. Tiny instance on a simulated gate-model device (QAOA state vector)
tiny = UHIPlanningProblem(generate_city(6, 6, seed=4, candidate_fraction=0.4), budget=6).program(2)
qa = QAOA(p=3, shots=512).sample(tiny.to_penalty_model().model, seed=0)
print(f"QAOA p=3 on {qa.samples.shape[1]} qubits: P(ground) = {qa.info['p_ground']:.3f} "
      f"(uniform {qa.info['p_ground_uniform']:.3f})")
print("figures written to", out)
