"""Reproduce every result in docs/RESULTS.md.

    python scripts/run_experiments.py              # all experiments (~30-60 min on 4 cores)
    python scripts/run_experiments.py --only E3    # one experiment
    python scripts/run_experiments.py --quick      # smoke run with tiny budgets

Outputs go to results/<experiment>/ as CSV (raw rows), JSON (summaries) and PNG.
Every random choice is seeded; the environment is recorded in results/env.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quhi import quadratize  # noqa: E402
from quhi.analysis import build_instance, run  # noqa: E402
from quhi.analysis import plotting as qp  # noqa: E402
from quhi.postprocess import postprocess  # noqa: E402
from quhi.schedules import suggest_beta_range  # noqa: E402
from quhi import BinaryPolynomial, ConstrainedBinaryProgram, LinearConstraint  # noqa: E402
from quhi.solvers import (QAOA, ConstrainedQAOA, FeasibleSA, enumerate_feasible,  # noqa: E402
                          all_energies)
from quhi.solvers.quantum import qaoa_state  # noqa: E402
from quhi.solvers import (QAOA, ExhaustiveSolver, MILPSolver, RandomSampler,  # noqa: E402
                          SimulatedAnnealing, SimulatedQuantumAnnealing, SteepestDescent,
                          TabuSearch, annealing_spectrum, schrodinger_anneal)
from quhi.uhi import DEFAULT_MIX, PARK, UHIPlanningProblem, generate_city  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "results"
RES = 1e-3          # objective resolution (°C) for problem-aware schedules
QUICK = False


def out(name: str) -> Path:
    p = ROOT / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(df: pd.DataFrame, d: Path, name: str):
    df.to_csv(d / f"{name}.csv", index=False)


def dump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def solver_suite(model, sweeps=1000, reads=32, P=16, with_fixed=True, with_matched=True):
    br = suggest_beta_range(model, RES)
    s = {
        "SA": SimulatedAnnealing(num_sweeps=sweeps, num_reads=reads, beta_range=br),
        "Tabu": TabuSearch(max_iter=max(2000, 40 * model.n), num_reads=reads),
        "SQA": SimulatedQuantumAnnealing(num_sweeps=sweeps, num_reads=reads, trotter_slices=P,
                                         beta_range=br, gamma_range=(0.3, 1e-6)),
        "Greedy": SteepestDescent(num_reads=reads),
        "Random": RandomSampler(num_reads=reads),
    }
    if with_fixed:
        s["SQA-fixedT"] = SimulatedQuantumAnnealing(num_sweeps=sweeps, num_reads=reads, trotter_slices=P,
                                                    mode="fixed", pt=0.05, gamma_range=(0.3, 1e-6))
    if with_matched:   # SA given the same number of spin-flip attempts as SQA (P+1 passes/sweep)
        s["SA-matched"] = SimulatedAnnealing(num_sweeps=sweeps * (P + 1), num_reads=reads, beta_range=br)
    return s


def summarise(df: pd.DataFrame, by=("solver",), cols=None) -> pd.DataFrame:
    cols = cols or ["p_success", "p_feasible", "mean_benefit_ratio", "best_gap_C", "tts99_s",
                    "time_per_read_s"]
    cols = [c for c in cols if c in df.columns]
    g = df.groupby(list(by))[cols]
    return g.median().join(g.quantile(0.25).add_suffix("_q25")).join(g.quantile(0.75).add_suffix("_q75"))


# ----------------------------------------------------------------- E1
def e1_encoding():
    """Certify the encoding: ground state of the safe-penalty QUBO == MILP constrained optimum."""
    d = out("E1_encoding")
    rows = []
    cfgs = [(6, s, (PARK,), 0) for s in range(4)] + [(8, s, (PARK,), 0) for s in range(4)] + \
           [(6, s, DEFAULT_MIX, 0) for s in range(2)] + [(8, s, (PARK,), 1) for s in range(2)]
    for R, seed, ivs, eq in cfgs:
        city = generate_city(R, R, seed=seed, candidate_fraction=0.4)
        prob = UHIPlanningProblem(city, interventions=ivs,
                                  budget=6 if len(ivs) > 1 else None, min_per_district=eq)
        cbp = prob.program(2)
        enc = cbp.to_penalty_model()
        if enc.model.n > 26:
            continue
        try:
            ms = MILPSolver().solve(cbp)
        except RuntimeError as e:          # e.g. equity demands exceed the budget
            print(f"skip {R}x{R}_s{seed}: {e}")
            continue
        ex = ExhaustiveSolver().sample(enc.model)
        x = ex.best[: cbp.n]
        L, U = cbp.objective.coefficient_bounds()
        rows.append({
            "instance": f"{R}x{R}_s{seed}_{'mix' if len(ivs) > 1 else 'park'}_eq{eq}",
            "n_decision": cbp.n, "n_slack": enc.model.n - cbp.n, "n_total": enc.model.n,
            "milp_opt": ms.best_energy, "qubo_ground_obj": float(cbp.objective_values(x)[0]),
            "qubo_ground_feasible": bool(cbp.is_feasible(x)[0]),
            "ground_degeneracy": ex.info["n_degenerate"],
            "safe_penalty": enc.penalty_weight, "range_penalty": U - L,
            "range_over_safe": (U - L) / enc.penalty_weight,
        })
    df = pd.DataFrame(rows)
    df["match"] = np.isclose(df.milp_opt, df.qubo_ground_obj) & df.qubo_ground_feasible
    save(df, d, "encoding_certification")
    dump({"all_match": bool(df.match.all()), "n_instances": len(df),
          "median_range_over_safe": float(df.range_over_safe.median())}, d / "summary.json")
    print(df.to_string())


# ----------------------------------------------------------------- E2
def _true_optimum(prob, cbp, kmax, score=None):
    """Exact minimiser of ``score`` (default: the untruncated saturating objective) over all
    feasible plans with at most ``kmax`` interventions (enumeration)."""
    score = score or prob.true_objective
    n = prob.n
    best, bx = np.inf, None
    for k in range(0, kmax + 1):
        combos = np.array(list(itertools.combinations(range(n), k)), dtype=int)
        combos = combos.reshape(len(combos), k)
        for chunk in np.array_split(combos, max(1, len(combos) // 20000)):
            X = np.zeros((len(chunk), n), dtype=np.int8)
            if k:
                X[np.arange(len(chunk))[:, None], chunk] = 1
            X = X[cbp.is_feasible(X)]
            if not len(X):
                continue
            f = score(X)
            i = int(np.argmin(f))
            if f[i] < best:
                best, bx = f[i], X[i]
    return best, bx


def e2_fidelity():
    """QUBO vs HUBO vs additive: regret on the exact saturating physics, vs cooling reach."""
    d = out("E2_fidelity")
    rows, keep = [], {}
    Ls = [50, 100, 200, 300] if not QUICK else [100]
    seeds = range(4) if not QUICK else range(1)
    for L, seed in itertools.product(Ls, seeds):
        iv = PARK.__class__("park", PARK.amplitude, float(L), PARK.cost)
        city = generate_city(12, 12, seed=seed, candidate_fraction=0.3)
        prob = UHIPlanningProblem(city, interventions=(iv,), prune_tol=0)
        cbp1 = prob.program(1)
        kmax = int(prob.budget // iv.cost)
        f_true, x_true = _true_optimum(prob, cbp1, kmax)
        f0 = prob.true_objective(np.zeros(prob.n))[0]
        for K in (1, 2, 3):
            cbp = prob.program(K)
            fK, xK = _true_optimum(prob, cbp1, kmax, cbp.objective.energies)   # exact optimum of the order-K model
            ft = prob.true_objective(xK)[0]
            rows.append({"decay_length_m": L, "seed": seed, "order": K, "n": prob.n,
                         "model_obj": fK, "true_obj": ft, "true_opt": f_true,
                         "regret_C": ft - f_true, "benefit_lost_pct": 100 * (ft - f_true) / (f0 - f_true),
                         "same_plan": bool(np.array_equal(xK, x_true)),
                         "model_error_C": fK - ft, "terms": cbp.objective.num_terms})
            if L == 200 and seed == 0:
                keep[f"order {K} ({['additive', 'QUBO', 'HUBO'][K - 1]})"] = xK
        # least-squares fitted QUBO (same size as the K=2 model, no truncation bias)
        fit = prob.fitted_quadratic(max_k=kmax, seed=seed)
        fK, xK = _true_optimum(prob, cbp1, kmax, fit.energies)
        ft = prob.true_objective(xK)[0]
        rows.append({"decay_length_m": L, "seed": seed, "order": "2-fit", "n": prob.n,
                     "model_obj": fK, "true_obj": ft, "true_opt": f_true,
                     "regret_C": ft - f_true, "benefit_lost_pct": 100 * (ft - f_true) / (f0 - f_true),
                     "same_plan": bool(np.array_equal(xK, x_true)),
                     "model_error_C": fK - ft, "terms": fit.num_terms})
        if L == 200 and seed == 0:
            keep["exact physics optimum"] = x_true
            qp.plot_plan_transition(prob, keep, d / "plans_L200_seed0.png",
                                    "Plans chosen by each surrogate, scored on exact saturating physics (L = 200 m)")
    df = pd.DataFrame(rows)
    df["order"] = df["order"].astype(str)
    save(df, d, "fidelity")
    s = df.groupby(["decay_length_m", "order"]).agg(
        regret_C_mean=("regret_C", "mean"), benefit_lost_pct_mean=("benefit_lost_pct", "mean"),
        same_plan_frac=("same_plan", "mean"), abs_model_error_C=("model_error_C", lambda v: np.abs(v).mean()),
        terms=("terms", "mean")).reset_index()
    save(s, d, "fidelity_summary")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    labels = {"1": "K=1 additive", "2": "K=2 QUBO (truncated)", "3": "K=3 HUBO (truncated)",
              "2-fit": "QUBO (least-squares fit)"}
    s["order"] = s["order"].astype(str)
    for K, g in s.groupby("order"):
        lab = labels[K]
        axs[0].plot(g.decay_length_m, g.benefit_lost_pct_mean, marker="o", label=lab)
        axs[1].plot(g.decay_length_m, g.abs_model_error_C, marker="o", label=lab)
    axs[0].set(xlabel="cooling decay length L (m)", ylabel="benefit lost vs exact optimum (%)",
               title="Plan quality on exact physics")
    axs[1].set(xlabel="cooling decay length L (m)", ylabel="|model - exact| at chosen plan (°C)",
               title="Surrogate error", yscale="log")
    for a in axs:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(d / "fidelity.png", dpi=150); plt.close(fig)
    print(s.to_string())


# ----------------------------------------------------------------- E3
def _planner_row(inst, prob, family, R):
    t0 = time.perf_counter()
    g = prob.greedy_plan()
    t = time.perf_counter() - t0
    feas = bool(inst.program.is_feasible(g)[0])
    f = float(inst.program.objective_values(g)[0])
    ok = feas and f <= inst.f_opt + 1e-7 * max(1.0, abs(inst.f_opt))
    return {"instance": inst.name, "family": family, "grid": R, "solver": "GreedyPlanner",
            "n_vars": inst.model.n, "p_success": float(ok), "p_feasible": float(feas),
            "best_gap_C": f - inst.f_opt, "time_per_read_s": t,
            "mean_benefit_ratio": (inst.f_baseline - f) / (inst.f_baseline - inst.f_opt) if feas else 0.0}


def e3_scaling():
    """Solver benchmark vs size on two QUBO families (certified optima)."""
    d = out("E3_scaling")
    fams = {"park": ([8, 10, 12, 14, 16], (PARK,), lambda R: None, 0),
            "mix+equity": ([8, 10, 12], DEFAULT_MIX, lambda R: 3 * R // 2 + 3, 1)}
    if QUICK:
        fams = {"park": ([8], (PARK,), lambda R: None, 0),
                "mix+equity": ([8], DEFAULT_MIX, lambda R: 3 * R // 2 + 3, 1)}
    seeds_inst = range(3) if not QUICK else range(1)
    run_seeds = range(4) if not QUICK else range(1)
    rows, planner, traces = [], [], {}
    for fam, (sizes, ivs, budget, eq) in fams.items():
        for R, si in itertools.product(sizes, seeds_inst):
            city = generate_city(R, R, seed=si, candidate_fraction=0.3)
            prob = UHIPlanningProblem(city, interventions=ivs, budget=budget(R), min_per_district=eq)
            try:
                inst = build_instance(f"{fam}_{R}x{R}_s{si}", prob.program(2),
                                      meta={"grid": R, "family": fam})
            except RuntimeError as e:
                print(f"skip {fam} {R} {si}: {e}")
                continue
            suite = solver_suite(inst.model, sweeps=1000 if not QUICK else 100,
                                 reads=32 if not QUICK else 8)
            if inst.model.n <= 16:
                suite["QAOA(p=4)"] = QAOA(p=4, shots=256)
            r, kept = run(inst, suite, run_seeds, keep=True, postprocess=True)
            rows += r
            planner.append(_planner_row(inst, prob, fam, R))
            if fam == "mix+equity" and R == 10 and si == 0:
                traces = {k: kept[k][0].info["traces"] for k in ("SA", "SQA", "SQA-fixedT", "Tabu") if k in kept}
            print(f"E3 {fam} {R}x{R} s{si} n={inst.model.n} done", flush=True)
    df = pd.DataFrame(rows)
    save(df, d, "scaling_raw")
    save(pd.DataFrame(planner), d, "greedy_planner")
    s = summarise(df, by=("family", "grid", "solver"),
                  cols=["n_vars", "p_success", "p_success_pp", "p_feasible", "mean_benefit_ratio",
                        "mean_benefit_ratio_pp", "best_gap_C", "tts99_s", "tts99_s_pp", "time_per_read_s"])
    save(s.reset_index(), d, "scaling_summary")
    for fam, g in df.groupby("family"):
        tag = fam.replace("+", "_")
        main = g[g.solver.isin(["SA", "SA-matched", "SQA", "SQA-fixedT", "Tabu", "Greedy"])]
        qp.plot_metric_vs(main, "n_vars", "p_success", path=d / f"p_success_vs_n_{tag}.png",
                          title=f"[{fam}] raw success probability, median & IQR",
                          xlabel="QUBO variables (decision + slack)", ylabel="P(success) per read")
        qp.plot_metric_vs(main.replace(np.inf, np.nan), "n_vars", "tts99_s", path=d / f"tts_vs_n_{tag}.png",
                          logy=True, title=f"[{fam}] TTS99 from raw samples (gaps = never solved)",
                          xlabel="QUBO variables", ylabel="TTS99 (s, single core)")
        qp.plot_metric_vs(main, "n_vars", "mean_benefit_ratio", path=d / f"benefit_vs_n_{tag}.png",
                          title=f"[{fam}] mean fraction of optimal cooling per read (infeasible = 0)",
                          xlabel="QUBO variables", ylabel="benefit ratio")
    if traces:
        qp.plot_convergence(traces, None, d / "convergence_mix_10x10.png",
                            title="Best penalised energy so far, mix+equity 10x10 (median, IQR over reads)")
    print(s.to_string())
    print(pd.DataFrame(planner).to_string())


# ----------------------------------------------------------------- E4
def e4_penalty():
    """Penalty-weight sweep: feasibility vs optimality trade-off, raw and post-processed."""
    d = out("E4_penalty")
    mults = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0] if not QUICK else [0.1, 1.0]
    rows = []
    insts = [("park_12x12", generate_city(12, 12, seed=1, candidate_fraction=0.3), (PARK,), None, 0),
             ("park_14x14", generate_city(14, 14, seed=2, candidate_fraction=0.3), (PARK,), None, 0),
             ("mix_8x8_equity", generate_city(8, 8, seed=2, candidate_fraction=0.35), DEFAULT_MIX, 12, 1)]
    for name, city, ivs, budget, eq in (insts if not QUICK else insts[:1]):
        prob = UHIPlanningProblem(city, interventions=ivs, budget=budget, min_per_district=eq)
        cbp = prob.program(2)
        f_opt = MILPSolver().solve(cbp).best_energy
        for m in mults:
            inst = build_instance(name, cbp, penalty_mult=m, f_opt=f_opt)
            suite = solver_suite(inst.model, sweeps=1000 if not QUICK else 100,
                                 reads=32 if not QUICK else 8, with_fixed=False, with_matched=False)
            suite = {k: v for k, v in suite.items() if k in ("SA", "SQA", "Tabu")}
            rows += run(inst, suite, range(3 if not QUICK else 1), postprocess=True)
        print(f"E4 {name} done", flush=True)
    df = pd.DataFrame(rows)
    save(df, d, "penalty_raw")
    s = df.groupby(["instance", "solver", "penalty_mult"])[
        ["p_feasible", "p_success", "p_success_pp", "mean_benefit_ratio", "mean_benefit_ratio_pp"]].mean()
    save(s.reset_index(), d, "penalty_summary")
    import matplotlib.pyplot as plt
    names = df.instance.unique()
    fig, axs = plt.subplots(len(names), 3, figsize=(14, 3.6 * len(names)), squeeze=False)
    for r, nm in enumerate(names):
        g = s.loc[nm].reset_index()
        for c, (col, lab) in enumerate([("p_feasible", "P(feasible), raw"), ("p_success", "P(optimal), raw"),
                                        ("p_success_pp", "P(optimal), post-processed")]):
            for sol, gg in g.groupby("solver"):
                axs[r, c].plot(gg.penalty_mult, gg[col], marker="o", label=sol)
            axs[r, c].set(xscale="log", xlabel="Λ / Λ_safe", ylabel=lab, title=f"{nm}: {lab}", ylim=(-0.03, 1.03))
            axs[r, c].axvline(1.0, color="k", ls=":", lw=1)
            axs[r, c].grid(alpha=0.3)
        axs[r, 0].legend()
    fig.tight_layout(); fig.savefig(d / "penalty_sweep.png", dpi=150); plt.close(fig)
    print(s.to_string())


# ----------------------------------------------------------------- E5
def e5_hubo():
    """Solve the HUBO (cubic saturation + quartic park-block synergy) natively vs after quadratisation."""
    d = out("E5_hubo")
    rows = []
    for seed in (range(3) if not QUICK else range(1)):
        city = generate_city(10, 10, seed=seed, candidate_fraction=0.45, road_spacing=5)
        prob = UHIPlanningProblem(city, cluster_bonus=0.4, prune_tol=1e-3)
        cbp = prob.program(3)
        ms = MILPSolver(time_limit=600).solve(cbp)
        for quad in (False, True):
            inst = build_instance(f"hubo_10x10_s{seed}", cbp, quadratized=quad, f_opt=ms.best_energy,
                                  meta={"milp_optimal": ms.info["optimal"], "hubo_degree": cbp.objective.degree})
            suite = solver_suite(inst.model, sweeps=1000 if not QUICK else 100,
                                 reads=32 if not QUICK else 8, with_fixed=False, with_matched=False)
            suite = {k: v for k, v in suite.items() if k in ("SA", "SQA", "Tabu", "Greedy")}
            rows += run(inst, suite, range(3 if not QUICK else 1),
                        extra={"formulation": "QUBO (Rosenberg)" if quad else "native HUBO"},
                        postprocess=True)
        print(f"E5 seed {seed} n={prob.n} deg={cbp.objective.degree} done", flush=True)
    df = pd.DataFrame(rows)
    save(df, d, "hubo_raw")
    s = df.groupby(["formulation", "solver"])[["n_vars", "p_feasible", "p_success", "p_success_pp",
                                               "mean_benefit_ratio", "time_per_read_s"]].mean()
    save(s.reset_index(), d, "hubo_summary")
    print(s.to_string())


# ----------------------------------------------------------------- E6
def e6_quantum():
    """Small-scale quantum simulations: QAOA depth, annealing gaps and adiabatic times vs penalty."""
    d = out("E6_quantum")
    city = generate_city(7, 7, seed=4, candidate_fraction=0.6, road_spacing=3)
    prob = UHIPlanningProblem(city, budget=9)          # 10 decision + 4 slack = 14 qubits
    cbp = prob.program(2)
    res = {"n_decision": cbp.n}
    rows = []
    f_opt = MILPSolver().solve(cbp).best_energy
    for m in ([0.1, 0.3, 1.0, 3.0, 10.0] if not QUICK else [1.0]):
        enc = cbp.to_penalty_model(penalty_weight=m * cbp.to_penalty_model().penalty_weight)
        n = enc.model.n
        ex = ExhaustiveSolver().sample(enc.model)
        gs_ok = bool(cbp.is_feasible(ex.best[: cbp.n])[0] and
                     np.isclose(cbp.objective_values(ex.best[: cbp.n])[0], f_opt))
        # Lanczos on s <= 0.95 (well-separated levels) plus the exact diagonal at s = 1
        s_grid = np.append(np.linspace(0, 0.95, 39 if not QUICK else 11), 1.0)
        spec = annealing_spectrum(enc.model, s_grid)
        inner = spec["s"] <= 0.95
        row = {"penalty_mult": m, "n_qubits": n, "ground_state_is_optimum": gs_ok,
               "min_gap": spec["min_gap"], "s_min_gap": spec["s_min_gap"],
               "final_gap": float(spec["gap"][-1]),
               "interior_min_gap_s_le_0.95": float(spec["gap"][inner].min()),
               "degeneracy": spec["degeneracy"], "adiabatic_time_scale_1_over_gap2": 1 / spec["min_gap"] ** 2}
        for T in ([10, 100, 1000] if not QUICK else [10]):
            row[f"p_ground_T{T}"] = schrodinger_anneal(enc.model, T, steps=max(400, 2 * T))["p_ground"]
        for p in ([1, 2, 4, 6] if not QUICK else [1]):
            q = QAOA(p=p, shots=1).sample(enc.model, seed=0)
            row[f"qaoa_p{p}_p_ground"] = q.info["p_ground"]
            row[f"qaoa_p{p}_approx_ratio"] = q.info["approx_ratio"]
        row["p_ground_uniform"] = q.info["p_ground_uniform"]
        rows.append(row)
        if m == 1.0:
            qp.plot_spectrum(spec, d / "spectrum_safe_penalty.png",
                             f"Annealing spectrum, Λ=Λ_safe, {n} qubits: min gap {spec['min_gap']:.3g}")
        print(f"E6 mult {m} n={n} gap={spec['min_gap']:.3g}", flush=True)
    df = pd.DataFrame(rows)
    save(df, d, "quantum_small")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(df.penalty_mult, df.min_gap, marker="o", label="min gap over s∈[0,1]")
    axs[0].plot(df.penalty_mult, df["interior_min_gap_s_le_0.95"], marker="s", label="min gap over s≤0.95")
    axs[0].plot(df.penalty_mult, df.final_gap, marker="^", ls="--", label="final gap (s=1)")
    axs[0].set(xscale="log", yscale="log", xlabel="Λ / Λ_safe", ylabel="gap (H_P scaled to [0,1])",
               title="Penalty weight vs annealing gaps")
    axs[0].legend()
    for c in [c for c in df.columns if c.startswith("p_ground_T")]:
        axs[1].plot(df.penalty_mult, df[c], marker="o", label=c.replace("p_ground_", ""))
    axs[1].set(xscale="log", xlabel="Λ / Λ_safe", ylabel="P(ground) after anneal", title="Exact Schrödinger annealing")
    axs[1].legend()
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(d / "gap_vs_penalty.png", dpi=150); plt.close(fig)
    print(df.to_string())


# ----------------------------------------------------------------- E7
def e7_showcase():
    """End-to-end planning run on a larger city with three intervention types + equity."""
    d = out("E7_showcase")
    city = generate_city(14, 14, seed=7, candidate_fraction=0.3, n_existing_parks=2)
    qp.plot_city(city, d / "city.png")
    prob = UHIPlanningProblem(city, interventions=DEFAULT_MIX, budget=24, min_per_district=1)
    cbp = prob.program(2)
    t0 = time.perf_counter()
    ms = MILPSolver(time_limit=600 if not QUICK else 30).solve(cbp)
    t_milp = time.perf_counter() - t0
    enc = cbp.to_penalty_model(penalty_weight=0.1 * cbp.to_penalty_model().penalty_weight)
    br = suggest_beta_range(enc.model, RES)
    sweeps = 2000 if not QUICK else 100
    plans, stats, traces = {}, {}, {}
    for label, s in {"SA": SimulatedAnnealing(num_sweeps=sweeps, num_reads=32, beta_range=br),
                     "SQA": SimulatedQuantumAnnealing(num_sweeps=sweeps, num_reads=16, beta_range=br,
                                                      gamma_range=(0.3, 1e-6)),
                     "Tabu": TabuSearch(max_iter=5000, num_reads=32)}.items():
        ss = s.sample(enc.model, seed=0)
        t0 = time.perf_counter()
        X = postprocess(cbp, ss.samples)
        tpp = time.perf_counter() - t0
        f = np.where(cbp.is_feasible(X), cbp.objective_values(X), np.inf)
        plans[f"{label} + post-processing"] = X[int(np.argmin(f))]
        traces[label] = ss.info["traces"]
        stats[label] = {"time_s": ss.wall_time + tpp, "best_obj": float(f.min()),
                        "gap_to_milp_C": float(f.min() - ms.best_energy),
                        "p_reach_milp": float(np.mean(f <= ms.best_energy + 1e-7))}
    greedy = prob.greedy_plan()
    all_plans = {"Greedy planner (no equity)": greedy, **plans,
                 f"MILP ({'optimal' if ms.info['optimal'] else 'time-limited'})": ms.best}
    qp.plot_plan_transition(prob, {"MILP": ms.best, "SQA + pp": plans["SQA + post-processing"],
                                   "Greedy planner": greedy},
                            d / "transition.png", "Baseline -> optimised intervention plans (14x14, 3 intervention types, equity)")
    qp.plot_cooling_map(prob, ms.best, d / "cooling_milp.png", "Cooling delivered by the MILP-optimal plan")
    qp.plot_convergence(traces, None, d / "convergence.png",
                        title="Best penalised energy so far (Λ = 0.1 Λ_safe)")
    summary = {"n_decision": cbp.n, "n_total_qubo": enc.model.n, "budget": prob.budget,
               "milp_time_s": t_milp, "milp_optimal": ms.info["optimal"], "milp_obj": ms.best_energy,
               "solvers": stats,
               "plans": {k: {**prob.report(x), "feasible": bool(cbp.is_feasible(x)[0]),
                             "objective_qubo_C": float(cbp.objective_values(x)[0])}
                         for k, x in all_plans.items()}}
    dump(summary, d / "summary.json")
    print(json.dumps(summary, indent=1, default=str))


# ----------------------------------------------------------------- E8
def _true_benefit(prob, X, x_ref):
    """Benefit ratio on the *exact* saturating physics, relative to the plan x_ref (MILP)."""
    X = np.atleast_2d(X)
    f0 = prob.true_objective(np.zeros(prob.n))[0]
    fr = prob.true_objective(x_ref)[0]
    return (f0 - prob.true_objective(X)) / (f0 - fr)


def _heuristic_row(inst_name, fam, method, cbp, prob, X, E_obj, f_opt, x_milp, wall, extra=None):
    feas = cbp.is_feasible(X)
    f = cbp.objective_values(X)
    ok = feas & (f <= f_opt + 1e-7 * max(1.0, abs(f_opt)))
    br = np.where(feas, _true_benefit(prob, X, x_milp), 0.0)
    p = float(ok.mean())
    return {"instance": inst_name, "family": fam, "method": method, "p_opt": p,
            "p_feasible": float(feas.mean()), "benefit_true": float(br.mean()),
            "best_benefit_true": float(br.max()), "wall_time_s": wall,
            "reads_for_99": (np.inf if p == 0 else (1.0 if p >= 0.99 else np.log(0.01) / np.log(1 - p))),
            **(extra or {})}


def _xmixer_exact(cbp, enc, qaoa_ss, x_milp, prob, f_opt):
    """Exact P(opt), P(feasible), E[benefit] of an X-mixer penalty-QAOA state."""
    cost = all_energies(enc.model)
    mu, sd = cost.mean(), cost.std() or 1.0
    th = qaoa_ss.info["angles"]
    p = qaoa_ss.info["p"]
    psi = qaoa_state((cost - mu) / sd, th[:p], th[p:])
    pr = np.abs(psi) ** 2
    pr /= pr.sum()
    n = enc.model.n
    Xall = ((np.arange(2 ** n)[:, None] >> np.arange(cbp.n)) & 1).astype(np.int8)
    feas = cbp.is_feasible(Xall)
    f = cbp.objective_values(Xall)
    ok = feas & (f <= f_opt + 1e-7 * max(1.0, abs(f_opt)))
    br = np.where(feas, _true_benefit(prob, Xall, x_milp), 0.0)
    return float(pr[ok].sum()), float(pr[feas].sum()), float(pr @ br)


def _e8_instances():
    """(name, family, prob, program).  Park: E6 instance, its exact-cardinality twin, E3 park
    8x8/10x10.  Mix: 6x6 cities with three intervention types and equity."""
    out = []
    c = generate_city(7, 7, seed=4, candidate_fraction=0.6, road_spacing=3)
    pr = UHIPlanningProblem(c, budget=9)
    out.append(("park_E6_7x7", "park", pr, pr.program(2)))
    k = pr.budget // PARK.cost
    card = ConstrainedBinaryProgram(pr.objective(2), [LinearConstraint({v: 1 for v in range(pr.n)}, "==", k, "cardinality")],
                                    names=pr.names)
    out.append((f"park_E6_7x7_exactly{k}", "park-cardinality", pr, card))
    grids = [8, 10] if not QUICK else [8]
    for R, si in itertools.product(grids, range(3) if not QUICK else range(1)):
        c = generate_city(R, R, seed=si, candidate_fraction=0.3)
        pr = UHIPlanningProblem(c)
        out.append((f"park_{R}x{R}_s{si}", "park", pr, pr.program(2)))
    mix_cfg = [(6, 0.6), (7, 0.4)] if not QUICK else [(6, 0.6)]
    for (R, cf), si in itertools.product(mix_cfg, range(3) if not QUICK else range(1)):
        c = generate_city(R, R, seed=si, candidate_fraction=cf)
        pr = UHIPlanningProblem(c, interventions=DEFAULT_MIX, budget=9, min_per_district=1)
        out.append((f"mix_{R}x{R}_s{si}", "mix+equity", pr, pr.program(2)))
    return out


def e8_mixers():
    """Constraint-preserving (XY / Dicke / projected) QAOA vs penalty QAOA vs classical baselines."""
    d = out("E8_mixers")
    ps = [1, 2] if not QUICK else [1]
    rows, notes = [], []
    for name, fam, prob, cbp in _e8_instances():
        t0 = time.perf_counter()
        ms = MILPSolver().solve(cbp)
        t_milp = time.perf_counter() - t0
        f_opt, x_milp = ms.best_energy, ms.best
        F = enumerate_feasible(cbp)
        base = {"n_decision": cbp.n, "n_feasible": len(F)}
        rows.append(_heuristic_row(name, fam, "MILP (HiGHS)", cbp, prob, x_milp, None, f_opt, x_milp, t_milp, base))
        t0 = time.perf_counter()
        g = prob.greedy_plan()
        rows.append(_heuristic_row(name, fam, "Greedy planner", cbp, prob, g, None, f_opt, x_milp,
                                   time.perf_counter() - t0, base))
        # --- constraint-preserving QAOA (native objective, no slack, no penalty)
        variants = [("XY-QAOA complete (swap+add/remove)", ("swap", "add_remove"), "complete")]
        if fam == "park-cardinality":
            variants = [("Dicke-XY complete", ("swap",), "complete"), ("Dicke-XY ring", ("swap",), "ring")]
        if len(F) > 4000:
            notes.append(f"{name}: |F|={len(F)} > 4000, constrained-QAOA state vector skipped")
        else:
            for label, moves, graph in variants:
                q0 = ConstrainedQAOA(moves=moves, swap_graph=graph)
                prep = q0.prepare(cbp, F)
                for p in ps:
                    q = ConstrainedQAOA(p=p, moves=moves, swap_graph=graph, shots=256, restarts=1, maxiter=100)
                    ss = q.sample_program(cbp, seed=0, prep=prep)
                    pr_ = ss.info["prob"]
                    br = pr_ @ _true_benefit(prob, F, x_milp)
                    pp = ss.info["p_opt"]
                    rows.append({"instance": name, "family": fam, "method": f"{label} p={p}", "p": p,
                                 "p_opt": pp, "p_feasible": float(cbp.is_feasible(ss.samples).mean()),
                                 "benefit_true": float(br), "best_benefit_true": float(_true_benefit(prob, ss.samples, x_milp).max()),
                                 "wall_time_s": ss.wall_time, "p_opt_uniform": ss.info["p_opt_uniform"],
                                 "mixer_components": ss.info["mixer_components"],
                                 "reads_for_99": (np.inf if pp == 0 else (1.0 if pp >= 0.99 else np.log(0.01) / np.log(1 - pp))),
                                 **base})
        # --- X-mixer penalty QAOA (existing baseline) on the safe-penalty QUBO
        enc = cbp.to_penalty_model()
        if enc.model.n <= 16:
            for p in ps:
                t0 = time.perf_counter()
                qs = QAOA(p=p, shots=256).sample(enc.model, seed=0)
                p_opt, p_feas, br = _xmixer_exact(cbp, enc, qs, x_milp, prob, f_opt)
                rows.append({"instance": name, "family": fam, "method": f"X-mixer penalty QAOA p={p}", "p": p,
                             "p_opt": p_opt, "p_feasible": p_feas, "benefit_true": br,
                             "wall_time_s": time.perf_counter() - t0, "n_qubits": enc.model.n,
                             "reads_for_99": (np.inf if p_opt == 0 else np.log(0.01) / np.log(1 - p_opt)), **base})
        else:
            notes.append(f"{name}: penalty QUBO has {enc.model.n} > 16 qubits, X-mixer QAOA skipped")
        # --- classical heuristics
        seeds = range(3) if not QUICK else range(1)
        sweeps = 1000 if not QUICK else 100
        for sd in seeds:
            fs = FeasibleSA(num_sweeps=sweeps, num_reads=32).sample_program(cbp, seed=sd)
            rows.append(_heuristic_row(name, fam, "FeasibleSA", cbp, prob, fs.samples, None, f_opt, x_milp,
                                       fs.wall_time, {**base, "seed": sd}))
            br_ = suggest_beta_range(enc.model, RES)
            for label, s in (("SA (safe-Λ QUBO)", SimulatedAnnealing(num_sweeps=sweeps, num_reads=32, beta_range=br_)),
                             ("Tabu (safe-Λ QUBO)", TabuSearch(max_iter=max(2000, 40 * enc.model.n), num_reads=32))):
                ss = s.sample(enc.model, seed=sd)
                rows.append(_heuristic_row(name, fam, label, cbp, prob, ss.samples[:, :cbp.n], None, f_opt, x_milp,
                                           ss.wall_time, {**base, "seed": sd}))
        print(f"E8 {name}: n={cbp.n} |F|={len(F)} done", flush=True)
    df = pd.DataFrame(rows)
    save(df, d, "raw")
    s = df.groupby(["family", "instance", "method"], sort=False)[
        ["p_opt", "p_feasible", "benefit_true", "wall_time_s", "reads_for_99"]].mean().reset_index()
    save(s, d, "summary")
    fam = df.groupby(["family", "method"], sort=False)[["p_opt", "p_feasible", "benefit_true", "wall_time_s"]].mean().reset_index()
    save(fam, d, "summary_by_family")
    (d / "e8.md").write_text("# E8 constraint-preserving mixers\n\n" +
                             "Per-family means over instances (and 3 seeds for heuristics). "
                             "QAOA P(opt) is the exact probability of the certified optimum; heuristic P(opt) is per read. "
                             "benefit_true uses exact saturating physics relative to the MILP plan; infeasible = 0.\n\n" +
                             fam.round(4).to_markdown(index=False) + "\n\n## Skipped\n\n" +
                             ("\n".join(f"- {n}" for n in notes) or "- none") + "\n")
    print(fam.round(4).to_string())
    print("\n".join(notes))


# ----------------------------------------------------------------- E9
def e9_hybrid():
    """Declared hybrid recipe: weak penalty (α·Λ_safe) + sampling + repair/feasible local search."""
    d = out("E9_hybrid")
    alphas = [0.01, 0.03, 0.1, 1.0] if not QUICK else [0.01, 1.0]
    seeds = range(2) if not QUICK else range(1)
    sweeps = 1000 if not QUICK else 100
    cases = [(R, si) for R in ([8, 10, 12] if not QUICK else [8]) for si in (range(3) if not QUICK else range(1))]
    rows = []
    for R, si in cases:
        city = generate_city(R, R, seed=si, candidate_fraction=0.3)
        prob = UHIPlanningProblem(city, interventions=DEFAULT_MIX, budget=3 * R // 2 + 3, min_per_district=1)
        cbp = prob.program(2)
        t0 = time.perf_counter()
        try:
            ms = MILPSolver(time_limit=300).solve(cbp)
        except RuntimeError as e:
            print(f"skip mix {R}x{R} s{si}: {e}")
            continue
        t_milp = time.perf_counter() - t0
        name = f"mix_{R}x{R}_s{si}"
        f_opt, x_milp = ms.best_energy, ms.best
        meta = {"instance": name, "grid": R, "n_decision": cbp.n, "milp_optimal": ms.info["optimal"]}
        rows.append({**meta, "solver": "MILP (HiGHS)", "alpha": np.nan, "p_success": 1.0, "p_feasible": 1.0,
                     "mean_benefit_ratio": 1.0, "wall_time_s": t_milp, "time_per_read_s": t_milp})
        t0 = time.perf_counter()
        g = prob.greedy_plan()
        tg = time.perf_counter() - t0
        gf = bool(cbp.is_feasible(g)[0])
        fg = float(cbp.objective_values(g)[0])
        f0 = float(cbp.objective_values(np.zeros(cbp.n))[0])
        rows.append({**meta, "solver": "Greedy planner", "alpha": np.nan,
                     "p_success": float(gf and fg <= f_opt + 1e-7 * abs(f_opt)), "p_feasible": float(gf),
                     "mean_benefit_ratio": (f0 - fg) / (f0 - f_opt) if gf else 0.0, "wall_time_s": tg,
                     "time_per_read_s": tg})
        for sd in seeds:
            fs = FeasibleSA(num_sweeps=sweeps, num_reads=32).sample_program(cbp, seed=sd)
            f = cbp.objective_values(fs.samples)
            ok = f <= f_opt + 1e-7 * abs(f_opt)
            rows.append({**meta, "solver": "FeasibleSA", "alpha": np.nan, "seed": sd,
                         "p_success": float(ok.mean()), "p_feasible": 1.0,
                         "mean_benefit_ratio": float(((f0 - f) / (f0 - f_opt)).mean()),
                         "wall_time_s": fs.wall_time, "time_per_read_s": fs.wall_time / 32,
                         "flips_attempted": fs.info["flips_attempted"]})
        for a in alphas:
            inst = build_instance(name, cbp, penalty_mult=a, f_opt=f_opt, meta={"grid": R, "alpha": a})
            suite = solver_suite(inst.model, sweeps=sweeps, reads=32, with_fixed=False, with_matched=True)
            suite = {k: v for k, v in suite.items() if k in ("SA", "SA-matched", "SQA", "Tabu")}
            suite["SQA"].num_reads = 16
            rows += run(inst, suite, seeds, extra={"alpha": a, "milp_optimal": ms.info["optimal"]}, postprocess=True)
        print(f"E9 {name} n={cbp.n} done", flush=True)
    df = pd.DataFrame(rows)
    save(df, d, "raw")
    cols = ["p_feasible", "p_success", "p_success_pp", "mean_benefit_ratio", "mean_benefit_ratio_pp",
            "time_per_read_s", "time_per_read_s_pp"]
    cols = [c for c in cols if c in df.columns]
    s = df.groupby(["solver", "alpha"], dropna=False)[cols].mean().reset_index()
    save(s, d, "summary")
    s2 = df.groupby(["grid", "solver", "alpha"], dropna=False)[cols].mean().reset_index()
    save(s2, d, "summary_by_size")
    print(s.round(4).to_string())


# ----------------------------------------------------------------- E10
def e10_penalty_form():
    """Quadratic slack (exact) vs unbalanced and linear penalties (no slack, not exact)."""
    d = out("E10_penalty_form")
    cfgs = [(6, s, (PARK,), None, 0) for s in range(3)] + [(8, s, (PARK,), None, 0) for s in range(3)] + \
           [(6, s, DEFAULT_MIX, 6, 0) for s in range(2)] + [(8, 1, (PARK,), None, 1)]
    if QUICK:
        cfgs = cfgs[:1] + cfgs[6:7]
    rows = []
    for R, seed, ivs, budget, eq in cfgs:
        city = generate_city(R, R, seed=seed, candidate_fraction=0.4)
        prob = UHIPlanningProblem(city, interventions=ivs, budget=budget, min_per_district=eq)
        cbp = prob.program(2)
        name = f"{R}x{R}_s{seed}_{'mix' if len(ivs) > 1 else 'park'}_eq{eq}"
        try:
            ms = MILPSolver().solve(cbp)
        except RuntimeError as e:
            print(f"skip {name}: {e}")
            continue
        f_opt = ms.best_energy
        f0 = float(cbp.objective_values(np.zeros(cbp.n))[0])
        safe = cbp.to_penalty_model().penalty_weight
        lin = np.array([cbp.objective.terms.get((v,), 0.0) for v in range(cbp.n)])
        budget_c = next(c for c in cbp.constraints if c.name == "budget")
        cost = np.array([budget_c.coeffs.get(v, 1) for v in range(cbp.n)], dtype=float)
        mu = float(np.median(-lin / cost))                 # median cooling gain per cost unit
        phys = max(abs(c) for c in cbp.objective.terms.values())
        encs = [("quadratic slack (exact)", "Λ_safe", cbp.to_penalty_model())]
        for b in (0.1, 0.3, 1.0):
            encs.append(("unbalanced", f"{b}·Λ_safe", cbp.to_penalty_model(penalty_weight=b * safe, form="unbalanced")))
        for m in (0.5, 1.0, 2.0):
            encs.append(("linear", f"{m}·μ", cbp.to_penalty_model(penalty_weight=m * mu, form="linear")))
        for form, weight, enc in encs:
            model = enc.model
            row = {"instance": name, "form": form, "weight": weight, "n_vars": model.n,
                   "dynamic_range": max(abs(c) for c in model.terms.values()) / phys}
            if model.n <= 24:
                gs = ExhaustiveSolver().sample(model).best[: cbp.n]
                row["ground_feasible"] = bool(cbp.is_feasible(gs)[0])
                row["ground_is_optimum"] = bool(row["ground_feasible"] and
                                                cbp.objective_values(gs)[0] <= f_opt + 1e-7 * abs(f_opt))
            br = suggest_beta_range(model, RES)
            for sd in (range(3) if not QUICK else range(1)):
                ss = SimulatedAnnealing(num_sweeps=1000 if not QUICK else 100, num_reads=32, beta_range=br).sample(model, seed=sd)
                Xd = ss.samples[:, : cbp.n]
                feas = cbp.is_feasible(Xd)
                f = cbp.objective_values(Xd)
                Xp = postprocess(cbp, Xd)
                fp = cbp.objective_values(Xp)
                okp = cbp.is_feasible(Xp)
                rows.append({**row, "seed": sd, "p_feasible": float(feas.mean()),
                             "p_opt": float((feas & (f <= f_opt + 1e-7 * abs(f_opt))).mean()),
                             "benefit": float(np.where(feas, (f0 - f) / (f0 - f_opt), 0).mean()),
                             "p_opt_pp": float((okp & (fp <= f_opt + 1e-7 * abs(f_opt))).mean()),
                             "benefit_pp": float(np.where(okp, (f0 - fp) / (f0 - f_opt), 0).mean())})
        print(f"E10 {name} done", flush=True)
    df = pd.DataFrame(rows)
    save(df, d, "raw")
    agg = {"n_vars": "mean", "dynamic_range": "median", "p_feasible": "mean", "p_opt": "mean",
           "benefit": "mean", "p_opt_pp": "mean", "benefit_pp": "mean"}
    if "ground_feasible" in df:
        agg.update({"ground_feasible": "mean", "ground_is_optimum": "mean"})
    s = df.groupby(["form", "weight"], sort=False).agg(agg).reset_index()
    save(s, d, "summary")
    print(s.round(4).to_string())


# ----------------------------------------------------------------- E11
def e11_pubo_gap():
    """Annealing spectra: native cubic HUBO penalty model vs its Rosenberg quadratisation."""
    d = out("E11_pubo_gap")
    rows, specs = [], {}
    cfg = [(6, 0), (6, 1)]
    for R, seed in cfg:
        city = generate_city(R, R, seed=seed, candidate_fraction=0.5, road_spacing=3)
        prob = UHIPlanningProblem(city, budget=6, prune_tol=0)
        cbp = prob.program(3)
        f_opt = MILPSolver().solve(cbp).best_energy
        enc = cbp.to_penalty_model()
        q = quadratize(enc.model)
        # (C) native objective restricted to F: infeasible states padded with a documented
        # offset above max_F f (diagonal simulation only; not a QUBO and not hardware-realisable)
        n = cbp.n
        Xall = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
        feas = cbp.is_feasible(Xall)
        fv = cbp.objective_values(Xall)
        pad = fv[feas].max() + (fv[feas].max() - fv[feas].min())
        diagC = np.where(feas, fv, pad)
        models = [("A: native cubic HUBO + slack penalty", enc.model, None),
                  ("B: Rosenberg QUBO of A", q.qubo, None),
                  ("C: native HUBO, infeasible padded (diagonal only)", None, diagC)]
        for label, model, diag in models:
            nq = model.n if model is not None else n
            if nq > 18:
                rows.append({"instance": f"{R}x{R}_s{seed}", "model": label, "n_qubits": nq, "skipped": "n>18"})
                continue
            s_grid = np.append(np.linspace(0, 0.95, 39 if not QUICK else 11), 1.0)
            spec = _spectrum_diag(model, diag, s_grid)
            row = {"instance": f"{R}x{R}_s{seed}", "model": label, "n_qubits": nq, "degree": model.degree if model is not None else cbp.objective.degree,
                   "min_gap": spec["min_gap"], "s_min_gap": spec["s_min_gap"], "final_gap": float(spec["gap"][-1]),
                   "interior_min_gap_s_le_0.95": float(spec["gap"][spec["s"] <= 0.95].min()),
                   "inv_gap2": 1 / spec["min_gap"] ** 2}
            for T in ([10, 100, 1000] if not QUICK else [10]):
                row[f"p_ground_T{T}"] = _anneal_diag(model, diag, T)
            rows.append(row)
            print(f"E11 {R}x{R}_s{seed} {label}: n={nq} gap={spec['min_gap']:.3g}", flush=True)
    df = pd.DataFrame(rows)
    save(df, d, "raw")
    print(df.round(6).to_string())


def _spectrum_diag(model, diag, s_grid):
    if diag is None:
        return annealing_spectrum(model, s_grid)
    n = int(np.log2(len(diag)))
    return annealing_spectrum(_DiagModel(n, diag), s_grid)


def _anneal_diag(model, diag, T):
    if diag is None:
        return schrodinger_anneal(model, T, steps=max(400, 2 * T))["p_ground"]
    n = int(np.log2(len(diag)))
    return schrodinger_anneal(_DiagModel(n, diag), T, steps=max(400, 2 * T))["p_ground"]


class _DiagModel:
    """Minimal stand-in carrying an explicit diagonal (little-endian) for the exact solvers."""

    def __init__(self, n, diag):
        self.n = n
        self.diag = np.asarray(diag, float)
        self.degree = None


EXPS = {"E1": e1_encoding, "E2": e2_fidelity, "E3": e3_scaling, "E4": e4_penalty,
        "E5": e5_hubo, "E6": e6_quantum, "E7": e7_showcase, "E8": e8_mixers, "E9": e9_hybrid,
        "E10": e10_penalty_form, "E11": e11_pubo_gap}


def main():
    global QUICK, ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(EXPS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT))
    a = ap.parse_args()
    QUICK = a.quick
    ROOT = Path(a.out)
    ROOT.mkdir(parents=True, exist_ok=True)
    import numba
    import scipy
    envp = ROOT / "env.json"
    env = json.loads(envp.read_text()) if envp.exists() else {}
    env.update({"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "numba": numba.__version__, "platform": platform.platform()})
    env.setdefault("runs", {})
    for k in a.only:
        env["runs"][k] = {"quick": QUICK, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    dump(env, envp)
    for k in a.only:
        t0 = time.perf_counter()
        print(f"===== {k}: {EXPS[k].__doc__.strip().splitlines()[0]}", flush=True)
        EXPS[k]()
        print(f"===== {k} finished in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
