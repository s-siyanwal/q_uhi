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


# ----------------------------------------------------------------- E8b
E8B_SEEDS = (4, 5, 6)          # seed 4 is the E6/E8 instance; 5, 6 are built the same way


def _cardinality_twin(seed):
    """E6-style 7x7 park city with 'exactly k parks' (k = budget // cost): equality, no slack."""
    c = generate_city(7, 7, seed=seed, candidate_fraction=0.6, road_spacing=3)
    pr = UHIPlanningProblem(c, budget=9)
    k = pr.budget // PARK.cost
    cbp = ConstrainedBinaryProgram(pr.objective(2), [LinearConstraint({v: 1 for v in range(pr.n)}, "==", k,
                                                                      "cardinality")], names=pr.names)
    return f"park_7x7_s{seed}_exactly{k}", pr, cbp, k


def e8b_depth():
    """Dicke-XY depth curve (p = 1..6) on the exactly-k park instances vs classical references."""
    d = out("E8b_depth")
    ps = [1, 2, 3, 4, 6] if not QUICK else [1, 2]
    seeds = E8B_SEEDS if not QUICK else E8B_SEEDS[:1]
    rows, notes = [], []
    for s in seeds:
        name, prob, cbp, k = _cardinality_twin(s)
        t0 = time.perf_counter()
        ms = MILPSolver().solve(cbp)
        t_milp = time.perf_counter() - t0
        f_opt, x_milp = ms.best_energy, ms.best
        F = enumerate_feasible(cbp)
        cF = cbp.objective_values(F)
        n_opt = int((cF <= f_opt + 1e-7 * max(1.0, abs(f_opt))).sum())
        base = {"instance": name, "city_seed": s, "n": cbp.n, "k": k, "n_feasible": len(F), "n_optima": n_opt}
        rows.append({**_heuristic_row(name, "park-cardinality", "MILP (HiGHS)", cbp, prob, x_milp, None, f_opt,
                                      x_milp, t_milp), **base})
        t0 = time.perf_counter()
        g = prob.greedy_plan()
        rows.append({**_heuristic_row(name, "park-cardinality", "Greedy planner", cbp, prob, g, None, f_opt, x_milp,
                                      time.perf_counter() - t0), **base})
        rows.append({**base, "method": "uniform on F", "p_opt": n_opt / len(F), "p_feasible": 1.0,
                     "benefit_true": float(_true_benefit(prob, F, x_milp).mean()), "wall_time_s": 0.0})
        for sd in range(3):
            fs = FeasibleSA(num_sweeps=1000 if not QUICK else 100, num_reads=32).sample_program(cbp, seed=sd)
            rows.append({**_heuristic_row(name, "park-cardinality", "FeasibleSA", cbp, prob, fs.samples, None, f_opt,
                                          x_milp, fs.wall_time), **base, "seed": sd})
        bF = _true_benefit(prob, F, x_milp)
        for label, graph in (("Dicke-XY complete", "complete"), ("Dicke-XY ring", "ring")):
            prep = ConstrainedQAOA(moves=("swap",), swap_graph=graph).prepare(cbp, F)
            for p in ps:
                q = ConstrainedQAOA(p=p, moves=("swap",), swap_graph=graph, shots=256, restarts=1, maxiter=300)
                ss = q.sample_program(cbp, seed=0, prep=prep)
                rows.append({**base, "method": label, "p": p, "p_opt": ss.info["p_opt"], "p_feasible": 1.0,
                             "benefit_true": float(ss.info["prob"] @ bF), "wall_time_s": ss.wall_time,
                             "nfev": ss.info["nfev"], "mixer_components": ss.info["mixer_components"],
                             "angles": json.dumps(np.round(ss.info["angles"], 6).tolist())})
        enc = cbp.to_penalty_model()
        if enc.model.n <= 16:
            for p in ps:
                t0 = time.perf_counter()
                qs = QAOA(p=p, shots=256).sample(enc.model, seed=0)
                p_opt, p_feas, br = _xmixer_exact(cbp, enc, qs, x_milp, prob, f_opt)
                rows.append({**base, "method": "X-mixer penalty QAOA", "p": p, "p_opt": p_opt, "p_feasible": p_feas,
                             "benefit_true": br, "wall_time_s": time.perf_counter() - t0, "n_qubits": enc.model.n})
        else:
            notes.append(f"{name}: penalty QUBO has {enc.model.n} > 16 qubits, X-mixer QAOA skipped")
        print(f"E8b {name}: n={cbp.n} k={k} |F|={len(F)} optima={n_opt} done", flush=True)
    df = pd.DataFrame(rows)
    save(df, d, "raw")
    df["p"] = df.get("p", np.nan)
    s = df.groupby(["method", "p"], dropna=False, sort=False)[
        ["p_opt", "p_feasible", "benefit_true", "wall_time_s"]].agg(["mean", "min", "max"]).reset_index()
    s.columns = ["_".join(c).rstrip("_") for c in s.columns]
    save(s, d, "summary")
    _plot_e8b(df, d / "p_opt_vs_p.png")
    (d / "e8b.md").write_text(
        "# E8b Dicke-XY depth curve\n\n"
        f"Exactly-k park instances (7x7 cities, seeds {', '.join(map(str, seeds))}; n = {df['n'].iloc[0]}, "
        f"k = {df['k'].iloc[0]}, |F| = C(n,k) = {df['n_feasible'].iloc[0]}). QAOA values are exact probabilities "
        "from the state vector (L-BFGS-B, linear-ramp start + 1 random restart, maxiter 300). Heuristic P(opt) is "
        "per read (3 seeds x 32 reads). benefit_true = exact saturating physics relative to the MILP plan. "
        "Mean / min / max over city seeds.\n\n" + s.round(4).to_markdown(index=False) +
        "\n\n## Skipped\n\n" + ("\n".join(f"- {n}" for n in notes) or "- none") + "\n")
    print(s.round(4).to_string())


def _plot_e8b(df, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    colors = {"Dicke-XY complete": "#1f77b4", "Dicke-XY ring": "#ff7f0e", "X-mixer penalty QAOA": "#7f7f7f"}
    for m, col in colors.items():
        g = df[df.method == m]
        if g.empty:
            continue
        for _, gi in g.groupby("instance"):
            ax.plot(gi.p, gi.p_opt, color=col, alpha=0.25, lw=0.8)
        mean = g.groupby("p").p_opt.mean()
        ax.plot(mean.index, mean.values, "o-", color=col, label=m)
    refs = {"MILP (HiGHS)": ("k", "-"), "Greedy planner": ("#2ca02c", "--"), "FeasibleSA": ("#9467bd", "-."),
            "uniform on F": ("#8c564b", ":")}
    for m, (col, ls) in refs.items():
        v = df[df.method == m].p_opt.mean()
        ax.axhline(v, color=col, ls=ls, lw=1.2, label=f"{m} ({v:.3g})")
    ax.set_yscale("log")
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("P(certified optimum)")
    ax.set_title("E8b: exactly-3 parks, mean over 3 cities (thin: per city)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- ANIM
def _chunked_run(make, run_chunk, X0, total, n_chunks, b0, b1):
    """Run an annealer as ``n_chunks`` consecutive pieces of one geometric β schedule, carrying
    each read's state forward, to snapshot incumbents without changing the solver API.
    Returns per-chunk snapshots (n_chunks, R, n), the concatenated best-so-far trace (R, S)
    and the chunk end indices."""
    edges = np.geomspace(b0, b1, n_chunks + 1)
    per = max(1, total // n_chunks)
    X, snaps, trs, ends = X0, [], [], []
    for i in range(n_chunks):
        ss = run_chunk(make(per, (edges[i], edges[i + 1])), X, i)
        X = ss.samples
        snaps.append(X.copy())
        trs.append(np.atleast_2d(ss.info["traces"]))
        ends.append(sum(t.shape[1] for t in trs) - 1)
    tr = np.minimum.accumulate(np.concatenate(trs, axis=1), axis=1)
    return np.array(snaps), tr, ends


def _anim_showcases():
    name_a, prob_a, cbp_a, _ = _cardinality_twin(E8B_SEEDS[0])
    R = 10 if not QUICK else 8
    city = generate_city(R, R, seed=0, candidate_fraction=0.3)
    prob_b = UHIPlanningProblem(city, interventions=DEFAULT_MIX, budget=3 * R // 2 + 3, min_per_district=1)
    return [("A_" + name_a, prob_a, cbp_a), (f"B_mix_{R}x{R}_s0", prob_b, prob_b.program(2))]


def e_anim():
    """Animated search-space and temperature-field comparisons (GIFs) on two showcase instances."""
    from scipy.optimize import minimize

    from quhi.analysis import animate as an

    d = out("ANIM")
    NF = 40 if not QUICK else 8
    FPS = 10
    sweeps = 1000 if not QUICK else 100
    manifest = {"frames_max": NF, "fps": FPS, "gifs": {}, "notes": []}

    def gif(frames, fname, meta):
        info = an.write_gif(frames, d / fname, fps=FPS)
        manifest["gifs"][fname] = {**meta, **info}
        print(f"  {fname}: {info['frames']} frames, {info['bytes'] / 1e6:.2f} MB", flush=True)

    for inst, prob, cbp in _anim_showcases():
        ms = MILPSolver(time_limit=300).solve(cbp)
        f_opt, x_milp = ms.best_energy, ms.best
        T0 = prob.temperature_field(None)
        vmin, vmax = float(T0.min()), float(T0.max())
        meta0 = {"instance": inst, "n": cbp.n, "vmin_C": vmin, "vmax_C": vmax}
        plans = {}                                            # method -> list of (x, step, energy)

        # --- greedy planner: one frame per accepted cell
        _, steps = prob.greedy_plan(return_steps=True)
        plans["Greedy planner"] = [(x, f"step {i}/{len(steps) - 1}", float(cbp.objective_values(x)[0]))
                                   for i, x in enumerate(steps)]

        # --- FeasibleSA, incumbents snapshotted every sweeps/NF sweeps
        rng = np.random.default_rng(0)
        fsa = FeasibleSA(num_sweeps=sweeps, num_reads=8)
        b0, b1 = suggest_beta_range(cbp.objective, RES)
        snaps, tr, ends = _chunked_run(
            lambda s, br: FeasibleSA(num_sweeps=s, num_reads=8, beta_range=br),
            lambda solver, X, i: solver.sample_program(cbp, seed=100 + i, x0=X),
            fsa.feasible_starts(cbp, rng), sweeps, NF, b0, b1)
        r = int(np.argmin(cbp.objective_values(snaps[-1])))
        plans["FeasibleSA"] = [(snaps[i][r], f"sweep {ends[i] + 1}/{tr.shape[1]}",
                                float(cbp.objective_values(snaps[i][r])[0])) for i in range(len(ends))]
        search = {"FeasibleSA": (tr[r], ends, [snaps[i][r] for i in range(len(ends))], "constrained f(x)", "")}

        # --- E9 hybrid: SA-matched on the α = 0.01 penalty QUBO, then repair + local search
        safe = cbp.to_penalty_model()
        enc = cbp.to_penalty_model(penalty_weight=0.01 * safe.penalty_weight)
        lam = enc.penalty_weight
        br = suggest_beta_range(enc.model, RES)
        m_sweeps = sweeps * 17
        X0 = np.random.default_rng(1).integers(0, 2, (8, enc.model.n)).astype(np.int8)
        snaps, tr, ends = _chunked_run(
            lambda s, b: SimulatedAnnealing(num_sweeps=s, num_reads=8, beta_range=b),
            lambda solver, X, i: solver.sample(enc.model, seed=200 + i, x0=X), X0, m_sweeps, NF, *br)
        r = int(np.argmin(enc.model.energies(snaps[-1])))
        lab = "SA-matched α=0.01 + repair"
        hy = [(snaps[i][r][: cbp.n], f"sweep {ends[i] + 1}/{tr.shape[1]} (raw sample)",
               float(enc.model.energies(snaps[i][r])[0])) for i in range(len(ends))]
        xpp = postprocess(cbp, snaps[-1][r][: cbp.n])[0]
        hy.append((xpp, "after repair + feasible local search", float(cbp.objective_values(xpp)[0])))
        plans[lab] = hy
        search["SA α=0.01 (penalty QUBO)"] = (tr[r], ends, [s[r] for s in snaps], f"f+ΛP, Λ={lam:.3g}", "")

        # --- Tabu and SQA on the same QUBO: energy traces only, final plan shown
        for label, s in (("Tabu α=0.01 (penalty QUBO)", TabuSearch(max_iter=max(2000, 40 * enc.model.n), num_reads=8)),
                         ("SQA α=0.01 (penalty QUBO, PIMC)",
                          SimulatedQuantumAnnealing(num_sweeps=sweeps, num_reads=8, trotter_slices=16, beta_range=br,
                                                    gamma_range=(0.3, 1e-6)))):
            ss = s.sample(enc.model, seed=0)
            r = int(np.argmin(ss.energies))
            t = np.minimum.accumulate(np.atleast_2d(ss.info["traces"])[r])
            idx = list(an.frame_indices(len(t), NF))
            search[label] = (t, idx, [ss.samples[r]] * len(idx), f"f+ΛP, Λ={lam:.3g}",
                             "final plan (intermediate states not snapshotted)")

        for label, (t, idx, xs, ylab, note) in search.items():
            frames = [an.trace_frame(prob, cbp, t, k, x, label, f"iteration {k + 1}/{len(t)}", ylab,
                                     f_opt if ylab.startswith("constrained") else None, note)
                      for k, x in zip(idx, xs)][:NF]
            gif(frames, f"search_{_slug(label)}_{inst}.gif", {**meta0, "method": label, "energy_axis": ylab})

        # --- constraint-preserving QAOA (only if |F| fits)
        try:
            F = enumerate_feasible(cbp, max_states=4000)
        except ValueError:
            F = None
        if F is None or len(F) > 4000:
            manifest["notes"].append(f"{inst}: |F| > 4000, ConstrainedQAOA state vector not simulated; "
                                     "the QAOA cell of the comparison uses the E9 hybrid instead")
        else:
            moves = ("swap",) if len(cbp.constraints) == 1 and cbp.constraints[0].sense == "==" else ("swap", "add_remove")
            qlab = "Dicke-XY complete" if moves == ("swap",) else "XY-QAOA complete"
            prep = ConstrainedQAOA(moves=moves).prepare(cbp, F)
            cF = prep["cost"]
            opt = cF <= f_opt + 1e-7 * max(1.0, abs(f_opt))
            cs = (cF - cF.mean()) / (cF.std() or 1.0)
            P = 6 if not QUICK else 2
            k = (np.arange(P) + 0.5) / P
            th0 = np.concatenate([0.75 * k, 0.75 * (1 - k)])
            its = [th0.copy()]
            ev = lambda th: float(np.real(np.vdot(s_ := ConstrainedQAOA.state(prep, th[:P], th[P:]), cs * s_)))
            th = minimize(ev, th0, method="L-BFGS-B", callback=lambda v: its.append(v.copy()),
                          options={"maxiter": 300}).x
            states = [np.full(len(F), 1 / len(F))] + \
                     [np.abs(ConstrainedQAOA.state(prep, t_[:P], t_[P:])) ** 2 for t_ in its]
            ymax = max(float(s_.max()) for s_ in states) * 1.05
            hdr = lambda pr, step: (f"{qlab} p={P}  |  {step}   E[f] = {pr @ cF:.4f} °C\n"
                                    f"P(opt) = {pr[opt].sum():.3f}   |F| = {len(F)}   feasible: yes (by construction)")
            frames = [an.distribution_frame(cF, pr, opt, "probability mass on feasible plans, cheapest → hottest",
                                            hdr(pr, "Dicke/uniform start" if i == 0 else f"L-BFGS-B iterate {i}/{len(states) - 1}"),
                                            ylim=ymax)
                      for i, pr in ((i, states[i]) for i in an.frame_indices(len(states), NF))]
            gif(frames, f"search_{_slug(qlab)}_p{P}_{inst}.gif", {**meta0, "method": f"{qlab} p={P}",
                                                                  "n_feasible": len(F)})
            layers = [np.full(len(F), 1 / len(F))] + \
                     [np.abs(ConstrainedQAOA.state(prep, th[:l], th[P:P + l])) ** 2 for l in range(1, P + 1)]
            frames = [an.distribution_frame(cF, pr, opt, "probability mass on feasible plans, cheapest → hottest",
                                            hdr(pr, f"after layer {l}/{P} (optimised angles)"),
                                            ylim=max(s_.max() for s_ in layers) * 1.05)
                      for l, pr in enumerate(layers)]
            gif(frames, f"search_{_slug(qlab)}_layers_{inst}.gif", {**meta0, "method": f"{qlab} p={P} layers"})
            prob_final = states[-1] / states[-1].sum()
            # UHI view: draws from the final |ψ|² (p=2 as in E8) and the p=max state for the comparison
            q2 = ConstrainedQAOA(p=2, moves=moves, shots=NF, restarts=1, maxiter=300).sample_program(cbp, seed=0, prep=prep)
            plans[f"{qlab} p=2"] = [(x, f"draw {i + 1}/{NF} from final |ψ|² (not annealing time)",
                                     float(cbp.objective_values(x)[0])) for i, x in enumerate(q2.samples)]
            draws = np.random.default_rng(0).choice(len(F), size=NF, p=prob_final)
            plans[f"{qlab} p={P}"] = [(F[j], f"draw {i + 1}/{NF} from final |ψ|²", float(cF[j]))
                                      for i, j in enumerate(draws)]
            manifest["gifs"][f"search_{_slug(qlab)}_p{P}_{inst}.gif"]["p_opt_final"] = float(prob_final[opt].sum())

            # --- X-mixer penalty QAOA over the full 2^n diagonal (safe Λ), n <= 16 only
            if safe.model.n <= 16:
                cost = all_energies(safe.model)
                csx = (cost - cost.mean()) / (cost.std() or 1.0)
                nq = safe.model.n
                Xall = ((np.arange(2 ** nq)[:, None] >> np.arange(nq)) & 1).astype(np.int8)
                fe = cbp.is_feasible(Xall[:, : cbp.n])
                opx = fe & (cbp.objective_values(Xall[:, : cbp.n]) <= f_opt + 1e-7 * max(1.0, abs(f_opt)))
                its = [th0.copy()]
                evx = lambda t_: float(np.real(np.vdot(s_ := qaoa_state(csx, t_[:P], t_[P:]), csx * s_)))
                minimize(evx, th0, method="L-BFGS-B", callback=lambda v: its.append(v.copy()), options={"maxiter": 300})
                sts = [np.abs(qaoa_state(csx, t_[:P], t_[P:])) ** 2 for t_ in its]
                ymax = max(float(s_.max()) for s_ in sts) * 1.05
                frames = [an.distribution_frame(
                    cost, pr, opx, "all 2^n bitstrings sorted by penalised energy f+ΛP (safe Λ)",
                    f"X-mixer penalty QAOA p={P}  |  L-BFGS-B iterate {i}/{len(sts) - 1}\n"
                    f"P(opt) = {pr[opx].sum():.4f}   P(feasible) = {pr[fe].sum():.3f}   n = {nq} qubits",
                    feasible=fe, ylim=ymax, xlabel="bitstrings sorted by f+ΛP (low → high)")
                    for i, pr in ((i, sts[i]) for i in an.frame_indices(len(sts), NF))]
                gif(frames, f"search_xmixer_penalty_qaoa_p{P}_{inst}.gif", {**meta0, "method": f"X-mixer QAOA p={P}"})
            else:
                manifest["notes"].append(f"{inst}: penalty QUBO has {safe.model.n} > 16 qubits, X-mixer QAOA not animated")

        # --- UHI (physical-space) GIFs on one cooling scale per instance
        allx = [x for v in plans.values() for x, _, _ in v] + [x_milp]
        dtmax = float(max((T0 - prob.temperature_field(x)).max() for x in allx))
        meta0["dtmax_C"] = dtmax
        milp_lab = "MILP (HiGHS)"
        for label, seq in plans.items():
            eax = f"f+ΛP (Λ={lam:.3g})" if label.startswith("SA-matched") else "constrained f(x)"
            frames = [an.uhi_frame(prob, cbp, x_milp, milp_lab, "certified optimum (reference)", f_opt,
                                   vmin, vmax, dtmax, "constrained f(x)")]
            frames += [an.uhi_frame(prob, cbp, x, label, step, e, vmin, vmax, dtmax,
                                    "constrained f(x)" if "after repair" in step else eax)
                       for x, step, e in (seq[i] for i in an.frame_indices(len(seq), NF - 1))]
            gif(frames, f"uhi_{_slug(label)}_{inst}.gif", {**meta0, "method": label})

        # --- side-by-side comparison
        qkey = next((k for k in plans if k.endswith("p=6") or (QUICK and k.endswith("p=2") and "XY" in k)), None)
        third = qkey or lab
        cols = ["Greedy planner", "FeasibleSA", third]
        frames = []
        for t in range(NF):
            cells = {}
            for c in cols:
                seq = plans[c]
                x, step, _ = seq[min(len(seq) - 1, int(round(t * (len(seq) - 1) / (NF - 1))))]
                cells[c] = (x, step)
            cells[milp_lab] = (x_milp, "certified optimum (static)")
            frames.append(an.compare_frame(prob, cbp, cells, vmin, vmax,
                                           f"{inst}: locked temperature scale {vmin:.2f}–{vmax:.2f} °C" +
                                           ("" if qkey else "  (XY-QAOA does not fit: |F| > 4000; E9 hybrid shown)")))
        gif(frames, f"compare_temp_{inst}.gif", {**meta0, "columns": cols + [milp_lab]})
        finals = {c: plans[c][-1][0] for c in cols}
        if qkey:   # most probable plan for QAOA in the still
            finals[third] = F[int(np.argmax(prob_final))]
        finals[milp_lab] = x_milp
        png = "compare_final.png" if inst.startswith("A_") else f"compare_final_{inst}.png"
        qp.plot_plan_transition(prob, finals, d / png, f"{inst}: final plans" +
                                (" (QAOA: most probable plan of |ψ|²)" if qkey else ""))
        manifest.setdefault("instances", {})[inst] = {**meta0, "f_opt": f_opt,
                                                      "milp_exposure_T": float(prob.true_objective(x_milp)[0])}
    dump(manifest, d / "manifest.json")


def _slug(s):
    import re
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# ----------------------------------------------------------------- E12
E12_SEEDS = (100, 101, 102)
E12_CLASSICAL = ("MILP (HiGHS, 8 s cap)", "Greedy planner", "FeasibleSA", "Tabu-on-F", "SA-matched α=0.01 + repair")
E12_KIND = {"MILP (HiGHS, 8 s cap)": "classical", "Greedy planner": "classical", "FeasibleSA": "classical",
            "Tabu-on-F": "classical", "SA-matched α=0.01 + repair": "classical/hybrid",
            "FeasibleSQA": "quantum-inspired (PIMC on F)", "QAOA-seeded FeasibleSA": "hybrid (subspace-sim QAOA seeds)",
            "ConstrainedQAOA p=2": "quantum (subspace sim)", "ConstrainedQAOA p=4": "quantum (subspace sim)",
            "X-mixer penalty QAOA p=2": "quantum (state-vector sim), negative control"}


def _e12_instances():
    out = []
    sizes = (8, 10, 12) if not QUICK else (8,)
    seeds = E12_SEEDS if not QUICK else E12_SEEDS[:1]
    for R in sizes:
        for s in seeds:
            c = generate_city(R, R, seed=s, candidate_fraction=0.3)
            pr = UHIPlanningProblem(c, interventions=DEFAULT_MIX, budget=R, min_per_district=1,
                                    cluster_bonus=0.3 if R >= 10 else 0.0)
            out.append((f"H_{R}x{R}_s{s}", "H", pr, pr.program(2)))
    for s in seeds:
        c = generate_city(8, 8, seed=s, candidate_fraction=0.3)
        pr = UHIPlanningProblem(c)
        out.append((f"C_park_8x8_s{s}", "C", pr, pr.program(2)))
    return out


def _anytime_sa_hybrid(cbp, enc, T, seed, sweeps=17000, reads=4):
    """E9 hybrid under a wall-clock budget: SA on the α=0.01 QUBO, then repair + local search."""
    br = suggest_beta_range(enc.model, RES)
    t0 = time.perf_counter()
    best, rng = [], np.random.default_rng(seed)
    while True:
        tb = time.perf_counter()
        ss = SimulatedAnnealing(num_sweeps=sweeps, num_reads=reads, beta_range=br).sample(
            enc.model, seed=int(rng.integers(2 ** 31)))
        X = postprocess(cbp, ss.samples[:, :cbp.n])
        now = time.perf_counter()
        if best and now - t0 > T:
            break
        best.append(X)
        if now - t0 + (now - tb) > T:
            break
    return np.concatenate(best), time.perf_counter() - t0


def _best_feasible(cbp, X):
    X = np.atleast_2d(X)[:, :cbp.n]
    feas = cbp.is_feasible(X)
    if not feas.any():
        return X[0], False
    f = np.where(feas, cbp.objective_values(X), np.inf)
    return X[int(np.argmin(f))], True


def e12_comparable():
    """E12: can a quantum / quantum-inspired method reach comparable benefit at matched wall-clock?"""
    from quhi.solvers.feasible_sqa import FeasibleSQA
    from quhi.solvers.feasible_tabu import TabuOnF
    from quhi.solvers.warmstart import QaoaSeededFeasibleSA

    d = out("E12_comparable")
    run_seeds = (0, 1) if not QUICK else (0,)
    rows, notes, plans_keep = [], [], {}
    for name, fam, prob, cbp in _e12_instances():
        # warm up the numba kernels on this instance (compile time is excluded from T*)
        FeasibleSA(num_sweeps=5, num_reads=1).sample_program(cbp, seed=0)
        FeasibleSQA(num_sweeps=5, num_reads=1).sample_program(cbp, seed=0)
        TabuOnF(max_iter=5, num_reads=1).sample_program(cbp, seed=0)
        t0 = time.perf_counter()
        ms = MILPSolver(time_limit=8.0).solve(cbp)
        t_milp = time.perf_counter() - t0
        certified = bool(ms.info["optimal"])
        f_opt, x_milp = ms.best_energy, ms.best
        try:
            nF = len(enumerate_feasible(cbp, max_states=20000))
            F = enumerate_feasible(cbp) if nF <= 4000 else None
        except ValueError:
            nF, F = None, None
        fsa_t = [FeasibleSA().sample_program(cbp, seed=sd).wall_time for sd in (0, 1)]
        T = max(float(np.median(fsa_t)), min(t_milp, 8.0))
        enc = cbp.to_penalty_model(penalty_weight=0.01 * cbp.to_penalty_model().penalty_weight)
        safe = cbp.to_penalty_model()
        base = {"instance": name, "family": fam, "n": cbp.n, "n_feasible": nF if nF is not None else ">20k",
                "certified": certified, "milp_gap": ms.info.get("mip_gap", np.nan), "T_star": T}
        results = []                                   # (method, seed, plan, wall, extra)
        results.append(("MILP (HiGHS, 8 s cap)", 0, x_milp, t_milp, {}))
        t0 = time.perf_counter()
        g = prob.greedy_plan()
        results.append(("Greedy planner", 0, g, time.perf_counter() - t0, {}))
        for sd in run_seeds:
            ss = FeasibleSA(num_reads=4, time_limit_s=T).sample_program(cbp, seed=sd)
            results.append(("FeasibleSA", sd, _best_feasible(cbp, ss.samples)[0], ss.wall_time, {}))
            ss = TabuOnF(num_reads=2, time_limit_s=T).sample_program(cbp, seed=sd)
            results.append(("Tabu-on-F", sd, _best_feasible(cbp, ss.samples)[0], ss.wall_time, {}))
            X, w = _anytime_sa_hybrid(cbp, enc, T, sd)
            results.append(("SA-matched α=0.01 + repair", sd, _best_feasible(cbp, X)[0], w, {}))
            ss = FeasibleSQA(num_sweeps=500, num_reads=2, trotter_slices=8, time_limit_s=T).sample_program(cbp, seed=sd)
            results.append(("FeasibleSQA", sd, _best_feasible(cbp, ss.samples)[0], ss.wall_time,
                            {"batches": ss.info["batches"]}))
            ss = QaoaSeededFeasibleSA(time_limit_s=T, greedy=g).sample_program(cbp, seed=sd, F=F)
            results.append(("QAOA-seeded FeasibleSA", sd, _best_feasible(cbp, ss.samples)[0], ss.info["wall_time"],
                            {"quantum_seed": ss.info["quantum_seed"], "frac_qaoa": ss.info["frac_qaoa"]}))
            if F is not None:
                for p in (2, 4):
                    q = ConstrainedQAOA(p=p, shots=256, restarts=1, time_limit_s=T)
                    ss = q.sample_program(cbp, seed=sd)
                    results.append((f"ConstrainedQAOA p={p}", sd, _best_feasible(cbp, ss.samples)[0], ss.wall_time,
                                    {"qaoa_p_opt_exact": ss.info["p_opt"]}))
            if safe.model.n <= 14:
                t0 = time.perf_counter()
                qs = QAOA(p=2, shots=256).sample(safe.model, seed=sd)
                x, _ = _best_feasible(cbp, qs.samples)
                results.append(("X-mixer penalty QAOA p=2", sd, x, time.perf_counter() - t0, {}))
        if F is None:
            notes.append(f"{name}: |F| = {base['n_feasible']} > 4000, ConstrainedQAOA not simulated; "
                         "QAOA-seeded FeasibleSA ran with greedy/random seeds (no quantum seed)")
        if safe.model.n > 14:
            notes.append(f"{name}: safe penalty QUBO has {safe.model.n} > 14 qubits, X-mixer control skipped")
        # scoring on exact physics
        f0 = float(prob.true_objective(np.zeros(prob.n))[0])
        cool = lambda x: f0 - float(prob.true_objective(x)[0])
        feas_cool = [cool(x) for _, _, x, _, _ in results if cbp.is_feasible(x)[0]]
        ref = cool(x_milp) if certified else max(feas_cool)
        for m, sd, x, w, extra in results:
            feas = bool(cbp.is_feasible(x)[0])
            over = (w > 1.5 * T) and not m.startswith("MILP")
            b = cool(x) / ref if (feas and not over) else 0.0
            opt = feas and certified and float(cbp.objective_values(x)[0]) <= f_opt + 1e-7 * max(1.0, abs(f_opt))
            rows.append({**base, "method": m, "kind": E12_KIND[m], "seed": sd, "wall_s": w, "over_T": over,
                         "feasible": feas, "opt": opt if certified else np.nan, "benefit_true": b, **extra})
            if sd == 0:
                plans_keep.setdefault(name, {})[m] = x
        print(f"E12 {name}: n={cbp.n} |F|={base['n_feasible']} T*={T:.2f}s certified={certified}", flush=True)
    df = pd.DataFrame(rows)
    save(df, d, "raw")
    summ = _e12_summary(df)
    save(summ, d, "summary")
    _plot_e12(df, d / "benefit_vs_time.png")
    for fam, key in (("H", "H_10x10_s100"), ("C", "C_park_8x8_s100")):
        if key in plans_keep:
            pk = plans_keep[key]
            sel = {k: pk[k] for k in ("MILP (HiGHS, 8 s cap)", "Greedy planner", "FeasibleSA", "FeasibleSQA",
                                      "QAOA-seeded FeasibleSA") if k in pk}
            prob = next(p for n_, f_, p, c_ in _e12_instances() if n_ == key)
            qp.plot_plan_transition(prob, sel, d / f"plans_{key}.png", f"E12 {key}: returned plans (sampler seed 0)")
    (d / "e12.md").write_text(
        "# E12 comparable-performance bakeoff\n\nDefinition and conventions: docs/RESULTS.md (E12). "
        "benefit_true = exact-physics cooling / MILP-plan cooling; infeasible or over 1.5 T* = 0. "
        "comparable = P(feas)=1, family-mean benefit ≥ 0.95 × best classical family mean, no run over 1.5 T*.\n\n"
        + summ.round(4).to_markdown(index=False) + "\n\n## Skipped\n\n" + ("\n".join(f"- {n}" for n in notes) or "- none") + "\n")
    print(summ.round(4).to_string())
    print("\n".join(notes))


def _e12_summary(df):
    out = []
    for fam, g in df.groupby("family", sort=False):
        means = g.groupby("method").benefit_true.mean()
        best_cl = max(means[m] for m in E12_CLASSICAL if m in means)
        best_cl_name = max((m for m in E12_CLASSICAL if m in means), key=lambda m: means[m])
        for m, h in g.groupby("method", sort=False):
            cert = h.certified.all()
            p_feas = h.feasible.mean()
            mb = h.benefit_true.mean()
            comp = bool(p_feas == 1.0 and mb >= 0.95 * best_cl and not h.over_T.any())
            out.append({"method": m, "kind": h.kind.iloc[0], "family": fam, "n_mean": h.n.mean(),
                        "n_feasible": ",".join(sorted({str(v) for v in h.n_feasible}, key=len)),
                        "instances": h.instance.nunique(), "certified": f"{h.groupby('instance').certified.first().sum()}/{h.instance.nunique()}",
                        "P_feas": p_feas, "P_opt": h.opt.astype(float).mean() if cert else np.nan,
                        "benefit_mean": mb, "benefit_min": h.benefit_true.min(), "benefit_max": h.benefit_true.max(),
                        "wall_s_mean": h.wall_s.mean(), "T_star_mean": h.T_star.mean(),
                        "best_classical": f"{best_cl_name} ({best_cl:.4f})", "comparable": "yes" if comp else "no"})
    return pd.DataFrame(out)


def _plot_e12(df, path):
    import matplotlib.pyplot as plt
    fams = list(dict.fromkeys(df.family))
    fig, axs = plt.subplots(1, len(fams), figsize=(6.2 * len(fams), 4.6))
    axs = np.atleast_1d(axs)
    for ax, fam in zip(axs, fams):
        g = df[df.family == fam].groupby("method", sort=False).agg(w=("wall_s", "mean"), b=("benefit_true", "mean"),
                                                                    k=("kind", "first"))
        for m, r in g.iterrows():
            mk = "o" if r.k.startswith("classical") else ("*" if "inspired" in r.k else "s")
            ax.scatter(r.w, r.b, marker=mk, s=70)
            ax.annotate(m, (r.w, r.b), fontsize=7, xytext=(4, 2), textcoords="offset points")
        ax.axhline(0.95 * max(g.loc[[m for m in E12_CLASSICAL if m in g.index], "b"]), color="k", ls="--", lw=1,
                   label="0.95 × best classical")
        ax.axvline(df[df.family == fam].T_star.mean(), color="gray", ls=":", lw=1, label="mean T*")
        ax.set_xscale("log")
        ax.set_xlabel("mean wall-clock per run (s)")
        ax.set_ylabel("mean benefit_true (exact physics / MILP plan)")
        ax.set_title(f"E12 family {fam}  (o classical, * quantum-inspired, s quantum sim)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- E12b
E12B_SQA = dict(num_sweeps=500, num_reads=2, trotter_slices=8)       # the E12 FeasibleSQA configuration


def _e12b_clocks():
    """Pre-declared clocks from E12 raw.csv: tight T* = median FeasibleSA wall on that instance;
    W* = proposals FeasibleSQA used in E12 (median batches x num_reads x num_sweeps x P x n)."""
    raw = pd.read_csv(ROOT / "E12_comparable" / "raw.csv")
    T = raw[raw.method == "FeasibleSA"].groupby("instance").wall_s.median()
    q = raw[raw.method == "FeasibleSQA"].groupby("instance").agg(b=("batches", "median"), n=("n", "first"))
    per_read = E12B_SQA["num_sweeps"] * E12B_SQA["trotter_slices"]
    reads = (q.b * E12B_SQA["num_reads"]).round().astype(int)
    W = reads * per_read * q.n
    seeded = raw[raw.method == "QAOA-seeded FeasibleSA"].groupby("instance").quantum_seed.all()
    return T.to_dict(), W.astype(int).to_dict(), reads.to_dict(), seeded.to_dict()


def e12b_ablation():
    """E12b: E12 with the tight clock (FeasibleSA's own wall) and with equal proposal counts."""
    from quhi.solvers.feasible_sqa import FeasibleSQA
    from quhi.solvers.feasible_tabu import TabuOnF
    from quhi.solvers.warmstart import QaoaSeededFeasibleSA

    d = out("E12b_ablation")
    Tc, Wc, Rc, seeded = _e12b_clocks()
    run_seeds = (0, 1) if not QUICK else (0,)
    rows = []
    for name, fam, prob, cbp in _e12_instances():
        FeasibleSA(num_sweeps=5, num_reads=1).sample_program(cbp, seed=0)
        FeasibleSQA(num_sweeps=5, num_reads=1).sample_program(cbp, seed=0)
        TabuOnF(max_iter=5, num_reads=1).sample_program(cbp, seed=0)
        n = cbp.n
        T, W, r_sqa = float(Tc[name]), int(Wc[name]), int(Rc[name])
        t0 = time.perf_counter()
        ms = MILPSolver(time_limit=8.0).solve(cbp)
        t_milp = time.perf_counter() - t0
        certified = bool(ms.info["optimal"])
        use_q = bool(seeded.get(name, False))
        F = enumerate_feasible(cbp) if use_q else None
        g = prob.greedy_plan()
        base = {"instance": name, "family": fam, "n": n, "certified": certified, "T_tight": T, "W_star": W}
        res = [("MILP (HiGHS, 8 s cap)", "none", 0, ms.best, t_milp, np.nan)]
        per_sweep = n
        for sd in run_seeds:
            # tight clock: E12 configurations, time limit = FeasibleSA's own wall
            ss = FeasibleSA(num_reads=4, time_limit_s=T).sample_program(cbp, seed=sd)
            res.append(("FeasibleSA", "tight", sd, _best_feasible(cbp, ss.samples)[0], ss.wall_time,
                        ss.info["flips_attempted"]))
            ss = TabuOnF(num_reads=2, time_limit_s=T).sample_program(cbp, seed=sd)
            res.append(("Tabu-on-F", "tight", sd, _best_feasible(cbp, ss.samples)[0], ss.wall_time,
                        ss.info["proposals"]))
            ss = FeasibleSQA(**E12B_SQA, time_limit_s=T).sample_program(cbp, seed=sd)
            res.append(("FeasibleSQA", "tight", sd, _best_feasible(cbp, ss.samples)[0], ss.wall_time,
                        ss.info["batches"] * E12B_SQA["num_reads"] * E12B_SQA["num_sweeps"]
                        * E12B_SQA["trotter_slices"] * per_sweep))
            if use_q:
                ss = QaoaSeededFeasibleSA(time_limit_s=T, greedy=g).sample_program(cbp, seed=sd, F=F)
                res.append(("QAOA-seeded FeasibleSA", "tight", sd, _best_feasible(cbp, ss.samples)[0],
                            ss.info["wall_time"], ss.info["sa_proposals"]))
            # equal work: every method gets W* proposals, no clock
            ss = FeasibleSA(num_reads=-(-W // (1000 * n))).sample_program(cbp, seed=sd)
            res.append(("FeasibleSA", "equal_work", sd, _best_feasible(cbp, ss.samples)[0], ss.wall_time,
                        ss.info["flips_attempted"]))
            ss = TabuOnF(num_reads=2, max_proposals=W).sample_program(cbp, seed=sd)
            res.append(("Tabu-on-F", "equal_work", sd, _best_feasible(cbp, ss.samples)[0], ss.wall_time,
                        ss.info["proposals"]))
            kw = {**E12B_SQA, "num_reads": r_sqa}
            ss = FeasibleSQA(**kw).sample_program(cbp, seed=sd)
            res.append(("FeasibleSQA", "equal_work", sd, _best_feasible(cbp, ss.samples)[0], ss.wall_time,
                        r_sqa * E12B_SQA["num_sweeps"] * E12B_SQA["trotter_slices"] * per_sweep))
            if use_q:
                ss = QaoaSeededFeasibleSA(time_limit_s=T, greedy=g, sa_proposals=W).sample_program(cbp, seed=sd, F=F)
                res.append(("QAOA-seeded FeasibleSA", "equal_work", sd, _best_feasible(cbp, ss.samples)[0],
                            ss.info["wall_time"], ss.info["sa_proposals"]))
        f0 = float(prob.true_objective(np.zeros(prob.n))[0])
        cool = lambda x: f0 - float(prob.true_objective(x)[0])
        ref = cool(ms.best)
        for m, clk, sd, x, w, prop in res:
            feas = bool(cbp.is_feasible(x)[0])
            over = clk == "tight" and w > 1.5 * T
            b = cool(x) / ref if (feas and not over) else 0.0
            opt = feas and certified and float(cbp.objective_values(x)[0]) <= ms.best_energy + 1e-7 * max(1.0, abs(ms.best_energy))
            rows.append({**base, "method": m, "clock": clk, "seed": sd, "wall_s": w, "proposals": prop,
                         "over_T": over, "feasible": feas, "opt": opt, "benefit_true": b})
        print(f"E12b {name}: n={n} T_tight={T:.2f}s W*={W:.3g} MILP {t_milp:.2f}s certified={certified}", flush=True)
    df = pd.DataFrame(rows)
    save(df, d, "raw")
    _e12b_write(df, d)


def _e12b_write(df, d):
    summ = _e12b_summary(df)
    save(summ, d, "summary")
    (d / "e12b.md").write_text(
        "# E12b equal-work + tight-clock ablation of E12\n\nClocks (pre-declared, from E12 raw.csv): "
        "tight = median FeasibleSA wall on that instance in E12; equal_work W* = proposals FeasibleSQA used in "
        "E12 (median batches × 2 reads × 500 sweeps × 8 slices × n); every method gets W* proposals "
        "(FeasibleSA: more reads of 1000 sweeps; Tabu-on-F: batches until examined moves ≥ W*; "
        "QAOA-seeded: W* SA proposals, QAOA simulation time not counted). benefit_true = exact-physics cooling / "
        "MILP-plan cooling; infeasible = 0; tight clock: wall > 1.5 T = 0. comparable_to_feasible_sa = mean "
        "benefit ≥ 0.95 × FeasibleSA mean on the same clock and the same instances; beats_feasible_sa = strictly "
        "higher. w/t/l = instances where the method's seed-mean benefit is above / equal to / below FeasibleSA's.\n\n"
        + summ.round(4).to_markdown(index=False) + "\n")
    print(summ.round(4).to_string())


def _e12b_summary(df):
    out = []
    for (fam, clk), g in df.groupby(["family", "clock"], sort=False):
        fsa = g[g.method == "FeasibleSA"]
        for m, h in g.groupby("method", sort=False):
            mb = h.benefit_true.mean()
            inst = h.groupby("instance").benefit_true.mean()
            f_same = fsa[fsa.instance.isin(inst.index)]            # FeasibleSA on the same instances
            fsa_mean = f_same.benefit_true.mean() if len(f_same) else np.nan
            fsa_inst = f_same.groupby("instance").benefit_true.mean()
            common = inst.index.intersection(fsa_inst.index)
            dlt = inst[common] - fsa_inst[common]
            if clk == "none":
                comp = beats = wtl = ""
            else:
                comp = "yes" if mb >= 0.95 * fsa_mean else "no"
                beats = "yes" if (m != "FeasibleSA" and mb > fsa_mean + 1e-12) else "no"
                wtl = f"{(dlt > 1e-9).sum()}/{(dlt.abs() <= 1e-9).sum()}/{(dlt < -1e-9).sum()}"
            out.append({"method": m, "clock": clk, "family": fam, "instances": h.instance.nunique(),
                        "P_feas": h.feasible.mean(), "P_opt": h.opt.astype(float).mean(), "benefit_mean": mb,
                        "benefit_min": h.benefit_true.min(), "wall_s": h.wall_s.mean(),
                        "proposals": h.proposals.mean(), "FeasibleSA_benefit_same_clock_same_instances": fsa_mean,
                        "comparable_to_feasible_sa": comp, "beats_feasible_sa": beats, "w/t/l_vs_FeasibleSA": wtl})
    return pd.DataFrame(out)


# ----------------------------------------------------------------- E13
def _lst_tile_city(seed=0, R=16):
    """LST-shaped 16x16 tile: compact hot core + cool edge, population on the core, candidate
    lots removed from the core (E7-style mask).  Land use / roads / districts from generate_city."""
    from quhi.uhi.city import BUILDING, CANDIDATE, City
    rng = np.random.default_rng(seed)
    base = generate_city(R, R, seed=seed, candidate_fraction=0.25, coast=False, n_existing_parks=2)
    lu = base.land_use.copy()
    rr, cc = np.meshgrid(np.linspace(0, 1, R), np.linspace(0, 1, R), indexing="ij")
    r = np.hypot(rr - 0.5, cc - 0.5)
    core = np.exp(-(r / 0.16) ** 2)                                   # compact core
    edge = np.clip((r - 0.4) / 0.3, 0, 1)                             # cool rural edge
    T0 = 29.0 + 6.0 * core - 1.5 * edge + 0.05 * rng.normal(size=(R, R))
    in_core = (core > 0.3) & (lu == CANDIDATE)
    idx = np.flatnonzero(in_core.ravel())
    keep = rng.choice(idx, size=min(2, idx.size), replace=False) if idx.size else idx
    lu.ravel()[np.setdiff1d(idx, keep)] = BUILDING                    # scarce lots in the core
    pop = np.where(lu == BUILDING, 20 + 900 * core, 0.0) * rng.uniform(0.8, 1.2, size=(R, R))
    return City(lu, T0, np.round(pop), base.district, base.cell_size, f"lst_tile_{R}x{R}_s{seed}")


def e13_tile():
    """E13: one LST-shaped tile; does a hot core with scarce lots open a classical gap?"""
    from quhi.solvers.feasible_sqa import FeasibleSQA
    from quhi.solvers.feasible_tabu import TabuOnF

    d = out("E13_tile")
    city = _lst_tile_city(0)
    qp.plot_city(city, d / "city.png")
    prob = UHIPlanningProblem(city, interventions=DEFAULT_MIX, budget=16, min_per_district=1, cluster_bonus=0.3)
    cbp = prob.program(2)
    FeasibleSA(num_sweeps=5, num_reads=1).sample_program(cbp, seed=0)
    FeasibleSQA(num_sweeps=5, num_reads=1).sample_program(cbp, seed=0)
    TabuOnF(max_iter=5, num_reads=1).sample_program(cbp, seed=0)
    try:
        nF = len(enumerate_feasible(cbp, max_states=4000))
    except ValueError:
        nF = ">4000"
    t0 = time.perf_counter()
    ms = MILPSolver(time_limit=8.0).solve(cbp)
    t_milp = time.perf_counter() - t0
    certified = bool(ms.info["optimal"])
    T = float(np.median([FeasibleSA().sample_program(cbp, seed=sd).wall_time for sd in (0, 1)]))
    enc = cbp.to_penalty_model(penalty_weight=0.01 * cbp.to_penalty_model().penalty_weight)
    res = [("MILP (HiGHS, 8 s cap)", 0, ms.best, t_milp)]
    t0 = time.perf_counter()
    g = prob.greedy_plan()
    res.append(("Greedy planner", 0, g, time.perf_counter() - t0))
    for sd in (0, 1):
        ss = FeasibleSA(num_reads=4, time_limit_s=T).sample_program(cbp, seed=sd)
        res.append(("FeasibleSA", sd, _best_feasible(cbp, ss.samples)[0], ss.wall_time))
        ss = TabuOnF(num_reads=2, time_limit_s=T).sample_program(cbp, seed=sd)
        res.append(("Tabu-on-F", sd, _best_feasible(cbp, ss.samples)[0], ss.wall_time))
        ss = FeasibleSQA(**E12B_SQA, time_limit_s=T).sample_program(cbp, seed=sd)
        res.append(("FeasibleSQA", sd, _best_feasible(cbp, ss.samples)[0], ss.wall_time))
        X, w = _anytime_sa_hybrid(cbp, enc, T, sd)
        res.append(("SA-matched α=0.01 + repair", sd, _best_feasible(cbp, X)[0], w))
    f0 = float(prob.true_objective(np.zeros(prob.n))[0])
    cool = lambda x: f0 - float(prob.true_objective(x)[0])
    ref = cool(ms.best)
    rows = []
    for m, sd, x, w in res:
        feas = bool(cbp.is_feasible(x)[0])
        over = (w > 1.5 * T) and not m.startswith("MILP")
        rep = prob.report(x)
        rows.append({"method": m, "seed": sd, "n": cbp.n, "n_feasible": nF, "budget": prob.budget,
                     "certified": certified, "T_tight": T, "wall_s": w, "over_T": over, "feasible": feas,
                     "opt": feas and float(cbp.objective_values(x)[0]) <= ms.best_energy + 1e-7 * max(1, abs(ms.best_energy)),
                     "benefit_true": cool(x) / ref if (feas and not over) else 0.0,
                     "cooling_true_C": cool(x), "exposure_temp_C": rep["exposure_temp_C"], "spend": rep["spend"]})
    df = pd.DataFrame(rows)
    save(df, d, "raw")
    keep = {m: x for m, sd, x, w in res if sd == 0}
    qp.plot_plan_transition(prob, {k: keep[k] for k in ("MILP (HiGHS, 8 s cap)", "Greedy planner", "FeasibleSA",
                                                         "FeasibleSQA", "SA-matched α=0.01 + repair")},
                            d / "plans.png", "E13 LST-shaped tile: returned plans (sampler seed 0)")
    summ = df.groupby("method", sort=False).agg(P_feas=("feasible", "mean"), P_opt=("opt", "mean"),
                                                benefit_mean=("benefit_true", "mean"), wall_s=("wall_s", "mean"))
    rr, cc = np.meshgrid(np.linspace(-0.5, 0.5, 16), np.linspace(-0.5, 0.5, 16), indexing="ij")
    core_c = int(((city.land_use == 2) & (np.exp(-(np.hypot(rr, cc) / 0.16) ** 2) > 0.3)).sum())
    milp_easy = certified and t_milp < 2.0
    g_feas = bool(cbp.is_feasible(g)[0])
    (d / "e13.md").write_text(
        f"# E13 LST-shaped tile (16x16, seed 0)\n\nn = {cbp.n}, |F| {nF}, budget {prob.budget}, equity ≥1 per "
        f"district, cluster_bonus 0.3, candidate lots left in the core: {core_c}. MILP {t_milp:.2f} s, certified "
        f"{certified}. Tight T* = {T:.2f} s (median default FeasibleSA wall). Greedy feasible: "
        f"{g_feas}.\n\nHiGHS certifies in < 2 s: {'yes' if milp_easy else 'no'}. Greedy feasible: "
        f"{'yes' if g_feas else 'no'}.\n\n" + summ.round(4).to_markdown() + "\n")
    print(summ.round(4).to_string(), "\nMILP < 2 s:", milp_easy, "greedy feasible:", g_feas, "MILP", t_milp, "T", T, "n", cbp.n, "|F|", nF)


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
        "E5": e5_hubo, "E6": e6_quantum, "E7": e7_showcase, "E8": e8_mixers, "E8b": e8b_depth, "E9": e9_hybrid,
        "E10": e10_penalty_form, "E11": e11_pubo_gap, "ANIM": e_anim, "E12": e12_comparable,
        "E12b": e12b_ablation, "E13": e13_tile}


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
