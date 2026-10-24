#!/usr/bin/env python3
"""
Stage 7 / E4 + E5: ablations, cost ratios and expert workload (RQ4).

E4  remove one agent at a time (thresholds re-selected for each variant with the same cost rule):
      full pipeline | without quality agent | without calibration (T = 1) | without retake action (R = 0)
E5  the full pipeline at several C_err : C_def ratios: operating point and expert load per 1,000 cases.

Evaluated on --eval-set (default PlantDoc test) under --scenario (default mixed degradations).
Outputs: results/e4_ablation_raw.csv, results/e5_cost_workload_raw.csv

Usage:  python scripts/ablate.py --runs runs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tomato_agent.experiments import MODES, SetData, choose_gate, evaluate_proposed, parse_costs
from tomato_agent.io_utils import read_json, run_dirs, write_csv

VARIANTS = {  # name -> (temperature override, retake budget, use quality)
    "full": (None, None, True),
    "no_quality_agent": (None, None, False),
    "no_calibration": ("none", None, True),
    "no_retake": (None, 0, True),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--results", default="results")
    ap.add_argument("--cost", default="10:1", help="cost ratio for E4")
    ap.add_argument("--sweep", nargs="+", default=["20:1", "10:1", "5:1", "2:1"], help="cost ratios for E5")
    ap.add_argument("--c-ret", type=float, default=0.1)
    ap.add_argument("--retake-budget", type=int, default=1)
    ap.add_argument("--temperature", default="pv")
    ap.add_argument("--select-set", default="pv_val")
    ap.add_argument("--eval-set", default="pd_test")
    ap.add_argument("--scenario", default="mixed")
    args = ap.parse_args()

    q = read_json(Path(args.artifacts) / "quality_thresholds.json")
    th, level = q["thresholds"], q["levels"][0]
    qdir = Path(args.artifacts) / "quality_features"
    main_cost = parse_costs([args.cost], args.c_ret)[0]
    e4, e5 = [], []
    for run in run_dirs(args.runs):
        bb, seed = run.parent.name, int(run.name[4:])
        cache = {}
        for name, (T, R, use_q) in VARIANTS.items():
            T = T or args.temperature
            R = args.retake_budget if R is None else R
            if T not in cache:
                cache[T] = (SetData(run, args.select_set, qdir, th, T), SetData(run, args.eval_set, qdir, th, T))
            sel, ev = cache[T]
            cfg, _ = choose_gate(sel, main_cost, args.select_set, level, R, use_q)
            for mode in MODES:
                row, _, _ = evaluate_proposed(ev, cfg, args.scenario, level, mode, args.eval_set, main_cost)
                e4.append({"variant": name, "mode": mode, "backbone": bb, "seed": seed, "cost": main_cost.label,
                           "set": args.eval_set, "scenario": args.scenario, "tau_accept": cfg.tau_accept,
                           "tau_defer": cfg.tau_defer, **row})
        sel, ev = cache[args.temperature]
        for c in parse_costs(args.sweep, args.c_ret):
            cfg, _ = choose_gate(sel, c, args.select_set, level, args.retake_budget, True)
            for mode in MODES:
                row, _, _ = evaluate_proposed(ev, cfg, args.scenario, level, mode, args.eval_set, c)
                e5.append({"cost": c.label, "mode": mode, "backbone": bb, "seed": seed, "set": args.eval_set,
                           "scenario": args.scenario, "tau_accept": cfg.tau_accept, "tau_defer": cfg.tau_defer,
                           "expert_cases_per_1000": 1000 * row["deferral_rate"],
                           "retakes_per_1000": 1000 * row["retakes_per_case"],
                           "wrong_auto_answers_per_1000": 1000 * row["auto_error_rate"], **row})
        print(f"{bb} seed{seed}: done")
    out = Path(args.results)
    write_csv(out / "e4_ablation_raw.csv", e4)
    write_csv(out / "e5_cost_workload_raw.csv", e5)
    print(f"Wrote {out}/e4_ablation_raw.csv and e5_cost_workload_raw.csv")


if __name__ == "__main__":
    main()
