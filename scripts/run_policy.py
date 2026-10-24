#!/usr/bin/env python3
"""
Stage 6 / E2 + E3: run the agent policy.

For every seed and backbone:
  * choose (tau_accept, tau_defer) with the cost rule on a selection set with mixed degradations and pessimistic
    retakes, for each cost ratio and temperature variant. Default: the lab temperature (pv) is selected on
    PlantVillage val, the field temperature (pd) on the PlantDoc calibration hold-out (never on a test set);
  * E2 (RQ2): static accuracy vs quality check + retake, per degradation (no deferral);
  * E3 (RQ3): B1 static, B2 ensemble, B3 threshold only, and the full pipeline, on clean and mixed
    test conditions: accuracy, accuracy at 90/80/70% coverage, ECE, AURC, and the cost-rule
    operating point (coverage, deferral rate, retake rate, system accuracy with the oracle expert).

Outputs (results/):
  thresholds.json                chosen thresholds per run, plus a recommended set per cost ratio
  e2_quality_retake_raw.csv      one row per run x set x degradation x retake mode
  e3_reliability_raw.csv         one row per run x set x scenario x method x temperature x cost
  curves_raw.csv                 risk on a fixed coverage grid (paper Figure 3)

Usage:  python scripts/run_policy.py --runs runs --costs 10:1 5:1
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tomato_agent.degrade import DEGRADATIONS
from tomato_agent.experiments import (MODES, SetData, choose_gate, curve_rows, evaluate_ensemble,
                                      evaluate_proposed, evaluate_threshold_only, parse_costs, quality_only)
from tomato_agent.io_utils import read_json, run_dirs, write_csv, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--results", default="results")
    ap.add_argument("--costs", nargs="+", default=["10:1", "5:1"], help="C_err:C_def ratios")
    ap.add_argument("--c-ret", type=float, default=0.1, help="cost of one retake, in units of C_def")
    ap.add_argument("--retake-budget", type=int, default=1)
    ap.add_argument("--temperatures", nargs="+", default=["pv", "pd"], help="pv = PlantVillage-fitted, pd = field-fitted")
    ap.add_argument("--select-set", default=None, choices=["pv_val", "pd_calib"],
                    help="use this one set for every temperature variant (old behaviour); default: pv->pv_val, pd->pd_calib")
    ap.add_argument("--select-mode", default="pess", choices=MODES)
    ap.add_argument("--eval-sets", nargs="+", default=["pv_test", "pd_test"])
    ap.add_argument("--e3-scenarios", nargs="+", default=["none", "mixed"])
    ap.add_argument("--main-backbone", default=None, help="backbone whose thresholds are 'recommended' (default: first)")
    args = ap.parse_args()

    qcfg = read_json(Path(args.artifacts) / "quality_thresholds.json")
    th, level = qcfg["thresholds"], qcfg["levels"][0]
    qdir = Path(args.artifacts) / "quality_features"
    costs = parse_costs(args.costs, args.c_ret)
    R = args.retake_budget
    dirs = run_dirs(args.runs)
    if not dirs:
        sys.exit("no runs")
    by_seed = defaultdict(list)
    for d in dirs:
        by_seed[int(d.name[4:])].append(d)
    main_bb = args.main_backbone or sorted({d.parent.name for d in dirs})[0]
    select_for = {"pv": "pv_val", "pd": "pd_calib", "none": "pv_val"}   # which hold-out picks the thresholds, per temperature
    if args.select_set:
        select_for = {t: args.select_set for t in select_for}

    e2, e3, curves, chosen = [], [], [], {}
    for seed, rds in sorted(by_seed.items()):
        for T in args.temperatures:
            sel_set = select_for[T]
            sel = {d.parent.name: SetData(d, sel_set, qdir, th, T) for d in rds}
            ev = {s: {d.parent.name: SetData(d, s, qdir, th, T) for d in rds} for s in args.eval_sets}
            for bb in sel:
                for c in costs:
                    cfg, m = choose_gate(sel[bb], c, sel_set, level, R, True, args.select_mode)
                    cfg0, _ = choose_gate(sel[bb], c, sel_set, level, 0, False, args.select_mode)
                    chosen[f"{bb}/seed{seed}/T{T}/{c.label}"] = {
                        "proposed": cfg.to_dict(), "threshold_only": cfg0.to_dict(), "selection_metrics": m,
                        "selected_on": sel_set}
                    for s in args.eval_sets:
                        data = ev[s][bb]
                        for scen in args.e3_scenarios:
                            base = {"set": s, "scenario": scen, "backbone": bb, "seed": seed, "temperature": T, "cost": c.label}
                            row, sc, cr = evaluate_threshold_only(data, cfg0, scen, level, s, c)
                            e3.append({**base, "method": "B3_threshold_only", "mode": "-", **row})
                            if c is costs[0]:
                                curves += curve_rows(sc, cr, {**base, "method": "B3_threshold_only", "mode": "-"})
                            for mode in MODES:
                                row, sc, cr = evaluate_proposed(data, cfg, scen, level, mode, s, c)
                                e3.append({**base, "method": "proposed", "mode": mode, **row,
                                           "tau_accept": cfg.tau_accept, "tau_defer": cfg.tau_defer})
                                if c is costs[0]:
                                    curves += curve_rows(sc, cr, {**base, "method": "proposed", "mode": mode})
            # cost-independent baselines
            for s in args.eval_sets:
                for scen in args.e3_scenarios:
                    base = {"set": s, "scenario": scen, "seed": seed, "temperature": T, "cost": "-", "mode": "-"}
                    for bb, data in ev[s].items():
                        if T == args.temperatures[0]:
                            row, sc, cr = evaluate_threshold_only(data, None, scen, level, s, None, which="raw_probs")
                            e3.append({**base, "temperature": "none", "backbone": bb, "method": "B1_static", **row})
                            curves += curve_rows(sc, cr, {**base, "temperature": "none", "backbone": bb, "method": "B1_static"})
                    if len(ev[s]) > 1:
                        row, sc, cr = evaluate_ensemble(list(ev[s].values()), scen, level, s)
                        e3.append({**base, "backbone": "+".join(sorted(ev[s])), "method": "B2_ensemble", **row})
                        curves += curve_rows(sc, cr, {**base, "backbone": "ensemble", "method": "B2_ensemble"})
            if T == args.temperatures[0]:
                for s in args.eval_sets:
                    for bb, data in ev[s].items():
                        for deg in ["none"] + DEGRADATIONS:
                            for mode in MODES:
                                e2.append({"set": s, "degradation": deg, "mode": mode, "backbone": bb, "seed": seed,
                                           **quality_only(data, deg, level, mode, R, s)})
        print(f"seed {seed}: done ({len(rds)} backbones)")

    rec = {}
    for c in costs:
        ks = [k for k in chosen if k.startswith(f"{main_bb}/") and k.endswith(f"/T{args.temperatures[0]}/{c.label}")]
        if ks:
            rec[c.label] = {"backbone": main_bb, "temperature": args.temperatures[0], "retake_budget": R,
                            "tau_accept": float(np.median([chosen[k]["proposed"]["tau_accept"] for k in ks])),
                            "tau_defer": float(np.median([chosen[k]["proposed"]["tau_defer"] for k in ks])),
                            "note": "median over seeds; used by the demo app and exported to the phone"}
    out = Path(args.results)
    write_json(out / "thresholds.json", {"recommended": rec, "per_run": chosen,
                                         "selected_on": {t: select_for[t] for t in args.temperatures}, "select_mode": args.select_mode,
                                         "c_ret": args.c_ret, "quality_level": level})
    write_csv(out / "e2_quality_retake_raw.csv", e2)
    write_csv(out / "e3_reliability_raw.csv", e3)
    write_csv(out / "curves_raw.csv", curves)
    print(f"Wrote {out}/thresholds.json, e2_quality_retake_raw.csv, e3_reliability_raw.csv, curves_raw.csv")
    for k, v in rec.items():
        print(f"  recommended {k}: tau_accept={v['tau_accept']:.3f} tau_defer={v['tau_defer']:.3f} ({main_bb})")


if __name__ == "__main__":
    main()
