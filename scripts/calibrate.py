#!/usr/bin/env python3
"""
Stage 4: temperature scaling.

For every run, fit T_pv on PlantVillage val logits and T_pd on the PlantDoc calibration hold-out
(field-calibrated variant). Report NLL / ECE / Brier before and after on the test sets.

Outputs:
  runs/<backbone>/seed<k>/calibration.json
  results/calibration.csv            one row per run x test set x temperature
  results/reliability_bins.csv       for the reliability diagram (paper Figure 4)

Usage:  python scripts/calibrate.py --runs runs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tomato_agent.calibration import calib_summary, fit_temperature, reliability_bins, softmax
from tomato_agent.io_utils import run_dirs, write_csv, write_json


def load(run, name):
    z = np.load(run / f"logits_{name}.npz", allow_pickle=True)
    return z["logits"], z["labels"].astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    rows, bins = [], []
    dirs = run_dirs(args.runs)
    if not dirs:
        sys.exit(f"No runs under {args.runs}/")
    for run in dirs:
        bb, seed = run.parent.name, int(run.name[4:])
        T_pv = fit_temperature(*load(run, "pv_val"))
        T_pd = fit_temperature(*load(run, "pd_calib"))
        out = {"T_pv": T_pv, "T_pd": T_pd}
        for test in ("pv_test", "pd_test"):
            z, y = load(run, test)
            for tname, T in (("pv", T_pv), ("pd", T_pd)):
                s = calib_summary(z, y, T)
                out[f"{test}_T{tname}"] = s
                rows.append({"backbone": bb, "seed": seed, "test_set": test, "temperature": tname, "T": T, **s})
            for tag, T in (("before", 1.0), ("after_Tpv", T_pv), ("after_Tpd", T_pd)):
                for b in reliability_bins(softmax(z, T), y):
                    bins.append({"backbone": bb, "seed": seed, "test_set": test, "calibration": tag, **b})
        write_json(run / "calibration.json", out)
        print(f"{bb} seed{seed}: T_pv={T_pv:.3f} T_pd={T_pd:.3f} | PlantDoc ECE "
              f"{out['pd_test_Tpv']['ece_before']:.3f} -> {out['pd_test_Tpv']['ece_after']:.3f} (T_pv), "
              f"{out['pd_test_Tpd']['ece_after']:.3f} (T_pd)")
    write_csv(Path(args.results) / "calibration.csv", rows)
    write_csv(Path(args.results) / "reliability_bins.csv", bins)
    print(f"Wrote calibration.json per run, {args.results}/calibration.csv, {args.results}/reliability_bins.csv")


if __name__ == "__main__":
    main()
