#!/usr/bin/env python3
"""
Stage 5c: compare the backbones with each other (run scripts/eval_metrics.py first).

Writes to results/metrics/:
  model_comparison.csv / .md    ranking of the backbones per test set on the field temperature, best value in bold
  model_pairwise_tests.csv      paired t-test and Wilcoxon between every pair of backbones across seeds
  fig8_model_comparison_<set>.png        accuracy, macro-F1, macro ROC-AUC, Brier, ECE as bars with SD over seeds
  fig9_macro_roc_<set>.png               macro-average one-vs-rest ROC curve of every backbone (first seed)
  fig10_reliability_<set>.png            reliability diagram of every backbone (first seed)
  macro_roc_<set>.csv                    the numbers behind fig9

Usage:  python scripts/compare_models.py --runs runs --results results
"""
import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tomato_agent.calibration import reliability_bins, softmax
from tomato_agent.classes import CLASSES
from tomato_agent.io_utils import read_json, run_dirs, write_csv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import K, SETS, roc_points  # noqa: E402

TRAPEZOID = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.0 renamed trapz
METRICS = [("accuracy", "Accuracy", True, True), ("macro_f1", "Macro-F1", True, True),
           ("auc_macro_ovr", "Macro ROC-AUC", True, False), ("ap_macro", "Macro PR-AUC", True, False),
           ("brier", "Brier", False, False), ("ece", "ECE", False, False), ("nll", "NLL", False, False)]
# (column, label, higher is better, show as %)


def field_temp(s):
    return "pd" if s == "pd_test" else "pv"


def fmt(v, sd, pct):
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(v)} ± {f(sd)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    mdir = Path(args.results) / "metrics"
    f = mdir / "metrics_per_run.csv"
    if not f.exists():
        sys.exit("Run scripts/eval_metrics.py first (results/metrics/metrics_per_run.csv is missing).")
    df = pd.read_csv(f)

    md = ["# Model comparison (mean ± SD over seeds, field temperature; best value in bold)\n"]
    rows, tests = [], []
    for s in SETS:
        d = df[(df["set"] == s) & (df["temperature"] == field_temp(s))]
        if d.empty:
            continue
        bbs = sorted(d["backbone"].unique())
        agg = d.groupby("backbone")
        md += [f"\n## {s} ({int(d['n'].iloc[0])} photos, {d.groupby('backbone')['seed'].nunique().min()} seed(s))\n",
               "| Backbone | " + " | ".join(label for _, label, _, _ in METRICS) + " |",
               "|---|" + "---|" * len(METRICS)]
        best = {}
        for col, _, hi, _ in METRICS:
            means = agg[col].mean()
            best[col] = means.idxmax() if hi else means.idxmin()
        for bb in bbs:
            g = d[d["backbone"] == bb]
            cells, row = [], {"set": s, "backbone": bb, "n_seeds": len(g)}
            for col, _, _, pct in METRICS:
                mean, sd = g[col].mean(), (g[col].std(ddof=1) if len(g) > 1 else 0.0)
                row[col], row[col + "_sd"] = float(mean), float(sd)
                txt = fmt(mean, sd, pct)
                cells.append(f"**{txt}**" if best[col] == bb and len(bbs) > 1 else txt)
            rows.append(row)
            md.append(f"| {bb} | " + " | ".join(cells) + " |")
        # paired tests between every pair of backbones across seeds
        from scipy.stats import ttest_rel, wilcoxon
        piv = {col: d.pivot(index="seed", columns="backbone", values=col) for col, *_ in METRICS}
        md += ["\n**Paired tests across seeds** (diff = first minus second; needs 2+ seeds, with 5 seeds the smallest "
               "possible Wilcoxon p is 0.0625):\n", "| Comparison | Metric | n | Mean diff | t-test p | Wilcoxon p |",
               "|---|---|---|---|---|---|"]
        for a, b in itertools.combinations(bbs, 2):
            for col, label, _, _ in METRICS:
                x = piv[col][[a, b]].dropna()
                diff = (x[a] - x[b]).to_numpy()
                tp = wp = float("nan")
                if len(x) >= 2 and np.ptp(diff) > 0:
                    tp = float(ttest_rel(x[a], x[b]).pvalue)
                    try:
                        wp = float(wilcoxon(x[a], x[b]).pvalue)
                    except ValueError:
                        pass
                tests.append({"set": s, "a": a, "b": b, "metric": col, "n": len(x),
                              "mean_diff": float(diff.mean()) if len(diff) else float("nan"), "t_p": tp, "wilcoxon_p": wp})
                md.append(f"| {a} vs {b} | {label} | {len(x)} | {tests[-1]['mean_diff']:+.4f} | {tp:.4f} | {wp:.4f} |")
    write_csv(mdir / "model_comparison.csv", rows)
    write_csv(mdir / "model_pairwise_tests.csv", tests)
    (mdir / "model_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # ---- curves from the saved logits of the first seed of each backbone
    curves = defaultdict(dict)  # set -> backbone -> (macro roc fpr, tpr, macro auc, bins)
    for rd in run_dirs(args.runs):
        bb = rd.parent.name
        cal = read_json(rd / "calibration.json") if (rd / "calibration.json").exists() else {}
        for s in SETS:
            fl = rd / f"logits_{s}.npz"
            if bb in curves[s] or not fl.exists():
                continue
            d = np.load(fl, allow_pickle=True)
            probs, labels = softmax(d["logits"], cal.get("T_" + field_temp(s), 1.0)), d["labels"].astype(int)
            grid = np.linspace(0, 1, 201)
            tprs, aucs = [], []
            for k in range(K):
                if (labels == k).sum() == 0 or (labels != k).sum() == 0:
                    continue
                fpr, tpr = roc_points(labels == k, probs[:, k], max_points=10 ** 9)
                tprs.append(np.interp(grid, fpr, tpr))
                aucs.append(float(TRAPEZOID(tprs[-1], grid)))
            curves[s][bb] = (grid, np.mean(tprs, 0), float(np.mean(aucs)), reliability_bins(probs, labels))
    for s, per in curves.items():
        write_csv(mdir / f"macro_roc_{s}.csv", [{"backbone": bb, "fpr": float(a), "tpr": float(b)}
                                                 for bb, (g, t, _, _) in per.items() for a, b in zip(g, t)])

    if args.no_figs:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for s in SETS:
            sub = [r for r in rows if r["set"] == s]
            if not sub:
                continue
            fig, axes = plt.subplots(1, len(METRICS) - 2, figsize=(3 * (len(METRICS) - 2), 3.6))
            for ax, (col, label, hi, pct) in zip(axes, [m for m in METRICS if m[0] not in ("ap_macro", "nll")]):
                vals = [(100 if pct else 1) * r[col] for r in sub]
                errs = [(100 if pct else 1) * r[col + "_sd"] for r in sub]
                ax.bar([r["backbone"] for r in sub], vals, yerr=errs, capsize=3, color="#3b7d4f")
                ax.set_title(label + ("  (higher better)" if hi else "  (lower better)"), fontsize=8)
                ax.tick_params(axis="x", rotation=40, labelsize=7)
            fig.suptitle(f"Backbone comparison on {s}", fontsize=10)
            fig.tight_layout()
            fig.savefig(mdir / f"fig8_model_comparison_{s}.png", dpi=160)
            plt.close(fig)
            per = curves.get(s, {})
            if per:
                fig, ax = plt.subplots(figsize=(5.5, 5))
                for bb, (g, t, auc, _) in per.items():
                    ax.plot(g, t, lw=1.6, label=f"{bb} (AUC {auc:.3f})")
                ax.plot([0, 1], [0, 1], "k--", lw=0.8)
                ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01),
                       title=f"Macro-average one-vs-rest ROC on {s}")
                ax.legend(fontsize=8, loc="lower right")
                fig.tight_layout()
                fig.savefig(mdir / f"fig9_macro_roc_{s}.png", dpi=160)
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
                for bb, (_, _, _, bins) in per.items():
                    pts = [(b["conf"], b["acc"]) for b in bins if b["count"] > 0]
                    ax.plot(*zip(*pts), marker="o", ms=3, lw=1.2, label=bb)
                ax.set(xlabel="Confidence", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1), title=f"Reliability on {s}")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(mdir / f"fig10_reliability_{s}.png", dpi=160)
                plt.close(fig)
    except Exception as e:  # figures are optional; the tables are already written
        print(f"figures skipped: {type(e).__name__}: {e}")
    print(f"\nWrote comparison to {mdir}/")


if __name__ == "__main__":
    main()
