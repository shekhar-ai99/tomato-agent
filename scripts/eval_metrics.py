#!/usr/bin/env python3
"""
Stage 5b: research metrics for the diagnosis models, from the saved logits (no GPU, no images needed).

For every run (backbone x seed), test set (pv_test, pd_test) and temperature (raw, pv, pd) it computes:
  accuracy, balanced accuracy, macro / weighted precision, recall, F1,
  ROC-AUC (one-vs-rest: macro, weighted, micro), average precision (PR-AUC),
  Brier score (multi-class, lower is better), ECE (15 bins), NLL,
  and 95% bootstrap confidence intervals (resampling test photos) for accuracy, macro-F1, macro-AUC, Brier, ECE.
Classes with no test photos in a set (mosaic_virus in PlantVillage) are left out of macro averages.

Everything is written as files, so nothing is lost when the session ends:
  results/metrics/metrics_per_run.csv           one row per run x set x temperature (with bootstrap CIs)
  results/metrics/metrics_summary.csv / .md     mean +- SD over seeds per backbone (paper tables)
  results/metrics/per_class_metrics.csv         precision / recall / F1 / support / AUC / AP per class
  results/metrics/confusion_<bb>_<set>.csv      confusion matrix summed over seeds (rows = truth)
  results/metrics/roc_<bb>_<set>.csv            ROC curve points per class (class, fpr, tpr), first seed
  results/metrics/metrics.json                  everything above in one file
  results/metrics/fig5_roc_*.png, fig6_confusion_*.png, fig7_per_class_f1_*.png   figures (skip with --no-figs)

Usage:  python scripts/eval_metrics.py --runs runs --results results
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import rankdata

from tomato_agent.calibration import brier, ece, nll, softmax
from tomato_agent.classes import CLASSES
from tomato_agent.io_utils import read_json, run_dirs, write_csv, write_json

SETS = ("pv_test", "pd_test")
K = len(CLASSES)


# ----------------------------------------------------------------------------- metric helpers
def auc_binary(y, s):
    """ROC-AUC by the rank-sum (Mann-Whitney) formula, ties handled. NaN if one class is missing."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_points(y, s, max_points=200):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.sum() == 0 or (~y).sum() == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tps, fps = np.cumsum(ys), np.cumsum(~ys)
    last = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]  # one point per distinct score
    tpr = np.r_[0.0, tps[last] / y.sum()]
    fpr = np.r_[0.0, fps[last] / (~y).sum()]
    if len(fpr) > max_points:
        keep = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[keep], tpr[keep]
    return fpr, tpr


def average_precision(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    return float((precision * ys).sum() / y.sum())


def confusion(labels, pred):
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (labels, pred), 1)
    return cm


def prf(cm):
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    predicted = cm.sum(0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros(K), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(K), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(K), where=(precision + recall) > 0)
    return precision, recall, f1, support


def core_metrics(probs, labels):
    """Metrics of one probability matrix. Classes without any test photo are excluded from the macro averages."""
    pred = probs.argmax(1)
    cm = confusion(labels, pred)
    precision, recall, f1, support = prf(cm)
    present = support > 0
    aucs = np.array([auc_binary(labels == k, probs[:, k]) if present[k] else np.nan for k in range(K)])
    aps = np.array([average_precision(labels == k, probs[:, k]) if present[k] else np.nan for k in range(K)])
    w = support / support.sum()
    ok = present & ~np.isnan(aucs)
    micro = auc_binary(np.eye(K, dtype=bool)[labels].ravel(), probs.ravel())
    return {
        "n": int(len(labels)), "accuracy": float((pred == labels).mean()),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_precision": float(precision[present].mean()), "macro_recall": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()), "weighted_f1": float((f1 * w).sum()),
        "auc_macro_ovr": float(aucs[ok].mean()) if ok.any() else float("nan"), "auc_weighted_ovr": float((aucs[ok] * w[ok]).sum() / w[ok].sum()),
        "auc_micro": micro, "ap_macro": nanmean(aps),
        "brier": brier(probs, labels), "ece": ece(probs, labels), "nll": float(-np.log(
            np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean()),
    }, cm, precision, recall, f1, support, aucs, aps


CI_METRICS = ("accuracy", "macro_f1", "auc_macro_ovr", "brier", "ece")


def bootstrap_ci(probs, labels, n_boot, seed=0):
    """95% percentile bootstrap interval, resampling test photos."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    draws = {m: [] for m in CI_METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = core_metrics(probs[idx], labels[idx])[0]
        for k in CI_METRICS:
            draws[k].append(m[k])
    out = {}
    for k, v in draws.items():
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        out[f"{k}_ci_lo"], out[f"{k}_ci_hi"] = float(lo), float(hi)
    return out


# ----------------------------------------------------------------------------- plotting
def plot_roc(path, labels, probs, title, aucs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for k in range(K):
        if (labels == k).sum() == 0:
            continue
        fpr, tpr = roc_points(labels == k, probs[:, k])
        ax.plot(fpr, tpr, lw=1.4, label=f"{CLASSES[k]} (AUC {aucs[k]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title, xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(path, cm, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(K), yticks=range(K), xlabel="Predicted", ylabel="True", title=title)
    ax.set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(CLASSES, fontsize=7)
    for i in range(K):
        for j in range(K):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_f1_bars(path, per_bb_f1, set_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bbs = list(per_bb_f1)
    x = np.arange(K)
    width = 0.8 / max(len(bbs), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, bb in enumerate(bbs):
        ax.bar(x + i * width, per_bb_f1[bb], width, label=bb)
    ax.set_xticks(x + width * (len(bbs) - 1) / 2)
    ax.set_xticklabels(CLASSES, rotation=40, ha="right", fontsize=8)
    ax.set(ylabel="F1 (mean over seeds)", ylim=(0, 1.02), title=f"Per-class F1 on {set_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def nanmean(xs):
    xs = np.asarray(xs, float)
    return float(np.nanmean(xs)) if np.isfinite(xs).any() else float("nan")


def ms(xs):
    xs = np.asarray(xs, float)
    return nanmean(xs), (float(np.nanstd(xs, ddof=1)) if np.isfinite(xs).sum() > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--bootstrap", type=int, default=500, help="bootstrap resamples per run (0 = no intervals)")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()

    out = Path(args.results) / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    runs = run_dirs(args.runs)
    if not runs:
        sys.exit(f"No trained runs in {args.runs} (need best.pt and logits_*.npz).")

    per_run, per_class, cm_sum, first_seed = [], [], defaultdict(lambda: np.zeros((K, K), np.int64)), {}
    f1_by_bb = {s: defaultdict(list) for s in SETS}
    for rd in runs:
        bb, seed = rd.parent.name, int(rd.name[4:])
        cal = read_json(rd / "calibration.json") if (rd / "calibration.json").exists() else {}
        temps = {"raw": 1.0, "pv": cal.get("T_pv", 1.0), "pd": cal.get("T_pd", 1.0)}
        for s in SETS:
            f = rd / f"logits_{s}.npz"
            if not f.exists():
                continue
            d = np.load(f, allow_pickle=True)
            logits, labels = d["logits"], d["labels"].astype(int)
            for tname, T in temps.items():
                probs = softmax(logits, T)
                m, cm, precision, recall, f1, support, aucs, aps = core_metrics(probs, labels)
                row = {"backbone": bb, "seed": seed, "set": s, "temperature": tname, "T": T, **m}
                if args.bootstrap:
                    row.update(bootstrap_ci(probs, labels, args.bootstrap))
                per_run.append(row)
                field_t = "pd" if s == "pd_test" else "pv"  # the temperature used for the figures and per-class tables
                if tname == field_t:
                    for k in range(K):
                        per_class.append({"backbone": bb, "seed": seed, "set": s, "class": CLASSES[k],
                                          "support": int(support[k]), "precision": precision[k], "recall": recall[k],
                                          "f1": f1[k], "auc_ovr": aucs[k], "average_precision": aps[k]})
                    cm_sum[(bb, s)] += cm
                    f1_by_bb[s][bb].append(f1)
                    if (bb, s) not in first_seed:
                        first_seed[(bb, s)] = (labels, probs, aucs, seed)
        print(f"{bb} seed{seed}: done", flush=True)

    write_csv(out / "metrics_per_run.csv", per_run)

    # mean +- SD over seeds
    groups = defaultdict(list)
    for r in per_run:
        groups[(r["backbone"], r["set"], r["temperature"])].append(r)
    cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "auc_macro_ovr", "auc_micro", "ap_macro",
            "brier", "ece", "nll"]
    summary = []
    for (bb, s, t), rs in sorted(groups.items()):
        row = {"backbone": bb, "set": s, "temperature": t, "n_seeds": len(rs), "n_photos": rs[0]["n"]}
        for c in cols:
            mean, sd = ms([r[c] for r in rs])
            row[c], row[c + "_sd"] = mean, sd
        if args.bootstrap:
            for c in CI_METRICS:
                row[c + "_ci_lo"] = nanmean([r[c + "_ci_lo"] for r in rs])
                row[c + "_ci_hi"] = nanmean([r[c + "_ci_hi"] for r in rs])
        summary.append(row)
    write_csv(out / "metrics_summary.csv", summary)

    # per-class table averaged over seeds
    pc = defaultdict(list)
    for r in per_class:
        pc[(r["backbone"], r["set"], r["class"])].append(r)
    pc_rows = []
    for (bb, s, c), rs in sorted(pc.items()):
        pc_rows.append({"backbone": bb, "set": s, "class": c, "support": rs[0]["support"],
                        **{k: nanmean([r[k] for r in rs]) for k in ("precision", "recall", "f1", "auc_ovr",
                                                                                  "average_precision")}})
    write_csv(out / "per_class_metrics.csv", pc_rows)

    for (bb, s), cm in cm_sum.items():
        write_csv(out / f"confusion_{bb}_{s}.csv",
                  [{"true \\ predicted": CLASSES[i], **{CLASSES[j]: int(cm[i, j]) for j in range(K)}} for i in range(K)])
    for (bb, s), (labels, probs, aucs, seed) in first_seed.items():
        rows = []
        for k in range(K):
            if (labels == k).sum():
                fpr, tpr = roc_points(labels == k, probs[:, k])
                rows += [{"class": CLASSES[k], "fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr)]
        write_csv(out / f"roc_{bb}_{s}.csv", rows)

    # markdown tables
    def cell(r, c, pct=False):
        v, sd = r[c], r[c + "_sd"]
        f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
        txt = f"{f(v)} ± {f(sd)}" if r["n_seeds"] > 1 else f(v)
        if args.bootstrap and c + "_ci_lo" in r:
            txt += f" [{f(r[c + '_ci_lo'])}, {f(r[c + '_ci_hi'])}]"
        return txt

    md = ["# Diagnosis metrics (mean ± SD over seeds; [95% bootstrap CI over test photos])\n",
          "Accuracy, balanced accuracy and F1 in %, ROC-AUC (one-vs-rest) and PR-AUC as 0-1, "
          "Brier and NLL lower is better. `raw` = no temperature, `pv` = fitted on PlantVillage val, "
          "`pd` = fitted on the PlantDoc calibration hold-out.\n"]
    for s in SETS:
        sub = [r for r in summary if r["set"] == s]
        if not sub:
            continue
        md += [f"\n## {s} ({sub[0]['n_photos']} photos)\n",
               "| Backbone | Temp | Accuracy % | Balanced acc % | Macro-F1 % | Macro ROC-AUC | Micro ROC-AUC | "
               "Macro PR-AUC | Brier | ECE | NLL |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            md.append(f"| {r['backbone']} | {r['temperature']} | {cell(r, 'accuracy', True)} | "
                      f"{cell(r, 'balanced_accuracy', True)} | {cell(r, 'macro_f1', True)} | {cell(r, 'auc_macro_ovr')} | "
                      f"{cell(r, 'auc_micro')} | {cell(r, 'ap_macro')} | {cell(r, 'brier')} | {cell(r, 'ece')} | "
                      f"{cell(r, 'nll')} |")
    md += ["\n## Per-class metrics (field temperature, mean over seeds)\n"]
    for s in SETS:
        for bb in sorted({r["backbone"] for r in pc_rows}):
            sub = [r for r in pc_rows if r["set"] == s and r["backbone"] == bb]
            if not sub:
                continue
            md += [f"\n**{bb} on {s}**\n", "| Class | Support | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
                   "|---|---|---|---|---|---|---|"]
            md += [f"| {r['class']} | {r['support']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                   f"{r['auc_ovr']:.3f} | {r['average_precision']:.3f} |" for r in sub]
    (out / "metrics_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(out / "metrics.json", {"summary": summary, "per_class": pc_rows, "per_run": per_run})

    if not args.no_figs:
        try:
            for (bb, s), (labels, probs, aucs, seed) in first_seed.items():
                plot_roc(out / f"fig5_roc_{bb}_{s}.png", labels, probs,
                         f"One-vs-rest ROC, {bb} seed{seed}, {s}", aucs)
            for (bb, s), cm in cm_sum.items():
                plot_confusion(out / f"fig6_confusion_{bb}_{s}.png", cm, f"{bb} on {s} (summed over seeds)")
            for s in SETS:
                if f1_by_bb[s]:
                    plot_f1_bars(out / f"fig7_per_class_f1_{s}.png", {bb: np.mean(v, 0) for bb, v in f1_by_bb[s].items()}, s)
        except Exception as e:  # figures are optional; the tables and CSVs above are already saved
            print(f"figures skipped: {type(e).__name__}: {e}")
    print(f"Wrote metrics to {out}/ (metrics_summary.md, metrics_summary.csv, per_class_metrics.csv, confusion_*.csv, "
          f"roc_*.csv, figures)")


if __name__ == "__main__":
    main()
