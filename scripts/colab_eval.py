#!/usr/bin/env python3
"""
Run in Colab AFTER training (or on your own computer after you unzip the results): all the evaluation and
comparison, with every output saved to files and one zip to download.

  %run scripts/colab_eval.py                 (inside a notebook: also shows the figures and downloads the zip)
  python scripts/colab_eval.py --runs runs   (anywhere)

Steps (each prints how long it took):
  1. calibration            python scripts/calibrate.py       -> results/calibration.csv   (skipped if already done)
  2. diagnosis metrics      python scripts/eval_metrics.py    -> results/metrics/   ROC/AUC, PR-AUC, Brier, ECE, NLL,
                                                                  per-class P/R/F1, confusion matrices, bootstrap CIs
  3. model comparison       python scripts/compare_models.py  -> results/metrics/   ranking, paired tests, ROC overlay
  4. field gap (E1)         python scripts/e1_gap.py          -> results/e1_gap.md
  5. paper tables           python scripts/report.py          -> results/paper_tables.md (needs stages 6-7 first)
Output: results/ (all tables, CSVs, figures) and tomato_eval.zip (figures included; this zip is only for downloading,
never commit it).
"""
import argparse
import os
import pathlib
import subprocess
import sys
import time


def run(title, cmd, optional=False):
    t = time.time()
    print(f"\n===== {title} =====\n+ {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    print(f"   {'done' if rc == 0 else 'FAILED (exit %d)' % rc}: {title} took {time.time() - t:.0f}s", flush=True)
    if rc and not optional:
        raise SystemExit(f"{title} failed")
    return rc == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--bootstrap", default="500")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    os.chdir(root)
    py = sys.executable
    if not pathlib.Path(args.runs).exists():
        sys.exit(f"{args.runs}/ not found. Train first (scripts/colab_run.py), or unzip tomato_results.zip here.")

    t0 = time.time()
    if not (pathlib.Path(args.results) / "calibration.csv").exists():
        run("1 calibration", [py, "scripts/calibrate.py", "--runs", args.runs, "--results", args.results], optional=True)
    run("2 diagnosis metrics", [py, "scripts/eval_metrics.py", "--runs", args.runs, "--results", args.results,
                                "--bootstrap", args.bootstrap])
    run("3 model comparison", [py, "scripts/compare_models.py", "--runs", args.runs, "--results", args.results])
    run("4 field gap (E1)", [py, "scripts/e1_gap.py", "--runs", args.runs, "--results", args.results], optional=True)
    if (pathlib.Path(args.results) / "e4_ablation_raw.csv").exists():
        run("5 paper tables", [py, "scripts/report.py", "--results", args.results], optional=True)

    res = pathlib.Path(args.results)
    print("\n===== Key results =====")
    for name in ("metrics/model_comparison.md", "e1_gap.md"):
        f = res / name
        if f.exists():
            print(f"\n--- {f} ---\n{f.read_text(encoding='utf-8')[:6000]}")

    try:  # inside a notebook: show the figures under the cell
        from IPython.display import Image, display
        for f in sorted((res / "metrics").glob("fig*.png")):
            print(f.name)
            display(Image(filename=str(f), width=620))
    except Exception:
        pass

    if not args.no_zip:
        zp = root / "tomato_eval.zip"
        zp.unlink(missing_ok=True)
        subprocess.run(["zip", "-qr", str(zp), args.results] if os.name != "nt" else
                       [py, "-c", f"import shutil; shutil.make_archive(r'{zp.with_suffix('')}', 'zip', r'{args.results}')"],
                       check=True)
        print(f"\nwrote {zp} ({zp.stat().st_size / 1e6:.1f} MB)")
        drive = pathlib.Path("/content/drive/MyDrive")
        if drive.exists():
            subprocess.run(["cp", str(zp), str(drive / "tomato_eval.zip")], check=True)
            print("copied to Google Drive: tomato_eval.zip")
        try:  # works when started with %run inside the notebook
            from google.colab import files
            files.download(str(zp))
            print("browser download started")
        except Exception as e:
            print(f"auto-download skipped ({type(e).__name__}); get {zp.name} from the Files pane")
    print(f"\nAll evaluation done in {time.time() - t0:.0f}s. Open {res}/metrics/metrics_summary.md and model_comparison.md")


if __name__ == "__main__":
    main()
