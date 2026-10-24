#!/usr/bin/env python3
"""
One command for Colab: download the images, run all 8 stages on the GPU, save the results (no images).
Start it inside the notebook with `%run scripts/colab_run.py` so the browser download works and the logs show.

While it runs you see which step is running, how long that step has taken, the total time and, for training,
the epoch/run progress with an ETA. A table of time per step is printed at the end.

Settings come from environment variables:
  BACKBONES    default "efficientnet_b0 resnet50 vit_small_patch16_224"
  SEEDS        default "0 1 2 3 4"        (pilot: BACKBONES=efficientnet_b0 SEEDS=0)
  PUSH_BRANCH  default "colab-results"    (never main)
  GITHUB_TOKEN needed for the push; without it the push is skipped (the zip is still made and downloaded)
  MAX_FILE_MB  default 95                 (GitHub rejects files over 100 MB; bigger ones are skipped)
  HEARTBEAT_S  default 30                 (seconds between the "still running" lines)
"""
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

BACKBONES = os.environ.get("BACKBONES", "efficientnet_b0 resnet50 vit_small_patch16_224")
SEEDS = os.environ.get("SEEDS", "0 1 2 3 4")
PUSH_BRANCH = os.environ.get("PUSH_BRANCH", "colab-results")
MAX_MB = float(os.environ.get("MAX_FILE_MB", "95"))
HEARTBEAT_S = float(os.environ.get("HEARTBEAT_S", "30"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "30"))  # train_diagnosis.py default; early stopping usually ends sooner
IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff")
N_RUNS = len(BACKBONES.split()) * len(SEEDS.split())

T0 = time.time()
# Rough seconds per step on a free T4 (guesses from one pilot run; per-run steps are multiplied by the number of runs).
TYPICAL = {"GPU": 10, "Drive": 1, "Tests": 15, "Download": 200,
           "1 prepare data": 30, "1 dedup": 240, "2 quality thresholds": 420, "3 train": 1300 * N_RUNS,
           "3b degraded logits": 300 * N_RUNS, "4 calibrate": 10, "5 E1 gap": 10, "5b metrics (ROC, AUC, Brier, ...)": 60, "5c compare models": 30, "6 E2/E3 policy": 30,
           "7 E4/E5 ablations": 30, "8 report": 20, "export for phone": 60, "Save": 30, "Push": 60}
STAGES = ["1 prepare data", "1 dedup", "2 quality thresholds", "3 train", "3b degraded logits", "4 calibrate",
          "5 E1 gap", "5b metrics (ROC, AUC, Brier, ...)", "5c compare models", "6 E2/E3 policy", "7 E4/E5 ablations", "8 report", "export for phone"]


def fmt(sec):
    m, s = divmod(int(max(sec, 0)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Tracker:
    """Shows the running step and its elapsed time live (a line every HEARTBEAT_S seconds), plus a table at the end."""

    def __init__(self):
        self.name, self.t, self.pos = None, None, ""
        self.log, self.runs_done, self.train_t, self.detail = [], 0, None, ""
        self._stop = threading.Event()
        threading.Thread(target=self._beat, daemon=True).start()

    def _beat(self):
        while not self._stop.wait(HEARTBEAT_S):
            if self.name is not None and not self.name.startswith("All 8 stages"):
                est = TYPICAL.get(self.name)
                est_txt = f" (typical ~{fmt(est)})" if est else ""
                print(f"   ... {self.pos}{self.name}: running {fmt(time.time() - self.t)}{est_txt}"
                      f" | total {fmt(time.time() - T0)}{self.detail}", flush=True)

    def finish(self):
        if self.name is not None:
            dt = time.time() - self.t
            if not self.name.startswith("All 8 stages"):
                print(f"   done: {self.name} took {fmt(dt)}", flush=True)
                self.log.append((self.name, dt))
            self.name = None

    def start(self, name, label=None, pos=""):
        self.finish()
        self.name, self.t, self.pos, self.detail = name, time.time(), pos, ""
        est = TYPICAL.get(name)
        extra = f" | typical ~{fmt(est)}" if est else ""
        print(f"\n===== {pos}{label or name} | total elapsed {fmt(time.time() - T0)}{extra} =====", flush=True)

    def on_line(self, line):
        """Follow the stages run_all.sh prints ('== 3 train') and the training progress lines."""
        m = re.match(r"== (.+)", line)
        if m and m.group(1).strip() in STAGES:
            name = m.group(1).strip()
            self.start(name, pos=f"stage {STAGES.index(name) + 1}/{len(STAGES)}: ")
            if name == "3 train":
                self.train_t = time.time()
            return
        m = re.search(r"seed(\d+) epoch=(\d+) .*seconds=([\d.]+)", line)
        if m:
            ep, sec = int(m.group(2)), float(m.group(3))
            msg = f" | run {self.runs_done + 1}/{N_RUNS}, epoch {ep}/{MAX_EPOCHS} max, " \
                  f"{sec:.0f}s/epoch, this run <= {fmt(max(MAX_EPOCHS - ep, 0) * sec)} left"
            if self.runs_done:
                per_run = (time.time() - self.train_t) / self.runs_done
                msg += f", all training ETA ~{fmt(per_run * (N_RUNS - self.runs_done))}"
            self.detail = msg
            print(f"   ->{msg[2:]}", flush=True)
        elif re.search(r"seed\d+: PV test acc", line):
            self.runs_done += 1
            print(f"   -> runs finished: {self.runs_done}/{N_RUNS}", flush=True)

    def summary(self):
        self.finish()
        self._stop.set()
        print("\n===== Summary: time per step =====")
        for name, dt in self.log:
            print(f"  {name:28s} {fmt(dt)}")
        print(f"  {'TOTAL':28s} {fmt(time.time() - T0)}", flush=True)


TRACK = Tracker()


def sh(*cmd, track=False, **kw):
    """Run a command and print its output from this process, so it also shows under a notebook cell
    when the script is started with %run (a child's own stdout would go to the kernel console)."""
    print("+", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", **kw)
    for line in p.stdout:
        print(line, end="", flush=True)
        if track:
            TRACK.on_line(line)
    if p.wait():
        raise subprocess.CalledProcessError(p.returncode, cmd)


def out(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def step(name, label=None, pos=""):
    TRACK.start(name, label, pos)


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    os.chdir(root)

    step("GPU")
    import torch
    if not torch.cuda.is_available():
        sys.exit("No GPU. In Colab: Runtime > Change runtime type > T4 GPU, then run again.")
    print(torch.__version__, torch.cuda.get_device_name(0))

    step("Drive", "Keep runs/ on Google Drive when it is mounted (resume after a disconnect)")
    drive = pathlib.Path("/content/drive/MyDrive")
    if drive.exists():
        (drive / "tomato_runs").mkdir(exist_ok=True)
        if not os.path.islink("runs") and os.path.exists("runs"):
            sh("rm", "-rf", "runs")
        if not os.path.islink("runs"):
            os.symlink(drive / "tomato_runs", "runs")
        print("runs/ -> Google Drive")
    else:
        print("Drive is not mounted; runs stay on this VM (a disconnect loses them)")

    step("Tests")
    sh(sys.executable, "-m", "pytest", "-q")

    step("Download", "Download the tomato images (public GitHub repos, never uploaded anywhere)")
    sh("bash", "scripts/download_data.sh")

    step(f"All 8 stages: {BACKBONES} x seeds {SEEDS}")
    env = {**os.environ, "BACKBONES": BACKBONES, "SEEDS": SEEDS, "TRAIN_EXTRA": "--amp --workers 2",
           "PYTHON": sys.executable}
    sh("bash", "run_all.sh", env=env, track=True)

    step("Save", "Save results (no images): zip, Drive copy, browser download")
    token = os.environ.get("GITHUB_TOKEN")
    keep = [d for d in ("runs", "manifests", "artifacts", "results", "export", "dataset_versions.txt")
            if pathlib.Path(d).exists()]
    sh("rm", "-f", "tomato_results.zip")
    sh("zip", "-qr", "tomato_results.zip", *keep, "-x", *[f"*{e}" for e in IMG_EXT])
    print("wrote", root / "tomato_results.zip")
    if drive.exists():
        sh("cp", "tomato_results.zip", str(drive / "tomato_results.zip"))
        print("copied to Google Drive: tomato_results.zip")
    try:  # works only when this script runs inside the notebook: use `%run scripts/colab_run.py`, not `!python`
        from google.colab import files
        files.download(str(root / "tomato_results.zip"))
        print("browser download started")
    except Exception as e:
        print(f"auto-download skipped ({type(e).__name__}); get the zip from the Files pane")
    if not token:
        print("No GITHUB_TOKEN: not pushing.")
        TRACK.summary()
        return
    step("Push", "Push results to Git (no images)")
    origin = out("git", "remote", "get-url", "origin").strip()
    sh("git", "config", "user.name", "colab-trainer")
    sh("git", "config", "user.email", "colab-trainer@users.noreply.github.com")
    sh("git", "checkout", "-B", PUSH_BRANCH)
    excl = [f":(exclude)*{e}" for e in IMG_EXT] + [f":(exclude)*{e.upper()}" for e in IMG_EXT]
    for d in keep:  # .gitignore hides these folders, so add with -f; images are excluded by the pathspecs
        sh("git", "add", "-f", "--", d, *excl)
    for f in filter(None, out("git", "diff", "--cached", "--name-only").splitlines()):
        mb = pathlib.Path(f).stat().st_size / 1e6
        if mb > MAX_MB:
            sh("git", "rm", "--cached", "-q", "--", f)
            print(f"skipped {f} ({mb:.0f} MB > {MAX_MB:.0f} MB; use Git LFS for files this big)")
    staged = out("git", "diff", "--cached", "--name-only").splitlines()
    assert not any(x.lower().endswith(IMG_EXT) for x in staged), "an image got staged"
    print(len(staged), "files staged")
    sh("git", "commit", "-q", "-m", f"Colab results: {BACKBONES} seeds {SEEDS}")
    url = origin.replace("https://", f"https://x-access-token:{token}@")
    r = subprocess.run(["git", "push", "--force", url, f"{PUSH_BRANCH}:{PUSH_BRANCH}"], capture_output=True, text=True)
    print((r.stdout + r.stderr).replace(token, "***"))
    r.check_returncode()
    print("Pushed to", origin.removesuffix(".git") + "/tree/" + PUSH_BRANCH)
    TRACK.summary()


if __name__ == "__main__":
    main()
