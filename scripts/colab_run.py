#!/usr/bin/env python3
"""
One command for Colab: download the images, run all 8 stages on the GPU, push the results (no images) to Git.
Run it from the repo root after the usual Colab setup (see the cell in README "Train on Colab").

Settings come from environment variables:
  BACKBONES    default "efficientnet_b0 resnet50 vit_small_patch16_224"
  SEEDS        default "0 1 2 3 4"        (pilot: BACKBONES=efficientnet_b0 SEEDS=0)
  PUSH_BRANCH  default "colab-results"    (never main)
  GITHUB_TOKEN needed for the push; without it the push is skipped and a zip is made instead
  MAX_FILE_MB  default 95                 (GitHub rejects files over 100 MB; bigger ones are skipped)
"""
import os
import pathlib
import subprocess
import sys

BACKBONES = os.environ.get("BACKBONES", "efficientnet_b0 resnet50 vit_small_patch16_224")
SEEDS = os.environ.get("SEEDS", "0 1 2 3 4")
PUSH_BRANCH = os.environ.get("PUSH_BRANCH", "colab-results")
MAX_MB = float(os.environ.get("MAX_FILE_MB", "95"))
IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff")


def sh(*cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def out(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def step(msg):
    print(f"\n===== {msg} =====", flush=True)


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    os.chdir(root)

    step("GPU")
    import torch
    if not torch.cuda.is_available():
        sys.exit("No GPU. In Colab: Runtime > Change runtime type > T4 GPU, then run again.")
    print(torch.__version__, torch.cuda.get_device_name(0))

    step("Keep runs/ on Google Drive when it is mounted (resume after a disconnect)")
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

    step("Download the tomato images (public GitHub repos, never uploaded anywhere)")
    sh("bash", "scripts/download_data.sh")

    step(f"All 8 stages: {BACKBONES} x seeds {SEEDS}")
    env = {**os.environ, "BACKBONES": BACKBONES, "SEEDS": SEEDS, "TRAIN_EXTRA": "--amp --workers 2",
           "PYTHON": sys.executable}
    sh("bash", "run_all.sh", env=env)

    step("Push results (no images)")
    token = os.environ.get("GITHUB_TOKEN")
    keep = [d for d in ("runs", "manifests", "artifacts", "results", "export", "dataset_versions.txt")
            if pathlib.Path(d).exists()]
    if not token:
        sh("zip", "-qr", "tomato_results.zip", *keep, "-x", *[f"*{e}" for e in IMG_EXT])
        print("No GITHUB_TOKEN: wrote tomato_results.zip instead (download it from the Files pane).")
        return
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


if __name__ == "__main__":
    main()
