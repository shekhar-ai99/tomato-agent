"""Extract the PlantDoc tomato photos from a --no-checkout clone on Windows.

`git checkout` fails on Windows because ~20 PlantDoc file names contain '?' (e.g. 'latest?cb=2010.jpg').
Each such '?' is written as '@' instead: '@' is the next ASCII character after '?', so the sorted order
that prepare_data.py uses for the calibration/test split stays the same as on Linux.

Usage: python scripts/checkout_plantdoc_windows.py PlantDoc-Dataset
"""
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
names = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "-z", "HEAD"],
                       capture_output=True, check=True).stdout.decode("utf-8").split("\0")
wanted = [n for n in names if n.startswith(("train/Tomato", "test/Tomato"))]
bad = '<>:"|*'
for n in wanted:
    out = repo / n.replace("?", "@")
    if any(c in out.name for c in bad):
        raise SystemExit(f"cannot write {n}: still has an invalid character")
    # \\?\ prefix lifts Windows' 260-character path limit (a few PlantDoc names are very long)
    out = Path("\\\\?\\" + str(out.resolve()))
    if out.exists():
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", f"HEAD:{n}"],
                          capture_output=True, check=True).stdout
    out.write_bytes(blob)
print(f"wrote {len(wanted)} files ({sum('?' in n for n in wanted)} renamed '?' -> '@')")
