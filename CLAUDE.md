# CLAUDE.md: tomato-agent

Offline v1.0 of *Confidence-Gated Agentic Diagnosis of Tomato Diseases* (paper by Nibedita Adhikari).
Four agents (quality → diagnosis → gate → expert referral) on PlantVillage + PlantDoc, 8 tomato classes.
Read README.md for the 8 stages and what each script writes.

## Rules
- **Offline only.** No cloud, no OpenAI/Azure keys, no paid APIs. Ollama on localhost is the only allowed LLM (demo only).
- **Do not run `git commit` or `git push`.** The owner commits manually. `git status`/`git diff` are fine.
- Machine is **Windows + PowerShell**. Use `.\run_all.ps1`, not the .sh scripts. If scripts are blocked:
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
- Do not change the class order in `tomato_agent/classes.py` (logits, manifests and the future Android app depend on it).
- `tomato_agent/quality.py` and `gate.decide` will be mirrored in the Android app (PatraSathi): if you change their maths, say so clearly.
- Do not change doses in `shared/advice.json`; it is a draft waiting for agronomist review.
- After any code change: `python -m pytest -q` must pass.
- Never use test sets (pv_test, pd_test) to choose thresholds or temperatures.

## Run order
1. `pip install -r requirements.txt`
2. `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
   - False + NVIDIA GPU present → `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall`
   - no NVIDIA GPU → do the smoke test here; train on Kaggle/Colab later.
3. `python -m pytest -q` (27 tests)
4. `.\run_all.ps1 -Smoke` (synthetic data, ~10 min on CPU; numbers are meaningless, it only checks the pipeline)
5. `.\scripts\download_data.ps1` (tomato images only, ~1.1 GB)
6. Pilot: `.\run_all.ps1 -Backbones efficientnet_b0 -Seeds 0`
7. Full study: `.\run_all.ps1` (3 backbones × 5 seeds, hours on a GPU)

Compare stage 1–2 output with `docs/reference_run/` (same dataset commits → same split counts and thresholds).

## Troubleshooting
- DataLoader hangs or "BrokenPipe" on Windows → add `--workers 0` to the failing script.
- CUDA out of memory → `--batch-size 32` (or 16) for train_diagnosis.py / predict_conditions.py.
- timm cannot download ImageNet weights (corporate network) → try another network, or pre-download on Kaggle.
- Training is resumable: finished runs (metrics.json present) are skipped; delete a run folder to redo it.

Outputs: paper tables in `results/paper_tables.md`, figures `results/fig*.png`, phone files in `export/`.
