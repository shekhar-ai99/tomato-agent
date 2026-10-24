# Handoff: tomato-agent (Windows laptop -> Mac)

Read CLAUDE.md and README.md first. Rules there still apply (offline only, no paid APIs, owner commits manually).
CLAUDE.md says "Windows + PowerShell"; on the Mac use bash and `./run_all.sh` instead of `.\run_all.ps1`.

## Where the project stands
- Code for all 4 agents, 8 stages, phone export: done. `python -m pytest -q` -> 28 passed.
- Windows smoke test (`run_all.ps1 -Smoke`) passes end to end.
- Stage 1 on the real data matches `docs/reference_run/` exactly (split counts, 46 dedup removals).
- Stage 2 thresholds did NOT match the reference on Windows: degradation seeds used backslash paths.
  Fixed in `tomato_agent/degrade.py` (`load_condition` normalises to `/`). The re-run was never finished, so the
  thresholds are still unchecked. On the Mac (forward slashes) they should match the reference directly.
- LangGraph demo works: `app/web_demo.py` (standard library, no Gradio) + `GraphPipeline` in `tomato_agent/graph.py`.
  Tested only with the smoke model (synthetic data, meaningless diagnoses).
- Real training has NOT happened. The Windows laptop is ARM64 with no NVIDIA GPU.

## Next steps
1. Mac setup (below), then `./scripts/download_data.sh` and stages 1-2, and compare with `docs/reference_run/`.
2. Pilot on Colab (free T4 from the Colab website, signed in as the user): paste in one cell
   ```python
   import os
   os.environ['GITHUB_TOKEN'] = ''   # empty = zip instead of push; never save a token in the notebook
   os.environ['BACKBONES'] = 'efficientnet_b0'
   os.environ['SEEDS'] = '0'
   %cd /content
   !rm -rf tomato-agent
   !git clone https://github.com/shekhar-ai99/tomato-agent.git
   %cd tomato-agent
   !nvidia-smi -L
   !pip install -q timm imagehash onnx onnxruntime scipy pandas matplotlib pytest
   !python scripts/colab_run.py
   ```
   Then the full study (drop the BACKBONES/SEEDS lines: 3 backbones x 5 seeds, 5-8 h).
   `scripts/colab_run.py` pushes models/logits/metrics (no images) to the `colab-results` branch.
3. Copy `runs/` back, then `python app/web_demo.py --run runs/efficientnet_b0/seed0` and test with a real leaf photo.
4. Paper checklist in README: confirm `PD_FOLDERS`, record dataset commits, dedup counts, agronomist review of `shared/advice.json`.

## Known pitfalls
- The VS Code Colab extension only offered CPU servers; use colab.research.google.com for a GPU.
- Colab "Failed to fetch": ad blocker / VPN / proxy, or Drive mount. Drive is optional for the script.
- On Windows, 20 PlantDoc files with `?` in the name and 2 very long names are renamed by
  `scripts/checkout_plantdoc_windows.py`. Not needed on Mac/Linux (`download_data.sh` works).
- Do not use pv_test / pd_test to choose thresholds or temperatures.

## Mac setup
```bash
git clone https://github.com/shekhar-ai99/tomato-agent.git && cd tomato-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt langgraph
python -m pytest -q
./scripts/download_data.sh          # ~1.1 GB, tomato images only; never commit them
```
