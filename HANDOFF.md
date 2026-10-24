# Handoff: tomato-agent

Read CLAUDE.md and README.md first. Rules there still apply (offline only, no paid APIs, owner commits manually).
CLAUDE.md says "Windows + PowerShell"; on a Mac/Linux use bash and `./run_all.sh`.

## State (24 Oct 2026)
* Code for all 4 agents, the gate, 8 stages + metrics stages 5b/5c, the LangGraph agent loop, the web demo, and the phone
  export is done. `python -m pytest -q` -> 34 passed.
* Pilot run on a Colab T4 (EfficientNet-B0, seed 0): PlantVillage test 99.1%, PlantDoc test 23.4%, gap 75.8 points
  (reframe gate passed). Stage 1-2 on Colab match `docs/reference_run/` exactly. Training is deterministic: reruns give
  identical numbers.
* PlantDoc label mapping (`PD_FOLDERS`) was checked by viewing sample photos of every folder: correct.
* Field errors collapse into late blight / early blight (domain shift, not a labelling bug).
* Two calibrations are supported everywhere: `pv` (temperature fitted on PlantVillage) and `pd` (fitted on the 149 PlantDoc
  calibration photos). Gate thresholds: pv tuned on PlantVillage val, pd tuned on `pd_calib` (never a test set).
  With the pilot model the field gate is 0.45 / 0.45 and defers about 99% of cases (21% field accuracy), by design of the
  cost rule.
* Phone export: INT8 is rejected automatically (best top-1 agreement with fp32 was 81%, minimum 95%); the config uses
  `model_fp32.onnx` (16 MB, about 7.5 ms per image on a laptop CPU). `patrasathi_config.json` has both calibrations.
* `docs/paper_edits.md` has verified text and numbers for the paper (FINAL vs PILOT marked).

## What is left
1. `git push` (Colab clones GitHub; it needs the metrics scripts, `colab_eval.py` and the calibration switch).
2. Full study on Colab (3 backbones x 5 seeds, about 5-8 h): paste the cell below, then run `%run scripts/colab_eval.py`.
3. Paste `docs/paper_edits.md` into the draft; replace PILOT numbers with mean +- SD; add related work and references.
4. Agronomist review of `shared/advice.json` (doses are a draft; do not change them).
5. Android app (PatraSathi): mirror `quality.py` and `gate.decide`; use `export/patrasathi_config.json`.
6. Optional research choice: a stronger field model (augmentation, domain adaptation, or using some PlantDoc train photos)
   would change the paper's design (PlantDoc is currently evaluation + calibration only). Decide after the full study.

## Colab cell (full study)
```python
import os
os.environ['GITHUB_TOKEN'] = ''   # empty = zip is downloaded; or paste a token for this session only (never commit it)
%cd /content
!rm -rf tomato-agent
!git clone https://github.com/shekhar-ai99/tomato-agent.git
%cd tomato-agent
!nvidia-smi -L
!pip install -q timm imagehash onnx onnxruntime scipy pandas matplotlib pytest
%run scripts/colab_run.py
```
Pilot: add `os.environ['BACKBONES']='efficientnet_b0'` and `os.environ['SEEDS']='0'` before `%cd`.
After training: `%run scripts/colab_eval.py` (calibration, ROC/AUC/Brier, comparison, figures, `tomato_eval.zip`).

## Run the demo locally
```bash
pip install langgraph
python app/web_demo.py --run tomato_results/runs/efficientnet_b0/seed0 --quality tomato_results/artifacts/quality_thresholds.json --thresholds tomato_results/results/thresholds.json
```
Open http://127.0.0.1:7860. The page has a Lab (pv) / Field (pd) switch. `tomato_results/` is the unzipped Colab output
(gitignored).

## Pitfalls
* The VS Code Colab extension only offered CPU servers; use colab.research.google.com for a GPU.
* Colab "Failed to fetch": ad blocker / VPN / proxy, or the Drive mount. Drive is optional for the script.
* On Windows, 20 PlantDoc files with `?` and 2 very long names are renamed by `scripts/checkout_plantdoc_windows.py`
  (not needed on Linux/Colab). Degradation seeds use forward-slash paths so results match across systems.
* Do not use pv_test / pd_test to choose thresholds or temperatures.
