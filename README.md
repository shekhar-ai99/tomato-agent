# tomato-agent

**Confidence-gated agentic diagnosis of tomato leaf diseases: v1.0, fully offline.**

This repo holds the code behind the paper *Confidence-Gated Agentic Diagnosis of Tomato Diseases* (Nibedita Adhikari, Utkal University). Four agents turn a single-shot classifier into a decision process:

| Agent | What it does | Code |
|---|---|---|
| Capture-quality | Flags blurry, dark, over-exposed or leaf-less photos and says why | `tomato_agent/quality.py` |
| Diagnosis | CNN/ViT → temperature-scaled class probabilities | `tomato_agent/models.py`, `calibration.py` |
| Verification (gate) | **accept**, **retake** or **defer**, with thresholds chosen by a cost rule | `tomato_agent/gate.py` |
| Expert-referral | Packages deferred cases (photos, top-3, entropy, reason) as JSON | `tomato_agent/expert.py` |

The live loop is in `tomato_agent/pipeline.py` (plain Python), with an optional LangGraph version in `tomato_agent/graph.py`.

**No cloud, no OpenAI, no API keys.** Everything runs on a laptop or a free Kaggle/Colab GPU. The only network access is the one-time dataset download and the ImageNet weights that `timm` fetches on first use. A local LLM through Ollama can optionally explain a diagnosis in the demo; it never changes the decision and is not used in any metric.

The Android app **PatraSathi** (repo `tomato-agent-andriod`) will use the files from `scripts/export_mobile.py`, so the phone makes the same decisions the paper evaluates.

---

## Quick start (Windows)

```powershell
# 1. environment (Python 3.10+). Install PyTorch with CUDA first: https://pytorch.org/get-started/locally/
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. check that everything works (synthetic data, tiny CPU models, ~5-10 minutes)
python -m pytest -q
.\run_all.ps1 -Smoke

# 3. real data: tomato parts only, ~1.1 GB
.\scripts\download_data.ps1

# 4. the full study (all 8 stages)
.\run_all.ps1
```

On Linux or macOS use `./scripts/download_data.sh` and `./run_all.sh` (add `--smoke` for the smoke test).
Paper tables then appear in `results/paper_tables.md` and the figures in `results/fig*.png`.

To run fewer models while you test on real data:
`.\run_all.ps1 -Backbones efficientnet_b0 -Seeds 0`.

---

## The 8 stages (paper Section 4.6)

| # | Script | Writes | Feeds |
|---|---|---|---|
| 1 | `prepare_data.py manifests` + `dedup` | `manifests/*.csv` (leaf-grouped splits, PlantDoc 20% calib / 80% test, near-duplicates removed) | everything |
| 2 | `set_quality_thresholds.py` | `artifacts/quality_features/*.npz`, `quality_thresholds.json`, `quality_detection.csv`, `degraded_examples/` | 6, 7, phone |
| 3 | `train_diagnosis.py` | `runs/<backbone>/seed<k>/` best.pt, logits, metrics | 3b-5 |
| 3b | `predict_conditions.py` | `cond_<set>.npz`: logits on clean and degraded copies | 6, 7 |
| 4 | `calibrate.py` | `calibration.json` per run (T_pv and T_pd), `results/calibration.csv`, `reliability_bins.csv` | 6, 7, Fig. 4 |
| 5 | `e1_gap.py` (E1, RQ1) | `results/e1_gap.md`, per-class recall, confusion matrices + the **reframe gate** | Table 5.1, Fig. 2 |
| 6 | `run_policy.py` (E2, E3; RQ2, RQ3) | `results/thresholds.json`, `e2_*_raw.csv`, `e3_*_raw.csv`, `curves_raw.csv` | Tables 5.2, 5.3, Fig. 3 |
| 7 | `ablate.py` (E4, E5; RQ4) | `results/e4_ablation_raw.csv`, `e5_cost_workload_raw.csv` | Table 5.4 |
| 8 | `report.py` | `results/paper_tables.md`, `stats_tests.csv`, `fig1b_gap.png`, `fig2-4*.png` | Section 5 |
| + | `export_mobile.py` | `export/model_int8.onnx`, `patrasathi_config.json`, `export_report.json` (size, latency) | Section 5.5, phone |

Every stage reads the previous stage's files, so any table can be regenerated from the stored splits and seeds. Interrupted training resumes (finished runs are skipped).

### How the experiments are set up

* **Degradations (Section 4.3).** Gaussian blur, motion blur, low light, over-exposure and occlusion, at level 2 ("moderate") by default (`--levels 1 2 3` for more). Each test photo gets two independently degraded copies. Copy 0 is the first shot. A retake is either the clean original (**optimistic** bound) or copy 1 (**pessimistic** bound), and both are reported.
* **Mixed scenario.** Each test case is randomly assigned one condition (clean or one of the 5 degradations). The draw is the same for every model, so comparisons are paired.
* **Quality thresholds** are chosen on validation photos only: PlantVillage val plus the PlantDoc calibration hold-out. Only the images are used, never their labels. Each signal may wrongly flag at most 3% of clean photos in *each* set. Using PlantVillage alone does not work: its plain backgrounds have almost no very bright or dark pixels, so PlantVillage-only thresholds flagged most clean field photos in testing. Each threshold also has a sensible floor or ceiling (`quality.BOUNDS`).
* **Gate thresholds** (tau_accept, tau_defer) are chosen by grid search to minimise `C_err·P(wrong & accepted) + C_def·P(deferred) + C_ret·E[retakes]` on PlantVillage val (mixed degradations, pessimistic retakes). The default ratios are C_err:C_def = 10:1 and 5:1, with C_ret = 0.1, and the paper's retake budget is R = 1.
* **Baselines:** B1 static model (uncalibrated), B2 ensemble of the backbones (same seed), and B3 calibrated confidence threshold only (no quality check, no retake). **Ablations:** without quality agent, without calibration, without retake.
* **Temperature:** `pv` = fitted on PlantVillage val; `pd` = field-calibrated on the 20% PlantDoc hold-out. Both are reported.
* **Expert = oracle** (ground truth), as in Section 3.5.

### Rough compute (RTX 3060 / 4060, 8–12 GB)

Training 3 backbones × 5 seeds takes about 5–8 hours in total with early stopping. Stage 3b (degraded copies for all models in one pass) takes about 30–60 minutes and is CPU-bound, so raise `--workers` if you have cores. Stages 4–8 take minutes. On Kaggle or Colab, run stage 3 there and copy `runs/` back.

---

## Demo (optional)

```bash
pip install -r requirements-optional.txt
python app/demo.py --run runs/efficientnet_b0/seed0
```

This opens a local web page (127.0.0.1 only). You upload or take a leaf photo, and the demo returns a diagnosis with advice, a retake tip, or "sent to expert", in English, Hindi or Odia. Deferred cases are saved to `cases/`.
To get explanations, install [Ollama](https://ollama.com), run `ollama pull llama3.1:8b` (or `qwen2.5vl:7b`) and tick *Explain with local LLM*.

## Farmer-facing content

`shared/retake_tips.json` and `shared/advice.json` hold the retake tips and the advice for all 8 classes in English, Hindi and Odia. **The advice is a draft:** an agronomist (OUAT or the local KVK) must check doses and translations before any field use. Doses are common label rates, and the app always tells farmers to confirm before spraying.

## Repository layout

```
tomato_agent/      the agents, gate, calibration, metrics, experiment helpers
scripts/           the 8 stages, export, synthetic data, dataset download
app/demo.py        local Gradio demo
shared/            advice + retake tips (en/hi/or), also shipped to the phone
tests/             unit tests (pytest); the gate test checks the batch simulation equals step-by-step decide()
run_all.ps1/.sh    one command for the whole study (-Smoke / --smoke for a quick check)
```

## Before submitting the paper

- [ ] Confirm the PlantDoc label mapping in `scripts/prepare_data.py` (`PD_FOLDERS`) by looking at sample images.
- [ ] Record the dataset commits (`dataset_versions.txt`, written by the download script) and PlantVillage's license.
- [ ] Report the dedup count printed by stage 1 and the per-class split counts (`manifests/split_report.csv`).
- [ ] Section 3.2: describe the leaf-visibility signal (share of yellow-to-green pixels) and that thresholds are chosen on validation photos (PlantVillage val + PlantDoc calibration hold-out, no labels).
- [ ] Section 4.3: state the degradation level(s) and that copies are deterministic (seeded by image path).
- [ ] Have the advice in `shared/advice.json` reviewed by an agronomist.

## Commit and push (manual)

```powershell
git init
git remote add origin https://github.com/shekhar-ai99/tomato-agent.git
git add .
git commit -m "tomato-agent v1.0: offline confidence-gated pipeline"
git branch -M main
git push -u origin main
```
`.gitignore` keeps datasets, runs, results and model files out of git.
