# Paper edits: text to paste into the draft

Every number here comes from files in this repo (`manifests/`, `artifacts/`, `results/`). Two kinds:

* **FINAL** = does not depend on training or on the number of seeds (data splits, dedup counts, quality thresholds and detection rates, dataset commits). Safe to paste now.
* **PILOT** = one backbone (EfficientNet-B0), one seed. Use only as placeholders. Replace with mean ± SD after the full study (3 backbones x 5 seeds) and `%run scripts/colab_eval.py`.

---

## 3.2 Capture-quality agent (replace the [TBD] about leaf visibility; FINAL)

Sharpness is the variance of the Laplacian, and exposure is described by the mean luminance and by the fractions of very dark and very bright pixels. Leaf visibility is the share of yellow-to-green pixels in the image. All signals are computed on the image resized to 256 pixels. An image is flagged when any signal crosses its threshold. The thresholds are chosen on validation photos only: the PlantVillage validation split and the 20% PlantDoc calibration hold-out. Only the images are used, never their labels. Each signal may wrongly flag at most 3% of clean photos in each of the two sets. Choosing the thresholds on PlantVillage alone is not enough: its plain backgrounds contain almost no very dark or very bright pixels, so thresholds fitted there flagged most clean field photos in our tests.

Chosen thresholds: minimum sharpness 80.7, minimum mean luminance 0.216, maximum dark fraction 0.643, maximum mean luminance 0.829, maximum bright fraction 0.637, minimum leaf fraction 0.137.

## 3.3 Diagnosis agent: calibration (add; PILOT numbers marked)

Temperature T is fitted by minimising the negative log-likelihood on a validation set. We report two variants: `pv` (fitted on the PlantVillage validation split) and `pd` (fitted on the 20% PlantDoc calibration hold-out, 149 photos).

## 3.4 Verification agent: how the thresholds are chosen (replace the sentence "Thresholds are selected on the validation set"; FINAL method)

The thresholds are chosen by grid search to minimise the expected cost on photos with mixed degradations and independently degraded retakes (the pessimistic bound). The lab-calibrated variant (`pv`) is tuned on the PlantVillage validation split. The field-calibrated variant (`pd`) is tuned on the PlantDoc calibration hold-out. No test photo is used to choose a temperature or a threshold. We report cost ratios C_err : C_def of 10:1 and 5:1, a retake cost C_ret of 0.1 C_def, and a retake budget R = 1.

## 4.1 Datasets (corrections; FINAL)

* Replace "we evaluate on the pooled 744 tomato images (675 train and 69 test)" with: "We pool the 744 PlantDoc tomato images in the eight shared classes (675 from its train folder and 69 from its test folder). A random 20% of each class (149 images) is held out as a calibration set, and the remaining 595 images are the field test set. None of them is used to train the diagnosis model."
* Dataset versions: PlantDoc commit `5467f60`, PlantVillage commit `7f7ecc7` (GitHub, taken 24 October 2026). Confirm PlantVillage's licence before submission.
* Label mapping: the folder-to-class mapping in Section 4.1 was checked by viewing sample photos of every folder, and it is correct. Replace the [TBD].
* Leakage check numbers: after the leaf-ID split, PlantVillage has 12,147 training, 1,466 validation and 1,467 test images. Perceptual-hash deduplication removed **46** training images (43 mapped to a leaf, 3 without a leaf ID). Mosaic virus has no leaf map, so its 373 images are training-only, and the in-domain validation and test sets contain seven classes.

| Class | PV train | PV val | PV test | PD calib | PD test |
|---|---|---|---|---|---|
| bacterial_spot | 1488 | 320 | 319 | 22 | 88 |
| early_blight | 697 | 151 | 152 | 18 | 70 |
| late_blight | 1637 | 136 | 136 | 22 | 89 |
| leaf_mold | 664 | 144 | 144 | 18 | 73 |
| septoria_leaf_spot | 1467 | 152 | 152 | 30 | 121 |
| yellow_leaf_curl_virus | 4534 | 411 | 412 | 15 | 61 |
| mosaic_virus | 373 | 0 | 0 | 11 | 43 |
| healthy | 1287 | 152 | 152 | 13 | 50 |
| **Total** | **12,147** | **1,466** | **1,467** | **149** | **595** |

## 4.2 Models and training (PILOT facts to confirm after the full study)

AdamW (weight decay 0.01) with one warm-up epoch followed by cosine decay, a per-backbone learning rate (see `DEFAULT_LR` in `scripts/train_diagnosis.py`), square-root class weights in the loss, 224 x 224 input, ImageNet initialisation, and augmentation by random resized crop (scale 0.5 to 1), horizontal flip and colour jitter (0.2 on brightness, contrast and saturation). Early stopping on validation loss with patience 5 and at most 30 epochs. The pilot EfficientNet-B0 stopped at epoch 20 (best epoch 15) and took about 65 s per epoch on a free Colab T4 GPU. Five seeds per backbone in the full study.

## 4.3 Degraded capture conditions (replace the [TBD]s; FINAL)

Gaussian blur, motion blur, low light, over-exposure and an occlusion patch, at level 2 ("moderate"). Each test photo gets two independently degraded copies, generated on the fly and seeded by the image path, so the same copies are produced on any machine. The first copy is the first shot. A retake is either the clean original (optimistic bound) or the second copy (pessimistic bound). Both are reported. A mixed scenario assigns each test photo one of six conditions (clean or one of the five degradations) at random, with the same draw for every model.

## 4.5 Metrics (add; computed by `scripts/eval_metrics.py` and `compare_models.py`)

In addition to accuracy, macro-F1, ECE, the risk-coverage curve and AURC: balanced accuracy; per-class precision, recall and F1; one-vs-rest ROC-AUC (macro, weighted and micro) and macro average precision (PR-AUC); the multi-class Brier score and negative log-likelihood; row-normalised confusion matrices; and 95% bootstrap confidence intervals over test photos (1,000 resamples) for accuracy, macro-F1, macro ROC-AUC, Brier score and ECE. Classes with no test photo in a set (mosaic virus in PlantVillage) are left out of the macro averages. Backbones are compared with paired t-tests and Wilcoxon tests across seeds; with five seeds the smallest possible two-sided Wilcoxon p-value is 0.0625, so effect sizes are reported alongside.

## 4.6 Experimental workflow (corrections)

* Stage 2: the degraded photos are not stored; they are regenerated deterministically when needed (see 4.3).
* Add stages 5b and 5c: research metrics and model comparison (Section 4.5).
* Stage 6: the thresholds for each temperature variant are chosen on its own hold-out set (3.4).

## 5.1 Laboratory-to-field gap (RQ1) (PILOT, EfficientNet-B0, seed 0)

PlantVillage test accuracy 99.1% (95% CI 98.6 to 99.6), macro ROC-AUC 1.000. PlantDoc test accuracy 23.4% (95% CI 19.8 to 26.6), macro-F1 15.4%, macro ROC-AUC 0.713 with the field temperature. Gap 75.8 points. The gate "field drop at least 0.10" passed, so the paper keeps the domain-shift framing.

PlantDoc recall per class: late blight 0.79, early blight 0.51, septoria 0.17, yellow leaf curl 0.10, healthy 0.06, leaf mold 0.04, bacterial spot 0.01, mosaic virus 0.00. The errors collapse into two classes: most field photos are predicted as late blight or early blight (see the confusion matrix), consistent with the model reading dark patches on cluttered backgrounds as blight rather than a wrong label mapping.

## 5.2 Quality agent (RQ2) (FINAL for detection; accuracy needs the full study)

Share of photos flagged by the quality agent (PlantVillage test / PlantDoc test): clean 2.3% / 7.1%; Gaussian blur 100% / 100%; motion blur 48.6% / 76.5%; low light 100% / 98.5%; over-exposure 87.0% / 73.3%; occlusion 86.1% / 18.2%. Occlusion is poorly detected on field photos because cluttered backgrounds contain many green and yellow pixels, which limits a colour-based leaf-visibility signal. Report this as a limitation.

## 5.3 / 5.4 Gated deferral (RQ3, RQ4) (PILOT)

At a cost ratio of 10:1 the lab-calibrated gate accepts at confidence 0.999 or more and defers below 0.933. The field-calibrated gate, tuned on the PlantDoc calibration photos, comes out at 0.45 / 0.45 and accepts only 0.7% of those photos, because at about 21% field accuracy deferring is cheaper than risking a wrong automatic diagnosis. State this plainly: the policy protects against a weak model but cannot make it useful, and the field results depend on how well the backbone transfers.

## 5.5 Edge feasibility (PILOT; replace the [Describe])

The EfficientNet-B0 ONNX model is 16.0 MB. On a laptop CPU, batch size 1, it took about 7.5 ms per image (PyTorch: about 27 ms). Static INT8 quantisation (QDQ, per-channel) with three calibration methods agreed with the fp32 model on only 54% (min-max), 81% (percentile) and 54% (entropy) of check photos, so INT8 is not used and the phone uses the fp32 model. Report the latency on the target phone separately.

## 6 Limitations (replace the [TBD] about calibration under shift; PILOT)

Re-fitting the temperature on 149 PlantDoc photos reduced the field ECE from 0.651 (PlantVillage temperature) to 0.057, and the Brier score from 1.360 to 0.831. This calibration set is small, so the field temperature and the field gate thresholds are themselves noisy estimates.

## Things still open

* Replace every PILOT number with the full-study mean ± SD.
* Add the related work and references marked [TBD]; the agent loop here is rule-based and orchestrated with LangGraph, with no LLM, so a reference to ReAct is not needed.
* Have an agronomist review `shared/advice.json` before any field claim.
