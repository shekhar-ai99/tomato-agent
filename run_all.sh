#!/usr/bin/env bash
# All 8 stages of the v1.0 study (paper Section 4.6).
#   ./run_all.sh            real data (run scripts/download_data.sh first)
#   ./run_all.sh --smoke    tiny synthetic data + tiny models on CPU, a few minutes, numbers meaningless
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
BACKBONES=${BACKBONES:-"efficientnet_b0 resnet50 vit_small_patch16_224"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
TRAIN_EXTRA=${TRAIN_EXTRA:-"--amp"}
PV=PlantVillage-Dataset; PD=PlantDoc-Dataset; W=.
if [ "${1:-}" = "--smoke" ]; then
  W=smoke; mkdir -p $W
  $PY scripts/make_synthetic_data.py --out $W/data --per-class 40
  PV=$W/data/PlantVillage-Dataset; PD=$W/data/PlantDoc-Dataset
  BACKBONES="resnet18 mobilenetv3_small_100"; SEEDS="0 1"
  TRAIN_EXTRA="--epochs 4 --img-size 96 --pretrained 0 --batch-size 32 --workers 2"
fi
M=$W/manifests; A=$W/artifacts; R=$W/runs; O=$W/results
DATA="--pv-root $PV --pd-root $PD --pv-manifest $M/pv_manifest_dedup.csv --pd-manifest $M/pd_manifest.csv"

echo "== 1 prepare data";        $PY scripts/prepare_data.py manifests --pv-root $PV --pd-root $PD --out $M
                                  $PY scripts/prepare_data.py dedup --pv-root $PV --manifest $M/pv_manifest.csv --out $M
echo "== 2 quality thresholds";  $PY scripts/set_quality_thresholds.py $DATA --out $A --examples 3
echo "== 3 train";               $PY scripts/train_diagnosis.py $DATA --backbones $BACKBONES --seeds $SEEDS --out $R $TRAIN_EXTRA
echo "== 3b degraded logits";    $PY scripts/predict_conditions.py $DATA --runs $R --quality $A/quality_thresholds.json
echo "== 4 calibrate";           $PY scripts/calibrate.py --runs $R --results $O
echo "== 5 E1 gap";              $PY scripts/e1_gap.py --runs $R --results $O
echo "== 5b metrics (ROC, AUC, Brier, ...)"; $PY scripts/eval_metrics.py --runs $R --results $O
echo "== 5c compare models";   $PY scripts/compare_models.py --runs $R --results $O
echo "== 6 E2/E3 policy";        $PY scripts/run_policy.py --runs $R --artifacts $A --results $O
echo "== 7 E4/E5 ablations";     $PY scripts/ablate.py --runs $R --artifacts $A --results $O
echo "== 8 report";              $PY scripts/report.py --results $O --backbone $(echo $BACKBONES | cut -d' ' -f1)
FIRST=$(echo $BACKBONES | cut -d' ' -f1); SEED0=$(echo $SEEDS | cut -d' ' -f1)
echo "== export for phone";      $PY scripts/export_mobile.py --run $R/$FIRST/seed$SEED0 $DATA --quality $A/quality_thresholds.json --thresholds $O/thresholds.json --out $W/export
echo "Done. Paper tables: $O/paper_tables.md"
