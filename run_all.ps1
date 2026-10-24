# All 8 stages of the v1.0 study (paper Section 4.6), for Windows PowerShell.
#   .\run_all.ps1            real data (run .\scripts\download_data.ps1 first)
#   .\run_all.ps1 -Smoke     tiny synthetic data + tiny models on CPU, a few minutes, numbers meaningless
param(
  [switch]$Smoke,
  [string[]]$Backbones = @("efficientnet_b0", "resnet50", "vit_small_patch16_224"),
  [int[]]$Seeds = @(0, 1, 2, 3, 4),
  [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$PV = "PlantVillage-Dataset"; $PD = "PlantDoc-Dataset"; $W = "."
$TrainExtra = @("--amp")
if ($Smoke) {
  $W = "smoke"; New-Item -ItemType Directory -Force $W | Out-Null
  & $Python scripts/make_synthetic_data.py --out "$W/data" --per-class 40
  $PV = "$W/data/PlantVillage-Dataset"; $PD = "$W/data/PlantDoc-Dataset"
  $Backbones = @("resnet18", "mobilenetv3_small_100"); $Seeds = @(0, 1)
  $TrainExtra = @("--epochs", "4", "--img-size", "96", "--pretrained", "0", "--batch-size", "32", "--workers", "0")
}
$M = "$W/manifests"; $A = "$W/artifacts"; $R = "$W/runs"; $O = "$W/results"
$Data = @("--pv-root", $PV, "--pd-root", $PD, "--pv-manifest", "$M/pv_manifest_dedup.csv", "--pd-manifest", "$M/pd_manifest.csv")
# Python warnings go to stderr; Windows PowerShell 5.1 turns that into a terminating error under "Stop",
# so judge each step by its exit code only.
function Step($name, [scriptblock]$cmd) {
  Write-Host "== $name" -ForegroundColor Green
  $ErrorActionPreference = "Continue"; $global:LASTEXITCODE = 0
  & $cmd
  $ErrorActionPreference = "Stop"
  if ($LASTEXITCODE) { throw "$name failed" }
}

Step "1 prepare data"       { & $Python scripts/prepare_data.py manifests --pv-root $PV --pd-root $PD --out $M }
Step "1 dedup"              { & $Python scripts/prepare_data.py dedup --pv-root $PV --manifest "$M/pv_manifest.csv" --out $M }
Step "2 quality thresholds" { & $Python scripts/set_quality_thresholds.py @Data --out $A --examples 3 }
Step "3 train"              { & $Python scripts/train_diagnosis.py @Data --backbones @Backbones --seeds @Seeds --out $R @TrainExtra }
Step "3b degraded logits"   { & $Python scripts/predict_conditions.py @Data --runs $R --quality "$A/quality_thresholds.json" }
Step "4 calibrate"          { & $Python scripts/calibrate.py --runs $R --results $O }
Step "5 E1 gap"             { & $Python scripts/e1_gap.py --runs $R --results $O }
Step "6 E2/E3 policy"       { & $Python scripts/run_policy.py --runs $R --artifacts $A --results $O }
Step "7 E4/E5 ablations"    { & $Python scripts/ablate.py --runs $R --artifacts $A --results $O }
Step "8 report"             { & $Python scripts/report.py --results $O --backbone $Backbones[0] }
Step "export for phone"     { & $Python scripts/export_mobile.py --run "$R/$($Backbones[0])/seed$($Seeds[0])" @Data --quality "$A/quality_thresholds.json" --thresholds "$O/thresholds.json" --out "$W/export" }
Write-Host "Done. Paper tables: $O/paper_tables.md"
