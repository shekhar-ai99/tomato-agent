# Download only the tomato parts of PlantVillage and PlantDoc (about 1.1 GB instead of many GB).
# Note: "--cone" mode does not accept wildcards such as train/Tomato*, so "--no-cone" is used.
# git prints progress on stderr, which Windows PowerShell 5.1 turns into an error under "Stop",
# so each git call is checked by its exit code instead.
$ErrorActionPreference = "Continue"
function Git { & git @args; if ($LASTEXITCODE) { throw "git $($args -join ' ') failed" } }
if (-not (Test-Path PlantDoc-Dataset)) {
  Git clone --filter=blob:none --no-checkout https://github.com/pratikkayal/PlantDoc-Dataset.git
  # ~20 PlantDoc file names contain '?', which Windows cannot create, so git checkout would fail.
  # The helper extracts the tomato photos itself and writes '?' as '@'.
  & python scripts/checkout_plantdoc_windows.py PlantDoc-Dataset
  if ($LASTEXITCODE) { throw "PlantDoc extraction failed" }
}
if (-not (Test-Path PlantVillage-Dataset)) {
  Git clone --filter=blob:none --no-checkout https://github.com/spMohanty/PlantVillage-Dataset.git
  Push-Location PlantVillage-Dataset
  Git sparse-checkout set --no-cone "/raw/color/Tomato*" "/leaf_grouping/"
  Git checkout
  Pop-Location
}
"PlantDoc commit:     $(git -C PlantDoc-Dataset rev-parse HEAD)"     | Tee-Object dataset_versions.txt
"PlantVillage commit: $(git -C PlantVillage-Dataset rev-parse HEAD)" | Tee-Object -Append dataset_versions.txt
