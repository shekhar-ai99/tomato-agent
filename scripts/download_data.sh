#!/usr/bin/env bash
# Download only the tomato parts of PlantVillage and PlantDoc (about 1.1 GB instead of many GB).
# Run from the folder where the datasets should live (default: the repo root).
set -euo pipefail
if [ ! -d PlantDoc-Dataset ]; then
  git clone --filter=blob:none --no-checkout https://github.com/pratikkayal/PlantDoc-Dataset.git
  (cd PlantDoc-Dataset && git sparse-checkout set --no-cone "/train/Tomato*" "/test/Tomato*" && git checkout)
fi
if [ ! -d PlantVillage-Dataset ]; then
  git clone --filter=blob:none --no-checkout https://github.com/spMohanty/PlantVillage-Dataset.git
  (cd PlantVillage-Dataset && git sparse-checkout set --no-cone "/raw/color/Tomato*" "/leaf_grouping/" && git checkout)
fi
echo "PlantDoc commit:     $(git -C PlantDoc-Dataset rev-parse HEAD)"     | tee dataset_versions.txt
echo "PlantVillage commit: $(git -C PlantVillage-Dataset rev-parse HEAD)" | tee -a dataset_versions.txt
