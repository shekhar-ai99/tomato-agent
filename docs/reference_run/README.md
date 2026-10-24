# Reference run (stages 1-2 on the real datasets, 24 Oct 2026)

Produced in a clean environment with `scripts/download_data.sh`, `prepare_data.py` and
`set_quality_thresholds.py` (defaults: level 2, 2 copies, max FPR 3%, thresholds chosen on
PlantVillage val + PlantDoc calibration photos). Your own run should give the same numbers when
the dataset commits match. Use them to check your setup, not as paper results.

* `split_report.csv`: 15,080 PlantVillage images in the 8 shared classes (12,147 train / 1,466 val /
  1,467 test; dedup removed 46 training images) and 744 PlantDoc images (149 calibration / 595 test).
* `quality_thresholds.json`: the chosen thresholds.
* `quality_detection.csv`: share of photos flagged per set and condition. Clean photos: 2-3%
  (PlantVillage) and about 7% (PlantDoc). Gaussian blur and low light are caught almost always,
  over-exposure 70-87%, motion blur 49-77%, occlusion 86% on PlantVillage but only about 20% on
  cluttered field photos (worth reporting as a limitation of colour-based leaf visibility).
* `degraded_examples.png`: one PlantDoc photo under each condition.
