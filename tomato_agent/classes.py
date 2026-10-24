"""The 8 tomato classes shared by PlantVillage and PlantDoc (paper Section 4.1).

The order is fixed: model outputs, saved logits, the Android app and shared/*.json all use it.
"""
CLASSES = [
    "bacterial_spot", "early_blight", "late_blight", "leaf_mold",
    "septoria_leaf_spot", "yellow_leaf_curl_virus", "mosaic_virus", "healthy",
]
C2I = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

DISPLAY_EN = {
    "bacterial_spot": "Bacterial spot",
    "early_blight": "Early blight",
    "late_blight": "Late blight",
    "leaf_mold": "Leaf mold",
    "septoria_leaf_spot": "Septoria leaf spot",
    "yellow_leaf_curl_virus": "Yellow leaf curl virus",
    "mosaic_virus": "Mosaic virus",
    "healthy": "Healthy",
}
