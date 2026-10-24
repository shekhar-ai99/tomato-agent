"""Diagnosis agent: loads a trained classifier (from scripts/train_diagnosis.py) and returns logits."""
import json
from pathlib import Path

import numpy as np

from .classes import CLASSES


def build_model(backbone, img_size=224, pretrained=False):
    import timm
    kwargs = {"img_size": img_size} if backbone.startswith("vit") else {}
    return timm.create_model(backbone, pretrained=pretrained, num_classes=len(CLASSES), **kwargs)


def eval_transform(model, img_size):
    """Exactly the evaluation transform of train_diagnosis.py: square resize, no centre crop."""
    import timm
    from torchvision import transforms as T
    cfg = timm.data.resolve_data_config({}, model=model)
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])]), cfg


class LoadedRun:
    """One trained (backbone, seed) run folder."""

    def __init__(self, run_dir, device=None):
        import torch
        self.run_dir = Path(run_dir)
        self.config = json.loads((self.run_dir / "config.json").read_text())
        self.backbone = self.config["backbone"]
        self.seed = int(self.config["seed"])
        self.img_size = int(self.config.get("img_size", 224))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.backbone, self.img_size, pretrained=False)
        self.model.load_state_dict(torch.load(self.run_dir / "best.pt", map_location="cpu"))
        self.model.to(self.device).eval()
        self.tf, self.data_cfg = eval_transform(self.model, self.img_size)
        cal = self.run_dir / "calibration.json"
        self.calibration = json.loads(cal.read_text()) if cal.exists() else {}

    def temperature(self, which="pv"):
        return float(self.calibration.get(f"T_{which}", 1.0)) if which != "none" else 1.0

    def logits(self, pil_images, batch_size=64):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(pil_images), batch_size):
                x = torch.stack([self.tf(im.convert("RGB")) for im in pil_images[i:i + batch_size]])
                out.append(self.model(x.to(self.device)).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(CLASSES)), np.float32)


class DiagnosisAgent:
    """Classifier (or small ensemble of classifiers) + temperature -> calibrated probabilities."""

    def __init__(self, runs, temperature="pv"):
        self.runs = [r if isinstance(r, LoadedRun) else LoadedRun(r) for r in runs]
        self.which_T = temperature

    def probs(self, pil_image):
        from .calibration import softmax
        ps = [softmax(r.logits([pil_image]), r.temperature(self.which_T))[0] for r in self.runs]
        return np.mean(ps, axis=0)
