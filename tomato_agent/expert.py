"""Expert-referral agent (paper Section 3.5).

Packages a deferred case for an agronomist: photos, quality assessment, top-k candidates with
probabilities, entropy and the reason for deferral. In the paper the expert is simulated by the
ground-truth label (oracle). The same JSON layout is written by the Android app, so cases exported
from phones can be loaded here for retraining (active learning).
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .calibration import entropy
from .classes import CLASSES

CASE_SCHEMA_VERSION = 1


def top_k(probs, k=3):
    idx = np.argsort(-np.asarray(probs))[:k]
    return [{"label": CLASSES[i], "prob": round(float(probs[i]), 4)} for i in idx]


def build_case(probs, reason, photos=(), quality=None, history=None, source="tomato-agent", extra=None):
    return {
        "schema": CASE_SCHEMA_VERSION,
        "case_id": uuid.uuid4().hex[:12],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "reason": reason,
        "top_k": top_k(probs),
        "entropy": round(float(entropy(np.asarray(probs)[None])[0]), 4),
        "quality": quality or {},
        "photos": [str(p) for p in photos],
        "history": history or [],
        "expert_label": None,
        "expert_note": None,
        **(extra or {}),
    }


class ExpertReferralAgent:
    def __init__(self, out_dir="cases"):
        self.out_dir = Path(out_dir)

    def refer(self, case):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"case_{case['case_id']}.json"
        path.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def oracle_answer(true_label):
        """Simulated expert used in the paper: always returns the ground truth."""
        return true_label
