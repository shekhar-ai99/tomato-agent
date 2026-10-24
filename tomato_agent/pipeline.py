"""The live agent loop: one farmer case, photo by photo (paper Figure 1).

    pipe = TomatoAgentPipeline.from_artifacts("runs/efficientnet_b0/seed0", "artifacts/quality_thresholds.json",
                                              "results/thresholds.json")
    case = pipe.new_case()
    step = pipe.submit(case, PIL.Image.open("leaf.jpg"))
    step.action  -> "accept" | "retake" | "defer"

This is a plain-Python state machine (no framework needed). tomato_agent.graph wraps the same
agents in LangGraph if you prefer that orchestration.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .classes import CLASSES
from .content import retake_tip
from .expert import ExpertReferralAgent, build_case, top_k
from .gate import ACCEPT, DEFER, RETAKE, GateConfig, decide
from .quality import CaptureQualityAgent


def gate_for(thresholds, cost_label="10:1", temperature="pv", backbone=None):
    """Gate settings for one temperature variant ('pv' = lab-fitted, 'pd' = field-fitted) from results/thresholds.json:
    the median over seeds of the per-run choices (falls back to the 'recommended' entry for older files)."""
    import numpy as np
    per = [v["proposed"] for k, v in (thresholds.get("per_run") or {}).items()
           if k.endswith(f"/T{temperature}/{cost_label}") and (backbone is None or k.startswith(f"{backbone}/"))]
    if per:
        return GateConfig(float(np.median([p["tau_accept"] for p in per])), float(np.median([p["tau_defer"] for p in per])),
                          int(np.median([p.get("retake_budget", 1) for p in per])))
    rec = thresholds.get("recommended", {})
    if not rec:
        raise ValueError("thresholds file has no recommended thresholds; run scripts/run_policy.py")
    pick = rec.get(cost_label) or next(iter(rec.values()))
    return GateConfig(pick["tau_accept"], pick["tau_defer"], pick.get("retake_budget", 1))


@dataclass
class Case:
    retakes_used: int = 0
    history: list = field(default_factory=list)
    photos: list = field(default_factory=list)
    closed: bool = False


@dataclass
class Step:
    action: str
    reason: str
    message: str
    quality: dict
    probs: Optional[np.ndarray] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    case_file: Optional[str] = None

    def top_k(self, k=3):
        return top_k(self.probs, k) if self.probs is not None else []


class TomatoAgentPipeline:
    def __init__(self, quality_agent, diagnosis_agent, gate_cfg, expert_agent=None, lang="en"):
        self.quality = quality_agent
        self.diagnosis = diagnosis_agent
        self.gate = gate_cfg
        self.expert = expert_agent or ExpertReferralAgent()
        self.lang = lang

    @classmethod
    def from_artifacts(cls, run_dirs, quality_file=None, thresholds_file=None, temperature="pv",
                       cost_label="10:1", cases_dir="cases", lang="en"):
        from .io_utils import read_json
        from .models import DiagnosisAgent
        run_dirs = [run_dirs] if isinstance(run_dirs, (str, bytes)) or hasattr(run_dirs, "joinpath") else run_dirs
        qa = CaptureQualityAgent.from_file(quality_file) if quality_file else CaptureQualityAgent()
        cfg = GateConfig()
        if thresholds_file:
            bb = Path(run_dirs[0]).parent.name if run_dirs else None
            cfg = gate_for(read_json(thresholds_file), cost_label, temperature, bb)
        return cls(qa, DiagnosisAgent(run_dirs, temperature), cfg, ExpertReferralAgent(cases_dir), lang)

    def new_case(self):
        return Case()

    def submit(self, case, image, photo_ref=None):
        if case.closed:
            raise RuntimeError("case already closed; start a new_case()")
        case.photos.append(photo_ref or f"photo_{len(case.photos)}")
        q = self.quality.assess(image)
        probs = self.diagnosis.probs(image) if self.diagnosis is not None else np.full(len(CLASSES), 1 / len(CLASSES))
        conf, k = float(probs.max()), int(probs.argmax())
        left = self.gate.retake_budget - case.retakes_used
        action, reason = decide(conf, q.ok, left, self.gate)
        case.history.append({"action": action, "reason": reason, "quality_reasons": q.reasons,
                             "label": CLASSES[k], "confidence": round(conf, 4)})
        step = Step(action, reason, "", q.features, probs, CLASSES[k], conf)
        if action == RETAKE:
            case.retakes_used += 1
            step.message = retake_tip(q.main_reason if reason == "quality" else "medium_confidence", self.lang)
        elif action == ACCEPT:
            case.closed = True
        elif action == DEFER:
            case.closed = True
            rec = build_case(probs, reason, case.photos, {"reasons": q.reasons, **q.features}, case.history)
            step.case_file = str(self.expert.refer(rec))
            step.message = retake_tip("deferred", self.lang)
        return step
