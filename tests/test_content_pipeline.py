import json

import numpy as np
import pytest
from PIL import Image

from tomato_agent.classes import CLASSES
from tomato_agent.content import LANGS, SHARED, advice, retake_tip
from tomato_agent.gate import ACCEPT, DEFER, RETAKE, GateConfig
from tomato_agent.pipeline import TomatoAgentPipeline
from tomato_agent.quality import CaptureQualityAgent


def test_content_complete_in_all_languages():
    tips = json.loads((SHARED / "retake_tips.json").read_text(encoding="utf-8"))
    for key in ("no_leaf", "blurry", "too_dark", "too_bright", "medium_confidence", "deferred"):
        for L in LANGS:
            assert tips[key][L].strip()
    for c in CLASSES:
        for L in LANGS:
            a = advice(c, L)
            assert a["name"] and a["about"] and a["do_now"] and a["disclaimer"]
    assert retake_tip("unknown_reason", "or") == tips["medium_confidence"]["or"]


def test_prepare_data_classes_match_package():
    import importlib.util
    from pathlib import Path
    for script in ("prepare_data.py", "train_diagnosis.py"):
        p = Path(__file__).resolve().parents[1] / "scripts" / script
        src = p.read_text(encoding="utf-8")
        start = src.index("CLASSES = [")
        ns = {}
        exec(src[start:src.index("]", start) + 1], ns)
        assert ns["CLASSES"] == CLASSES, script


class FakeDiagnosis:
    def __init__(self, seq):
        self.seq = list(seq)

    def probs(self, image):
        conf, k = self.seq.pop(0)
        p = np.full(len(CLASSES), (1 - conf) / (len(CLASSES) - 1))
        p[k] = conf
        return p


def good_leaf():
    a = np.zeros((256, 256, 3), np.uint8)
    a[..., 1] = 140
    a[..., 0] = 70
    a[::4, :, :] = 30  # texture so it is sharp
    return Image.fromarray(a)


def make(seq, tmp_path, R=1):
    return TomatoAgentPipeline(CaptureQualityAgent(), FakeDiagnosis(seq), GateConfig(0.8, 0.4, R),
                               lang="en", expert_agent=__import__("tomato_agent.expert", fromlist=["x"]).ExpertReferralAgent(tmp_path))


def test_flow_retake_then_accept(tmp_path):
    pipe = make([(0.6, 1), (0.9, 1)], tmp_path)
    case = pipe.new_case()
    s1 = pipe.submit(case, good_leaf())
    assert s1.action == RETAKE and s1.message
    s2 = pipe.submit(case, good_leaf())
    assert s2.action == ACCEPT and s2.label == CLASSES[1] and case.closed


def test_flow_quality_retake_then_defer_writes_case(tmp_path):
    pipe = make([(0.95, 0), (0.6, 0)], tmp_path)
    case = pipe.new_case()
    dark = Image.new("RGB", (256, 256), (5, 8, 5))
    s1 = pipe.submit(case, dark)
    assert s1.action == RETAKE and s1.reason == "quality"
    s2 = pipe.submit(case, good_leaf())
    assert s2.action == DEFER and s2.case_file
    rec = json.loads(open(s2.case_file, encoding="utf-8").read())
    assert rec["reason"] == "retakes_exhausted" and len(rec["top_k"]) == 3 and len(rec["history"]) == 2
    with pytest.raises(RuntimeError):
        pipe.submit(case, good_leaf())


def test_langgraph_matches_pipeline(tmp_path):
    pytest.importorskip("langgraph")
    from tomato_agent.expert import ExpertReferralAgent
    from tomato_agent.graph import build_graph, run
    g = build_graph(CaptureQualityAgent(), FakeDiagnosis([(0.6, 2), (0.95, 2)]), GateConfig(0.8, 0.4, 1),
                    ExpertReferralAgent(tmp_path))
    s = run(g, {"retakes_used": 0, "history": []}, good_leaf())
    assert s["action"] == RETAKE
    s = run(g, s, good_leaf())
    assert s["action"] == ACCEPT and len(s["history"]) == 2
