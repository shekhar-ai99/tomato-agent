"""Optional LangGraph version of the agent loop (pip install langgraph). Runs locally, no LLM needed.

Nodes: quality -> diagnose -> verify -> (retake | accept | refer). The retake edge ends the graph run
and returns a request for a new photo; call `run(graph, state, new_image)` again with the same state.
Same decisions as tomato_agent.pipeline, expressed as a graph so it can later grow the v2.0 nodes
(VLM verifier, advisory RAG) without touching the core agents.
"""
from typing import Any, Optional, TypedDict

from .classes import CLASSES
from .expert import build_case
from .gate import ACCEPT, DEFER, RETAKE, decide


class CaseState(TypedDict, total=False):
    image: Any
    retakes_used: int
    history: list
    quality_ok: bool
    quality_reasons: list
    quality_features: dict
    probs: Any
    action: Optional[str]
    reason: Optional[str]
    case_record: Optional[dict]


def build_graph(quality_agent, diagnosis_agent, gate_cfg, expert_agent):
    from langgraph.graph import END, StateGraph

    def quality(s):
        q = quality_agent.assess(s["image"])
        return {"quality_ok": q.ok, "quality_reasons": q.reasons, "quality_features": q.features}

    def diagnose(s):
        return {"probs": diagnosis_agent.probs(s["image"])}

    def verify(s):
        conf = float(s["probs"].max())
        left = gate_cfg.retake_budget - s.get("retakes_used", 0)
        action, reason = decide(conf, s["quality_ok"], left, gate_cfg)
        hist = s.get("history", []) + [{"action": action, "reason": reason,
                                        "label": CLASSES[int(s["probs"].argmax())], "confidence": conf}]
        out = {"action": action, "reason": reason, "history": hist}
        if action == RETAKE:
            out["retakes_used"] = s.get("retakes_used", 0) + 1
        return out

    def refer(s):
        rec = build_case(s["probs"], s["reason"], quality={"reasons": s["quality_reasons"]}, history=s["history"])
        expert_agent.refer(rec)
        return {"case_record": rec}

    g = StateGraph(CaseState)
    g.add_node("quality", quality)
    g.add_node("diagnose", diagnose)
    g.add_node("verify", verify)
    g.add_node("refer", refer)
    g.set_entry_point("quality")
    g.add_edge("quality", "diagnose")
    g.add_edge("diagnose", "verify")
    g.add_conditional_edges("verify", lambda s: s["action"], {ACCEPT: END, RETAKE: END, DEFER: "refer"})
    g.add_edge("refer", END)
    return g.compile()


def run(graph, state, image):
    return graph.invoke({**state, "image": image})


class _RecordingExpert:
    """Wraps the expert agent so the adapter can report where the case file was written."""

    def __init__(self, expert):
        self.expert = expert
        self.last_path = None

    def refer(self, case):
        self.last_path = self.expert.refer(case)
        return self.last_path


from .content import retake_tip  # noqa: E402
from .pipeline import Step, TomatoAgentPipeline  # noqa: E402


class GraphPipeline(TomatoAgentPipeline):
    """TomatoAgentPipeline whose submit() runs the LangGraph above (same Case/Step API, so the demo can use either)."""

    def submit(self, case, image, photo_ref=None):
        if case.closed:
            raise RuntimeError("case already closed; start a new_case()")
        recorder = _RecordingExpert(self.expert)
        graph = build_graph(self.quality, self.diagnosis, self.gate, recorder)
        case.photos.append(photo_ref or f"photo_{len(case.photos)}")
        out = run(graph, {"retakes_used": case.retakes_used, "history": list(case.history)}, image)
        case.history = out["history"]
        probs = out["probs"]
        action, reason = out["action"], out["reason"]
        reasons = out.get("quality_reasons") or []
        step = Step(action, reason, "", out["quality_features"], probs, CLASSES[int(probs.argmax())], float(probs.max()))
        if action == RETAKE:
            case.retakes_used = out["retakes_used"]
            step.message = retake_tip(reasons[0] if reason == "quality" and reasons else "medium_confidence", self.lang)
        elif action == ACCEPT:
            case.closed = True
        else:
            case.closed = True
            step.case_file = str(recorder.last_path)
            step.message = retake_tip("deferred", self.lang)
        return step
