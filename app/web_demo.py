#!/usr/bin/env python3
"""
Offline web demo of the agent loop (optional, not part of the paper metrics). No extra packages: Python's
standard library serves one HTML page, bound to 127.0.0.1 only.

  python app/web_demo.py --run runs/efficientnet_b0/seed0
  python app/web_demo.py --run smoke/runs/resnet18/seed0 --quality smoke/artifacts/quality_thresholds.json \
         --thresholds smoke/results/thresholds.json

Upload a leaf photo -> quality check -> diagnosis -> accept / retake (with a tip) / defer (case saved to
cases/ for the expert). The default engine is the LangGraph loop (pip install langgraph); --engine pipeline
runs the plain-Python loop, which makes the same decisions.
"""
import argparse
import io
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from tomato_agent.classes import CLASSES
from tomato_agent.content import advice, class_name, retake_tip
from tomato_agent.gate import DEFER, RETAKE
from tomato_agent.gate import ACCEPT
from tomato_agent.io_utils import read_json
from tomato_agent.pipeline import TomatoAgentPipeline, gate_for
from tomato_agent.gate import GateConfig

PAGE = Path(__file__).with_name("web_demo.html")
MAX_BYTES = 15 * 1024 * 1024
LANGS = ("en", "hi", "or")


class Demo:
    def __init__(self, pipe, engine, gates):
        self.pipe, self.engine, self.gates = pipe, engine, gates   # gates: {"pv": GateConfig, "pd": GateConfig}
        self.cases, self.case_temp, self.lock = {}, {}, threading.Lock()

    def diagnose(self, data, lang, case_id, temp="pv"):
        image = Image.open(io.BytesIO(data)).convert("RGB")
        with self.lock:
            case = self.cases.get(case_id)
            if case is None or case.closed or self.case_temp.get(case_id) != temp:
                case_id, case = uuid.uuid4().hex[:8], self.pipe.new_case()
                self.cases[case_id], self.case_temp[case_id] = case, temp
            self.pipe.diagnosis.which_T = temp          # pv = lab-fitted temperature, pd = field-fitted temperature
            self.pipe.gate = self.gates[temp]
            self.pipe.lang = lang
            step = self.pipe.submit(case, image)
            budget = self.pipe.gate.retake_budget
        tip_key = "deferred" if step.action == DEFER else None
        if step.action == RETAKE:
            tip_key = (self.pipe.quality.assess(image).main_reason or "medium_confidence")                 if step.reason == "quality" else "medium_confidence"
        by_lang = {}
        for lg in LANGS:  # all languages go back at once, so switching language never re-runs the case
            by_lang[lg] = {
                "name": class_name(step.label, lg),
                "message": retake_tip(tip_key, lg) if tip_key else "",
                "top": [{"name": class_name(t["label"], lg), "prob": t["prob"]} for t in step.top_k()],
                "advice": advice(step.label, lg) if step.action == ACCEPT else None,
            }
        out = {
            "case_id": case_id, "action": step.action, "reason": step.reason, "label": step.label,
            "confidence": step.confidence, "quality": {k: float(v) for k, v in step.quality.items()},
            "photos": len(case.photos), "retakes_used": case.retakes_used, "retake_budget": budget,
            "closed": case.closed, "case_file": step.case_file, "engine": self.engine, "by_lang": by_lang, "temp": temp,
            "T": self.pipe.diagnosis.runs[0].temperature(temp),
            "gate": {"accept": self.pipe.gate.tau_accept, "defer": self.pipe.gate.tau_defer},
        }
        return out


def make_handler(demo):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            body = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, PAGE.read_bytes(), "text/html")
            elif path == "/api/info":
                g = demo.pipe.gate
                self._send(200, {"engine": demo.engine, "accept": g.tau_accept, "defer": g.tau_defer,
                                 "retake_budget": g.retake_budget, "classes": list(CLASSES),
                                 "gates": {k: {"accept": v.tau_accept, "defer": v.tau_defer} for k, v in demo.gates.items()}})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/api/diagnose":
                return self._send(404, {"error": "not found"})
            q = parse_qs(url.query)
            lang = q.get("lang", ["en"])[0]
            temp = q.get("temp", ["pv"])[0]
            n = int(self.headers.get("Content-Length", 0))
            if lang not in LANGS or temp not in demo.gates or not 0 < n <= MAX_BYTES:
                return self._send(400, {"error": "bad request (photo must be under 15 MB)"})
            try:
                self._send(200, demo.diagnose(self.rfile.read(n), lang, q.get("case", [""])[0], temp))
            except Exception as e:  # unreadable image etc.
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--quality", default="artifacts/quality_thresholds.json")
    ap.add_argument("--thresholds", default="results/thresholds.json")
    ap.add_argument("--cost", default="10:1")
    ap.add_argument("--cases", default="cases")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--engine", choices=["langgraph", "pipeline"], default="langgraph")
    args = ap.parse_args()
    if args.engine == "langgraph":
        from tomato_agent.graph import GraphPipeline as Engine
    else:
        Engine = TomatoAgentPipeline
    pipe = Engine.from_artifacts(
        args.run, args.quality if Path(args.quality).exists() else None,
        args.thresholds if Path(args.thresholds).exists() else None, cost_label=args.cost, cases_dir=args.cases)
    th = read_json(args.thresholds) if Path(args.thresholds).exists() else None
    bb = Path(args.run[0]).parent.name
    gates = {t: (gate_for(th, args.cost, t, bb) if th else GateConfig()) for t in ("pv", "pd")}
    for t in gates:
        print(f"gate for temperature {t}: accept >= {gates[t].tau_accept:.4f}, defer < {gates[t].tau_defer:.4f}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Demo(pipe, args.engine, gates)))
    print(f"PatraSathi demo ({args.engine}) at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
