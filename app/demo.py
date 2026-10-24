#!/usr/bin/env python3
"""
Offline demo of the agent loop (optional, not part of the paper metrics). Runs on localhost only.

  python app/demo.py --run runs/efficientnet_b0/seed0
  python app/demo.py --run runs/efficientnet_b0/seed0 runs/resnet50/seed0     # small ensemble

Upload a leaf photo -> quality check -> diagnosis -> accept / retake (with a tip) / defer (case saved
to cases/ for the expert). A retake continues the same case until the budget is used.
Tick "Explain with local LLM" only if Ollama is running on this machine (no internet, no API key).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from tomato_agent import explain as llm
from tomato_agent.content import advice, class_name
from tomato_agent.gate import ACCEPT, DEFER, RETAKE
from tomato_agent.pipeline import TomatoAgentPipeline

LANGS = {"English": "en", "हिन्दी": "hi", "ଓଡ଼ିଆ": "or"}
UI = {
    "en": {ACCEPT: "✅ Diagnosis", RETAKE: "📷 Please retake the photo", DEFER: "🧑‍🌾 Sent to an expert",
           "do_now": "Do now", "organic": "Organic / cultural", "chemical": "If needed", "prevention": "Prevention",
           "conf": "calibrated confidence", "top": "Top candidates", "photo": "Photo", "retakes": "retakes used"},
    "hi": {ACCEPT: "✅ निदान", RETAKE: "📷 कृपया फिर से फ़ोटो लें", DEFER: "🧑‍🌾 विशेषज्ञ को भेजा गया",
           "do_now": "अभी करें", "organic": "जैविक / खेती के उपाय", "chemical": "ज़रूरत हो तो", "prevention": "रोकथाम",
           "conf": "विश्वसनीयता", "top": "संभावित रोग", "photo": "फ़ोटो", "retakes": "दोबारा फ़ोटो"},
    "or": {ACCEPT: "✅ ରୋଗ ନିର୍ଣ୍ଣୟ", RETAKE: "📷 ଦୟାକରି ପୁଣି ଫଟୋ ନିଅନ୍ତୁ", DEFER: "🧑‍🌾 ବିଶେଷଜ୍ଞଙ୍କୁ ପଠାଗଲା",
           "do_now": "ଏବେ କରନ୍ତୁ", "organic": "ଜୈବିକ / ଚାଷ ଉପାୟ", "chemical": "ଦରକାର ହେଲେ", "prevention": "ପ୍ରତିରୋଧ",
           "conf": "ବିଶ୍ୱସନୀୟତା", "top": "ସମ୍ଭାବ୍ୟ ରୋଗ", "photo": "ଫଟୋ", "retakes": "ପୁଣି ଫଟୋ"},
}


def render(step, case, lang, budget, use_llm, image):
    L = LANGS[lang]
    t = UI[L]
    md = [f"## {t[step.action]}", f"*{t['photo']} {len(case.photos)} · {t['retakes']} {case.retakes_used}/{budget}*"]
    if step.action == ACCEPT:
        a = advice(step.label, L)
        md += [f"### {a['name']}  ({step.confidence:.0%} {t['conf']})", a["about"]]
        for key in ("do_now", "organic", "chemical", "prevention"):
            if a[key]:
                md += [f"**{t[key]}**", "\n".join(f"- {x}" for x in a[key])]
        md.append(f"> {a['disclaimer']}")
        if use_llm:
            try:
                md += ["**Local LLM explanation (does not change the diagnosis):**",
                       llm.explain(a["name"], step.confidence, a["about"], L)]
            except Exception as e:
                md.append(f"*(local LLM not reachable: {e})*")
    else:
        md.append(step.message)
    if step.action == DEFER and step.case_file:
        md.append(f"Case file: `{step.case_file}`")
    md.append(f"**{t['top']}:** " + ", ".join(f"{class_name(t['label'], L)} {t['prob']:.0%}" for t in step.top_k()))
    q = step.quality
    md.append(f"<sub>quality: sharpness {q['sharpness']:.0f}, brightness {q['lum_mean']:.2f}, "
              f"leaf {q['leaf_frac']:.0%} · decision reason: {step.reason}</sub>")
    return "\n\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--quality", default="artifacts/quality_thresholds.json")
    ap.add_argument("--thresholds", default="results/thresholds.json")
    ap.add_argument("--cost", default="10:1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    pipe = TomatoAgentPipeline.from_artifacts(
        args.run, args.quality if Path(args.quality).exists() else None,
        args.thresholds if Path(args.thresholds).exists() else None, cost_label=args.cost)
    budget = pipe.gate.retake_budget

    def diagnose(image, lang, use_llm, case):
        if image is None:
            return "Upload or take a photo of one tomato leaf.", case
        if case is None or case.closed:
            case = pipe.new_case()
        pipe.lang = LANGS[lang]
        step = pipe.submit(case, image)
        return render(step, case, lang, budget, use_llm, image), case

    with gr.Blocks(title="PatraSathi · tomato-agent demo") as demo:
        gr.Markdown("# 🍅 PatraSathi · ପତ୍ରସାଥୀ\nConfidence-gated tomato leaf diagnosis, fully offline. "
                    f"Gate: accept ≥ {pipe.gate.tau_accept:.2f}, defer < {pipe.gate.tau_defer:.2f}, "
                    f"up to {budget} retake(s).")
        case = gr.State(None)
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Tomato leaf photo", sources=["upload", "webcam"])
                lang = gr.Radio(list(LANGS), value="English", label="Language")
                use_llm = gr.Checkbox(False, label="Explain with local LLM (Ollama)", interactive=llm.available())
                with gr.Row():
                    go = gr.Button("Diagnose", variant="primary")
                    new = gr.Button("New case")
            out = gr.Markdown()
        go.click(diagnose, [img, lang, use_llm, case], [out, case])
        new.click(lambda: (None, "", None), None, [case, out, img])
    demo.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
