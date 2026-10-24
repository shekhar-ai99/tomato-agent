"""Optional, demo-only plain-language explanation from a LOCAL model served by Ollama.

Never used in the paper metrics and never changes the decision: the calibrated CNN/ViT decides,
the language model only explains. Needs Ollama running on this machine (http://localhost:11434)
with e.g. `ollama pull llama3.1:8b` (text) or `ollama pull qwen2.5vl:7b` (vision). No internet, no
API keys.
"""
import base64
import io
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
LANG_NAME = {"en": "English", "hi": "Hindi", "or": "Odia"}


def available(timeout=1.5):
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def explain(label_name, confidence, about, lang="en", model="llama3.1:8b", image=None, timeout=120):
    prompt = (
        f"You help a tomato farmer in Odisha. A calibrated image classifier diagnosed '{label_name}' "
        f"with {confidence:.0%} confidence. Background: {about}\n"
        f"In {LANG_NAME.get(lang, 'English')}, in at most 4 short sentences, explain what this disease is "
        "and what to check on the plant to confirm it. Do NOT change the diagnosis and do NOT recommend "
        "doses; say to confirm with the local KVK before spraying."
    )
    body = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    if image is not None:  # vision model, e.g. qwen2.5vl:7b
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        body["images"] = [base64.b64encode(buf.getvalue()).decode()]
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"].strip()
