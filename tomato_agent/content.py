"""Farmer-facing content (retake tips, advice cards) in English, Hindi and Odia, from shared/*.json."""
import json
from functools import lru_cache
from pathlib import Path

SHARED = Path(__file__).resolve().parent.parent / "shared"
LANGS = ("en", "hi", "or")


@lru_cache(maxsize=None)
def _load(name):
    return json.loads((SHARED / name).read_text(encoding="utf-8"))


def retake_tip(reason, lang="en"):
    tips = _load("retake_tips.json")
    entry = tips.get(reason) or tips["medium_confidence"]
    return entry.get(lang) or entry["en"]


def advice(label, lang="en"):
    a = _load("advice.json")[label]
    pick = lambda d: d.get(lang) or d.get("en")
    return {k: pick(a[k]) for k in ("name", "about", "do_now", "organic", "chemical", "prevention")} | {
        "disclaimer": pick(_load("advice.json")["_meta"]["disclaimer"])}


def class_name(label, lang="en"):
    return advice(label, lang)["name"]
