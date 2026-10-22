"""One-off: build data/bank.json, a ~150-question 'Light' chapter bank used as the
existing collection in the demo (in addition to the Q1–Q10 test corpus).

Real banks contain the same question worded several ways, so the prompt asks
for some paraphrase families; that is what context clustering must compress.
The output is committed; re-run only to refresh it (needs OPENROUTER_API_KEY).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qgen.llm import chat_json  # noqa: E402

SUBTOPICS = [
    ("reflection", "reflection of light, laws of reflection, regular vs diffuse reflection, plane mirror images"),
    ("spherical mirrors", "concave and convex mirrors, focus, centre of curvature, ray diagrams, uses"),
    ("mirror formula", "mirror formula, magnification, sign convention, numericals"),
    ("refraction", "refraction of light, laws of refraction, refractive index, glass slab, apparent depth"),
    ("lenses", "convex and concave lenses, lens formula, power of a lens, numericals"),
    ("human eye", "parts of the eye, accommodation, myopia, hypermetropia, correction"),
    ("dispersion and scattering", "prism, dispersion, rainbow, scattering, blue sky, red sunset, Tyndall effect"),
]

SYSTEM = ("You write school physics exam questions (grade 9-11). Return only JSON "
          "{\"questions\": [{\"text\": str, \"difficulty\": \"easy\"|\"medium\"|\"hard\", "
          "\"type\": \"definition\"|\"conceptual\"|\"numerical\"|\"diagram\"|\"application\"|\"reasoning\"}]}")


def main():
    bank = []
    for topic, scope in SUBTOPICS:
        data = chat_json(SYSTEM, f"Write 20 questions on '{topic}' covering: {scope}. Make about 16 distinct "
                                 "questions; for 2 of them also add 2 alternative wordings each (same question, "
                                 "different phrasing), as real question banks contain such repeats.",
                         model="anthropic/claude-sonnet-5.5", temperature=0.7)
        for q in data["questions"]:
            bank.append({"text": q["text"], "tags": {"chapter": "light", "topic": topic,
                                                     "difficulty": q.get("difficulty", "medium"),
                                                     "type": q.get("type", "conceptual")}})
        print(topic, len(data["questions"]))
    (ROOT / "data" / "bank.json").write_text(json.dumps({"questions": bank}, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    print("total", len(bank))


if __name__ == "__main__":
    main()
