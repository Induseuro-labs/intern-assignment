"""Walkthrough for question 2.

    python -m q2_duplicate_questions.demo            # offline (scripted LLM)
    python -m q2_duplicate_questions.demo --claude   # real generation + LLM judge

Writes examples/q2/DEMO_LOG.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .core import QuestionBank, TfidfEmbedder, default_embedder
from .data import CORPUS, TAGS
from .detector import DuplicateDetector, llm_judge
from .generator import ClaudeQuestionLLM, ScriptedLLM, generate

# What an LLM that only half-follows instructions might return for "reflection of light".
SCRIPTED = [
    "Define reflection of light.",                                   # paraphrase of Q1
    "What is a plane mirror?",
    "Distinguish between regular and diffuse reflection.",           # paraphrase of Q9
    "What is the angle of incidence?",
    "Why is the image formed by a plane mirror laterally inverted?",
    "State the laws of reflection of light!",                        # exact dup of Q5 (punctuation)
    "If the angle of incidence is 50 degrees, what is the angle of reflection?",  # same template as Q7
    "What is the normal at the point of incidence?",
    "Give two examples of regular reflection in daily life.",
    "What is meant by a plane mirror?",                              # paraphrase of an accepted one
    "How many images are formed when two plane mirrors are placed at 90 degrees to each other?",
    "What is a virtual image?",
    "What is the use of a periscope?",
]

lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude", action="store_true")
    ap.add_argument("--tfidf", action="store_true", help="force the lightweight embedder")
    args = ap.parse_args()

    bank = QuestionBank(TfidfEmbedder() if args.tfidf else default_embedder())
    for qid, text, _ in CORPUS:
        bank.add(text, TAGS, id=qid)
    # a different chapter: tags scope generation context, never duplicate decisions
    bank.add("What is refraction of light?", {"subject": "physics", "chapter": "light", "topic": "refraction"})
    bank.add("State Snell's law.", {"subject": "physics", "chapter": "light", "topic": "refraction",
                                    "difficulty": "easy"})
    judge = llm_judge if args.claude else None
    det = DuplicateDetector(bank, judge=judge)

    log("# Question 2 demo log\n")
    log(f"Embedder: `{bank.embedder.name}`; LLM judge: {'on' if judge else 'off'}\n")
    log("## Independent duplicate detection\n")
    log("| candidate | verdict | matched | why |")
    log("|---|---|---|---|")
    for cand in [
        "define the REFLECTION of light",                      # exact (case/punctuation)
        "What is meant by reflection of light?",               # semantic
        "Briefly explain what reflection of light is.",        # semantic
        "Differentiate between diffuse and regular reflection.",  # semantic
        "If the angle of incidence is 45°, find the angle of reflection.",  # same numeric template
        "Explain reflection in general terms.",                # same as Q8 'What is reflection?'
        "What is refraction of light?",                        # exact, though it lives under other tags
        "What is the angle of incidence?",                     # different, same subject
        "What is total internal reflection?",                  # different
        "Why does a plane mirror form a laterally inverted image?",  # different
    ]:
        v = det.check(cand)
        log(f"| {cand} | **{v.label}** | {v.match_text or ''} | {v.reason} |")

    log("\n## Generation with avoidance + safeguard\n")
    llm = ClaudeQuestionLLM() if args.claude else ScriptedLLM(SCRIPTED)
    report = generate(bank, "reflection of light", n=6, llm=llm, tags={"topic": "reflection"}, detector=det)
    log(f"LLM backend: `{llm.name}`, rounds: {len(report.prompts)}\n")
    log("### Prompt sent in round 1 (bounded context, not the whole bank)\n")
    log("```text\n" + report.prompts[0] + "\n```\n")
    log("### Accepted (added to bank with origin=generated)\n")
    for q in report.accepted:
        log(f"- {q}")
    log("\n### Rejected by the safeguard (fed back to the next round)\n")
    for r in report.rejected:
        log(f"- {r['question']} -> **{r['verdict']}** of \"{r['match']}\" ({r['reason']})")
    log("\n### Sent to human review\n")
    for r in report.review or [{"question": "(none)", "verdict": "", "match": "", "reason": ""}]:
        log(f"- {r['question']} {('-> ' + r['reason']) if r['reason'] else ''}")
    if len(report.prompts) > 1:
        log("\n### Round 2 prompt tail (shows rejected candidates fed back)\n")
        log("```text\n" + report.prompts[1].split("Candidates already rejected")[1] + "\n```")

    out = Path("examples/q2")
    out.mkdir(parents=True, exist_ok=True)
    (out / "DEMO_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
