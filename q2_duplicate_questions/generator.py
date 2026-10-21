"""Generation-time duplicate avoidance.

The prompt never contains the whole collection. It contains a bounded
"coverage brief" built from the bank:

  * nearest neighbours  - the k questions most similar to the requested topic
                          (where duplicates are most likely to come from)
  * cluster exemplars   - k-means over the scoped pool's embeddings, one
                          representative per cluster, so the model sees the
                          *breadth* of what exists, not just the nearest corner
  * covered angles      - the most frequent (intent, concept) signatures, a
                          compact list of already-asked angles
  * recent rejects      - candidates the safeguard rejected in earlier rounds

Size is O(k) regardless of bank size; tags only scope which questions are
considered (subject/chapter), never whether two questions are duplicates.

Every candidate then goes through the independent DuplicateDetector; rejected
ones are fed back into the next round.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .core import QuestionBank, normalize, signature
from .detector import DuplicateDetector


@dataclass
class Context:
    nearest: list[str]
    exemplars: list[str]
    covered_angles: list[str]
    pool_size: int


def build_context(bank: QuestionBank, topic: str, tags: dict | None = None, k_near: int = 15,
                  k_clusters: int = 8, max_angles: int = 25) -> Context:
    mask = bank.filter_mask(tags) if len(bank) else np.zeros(0, bool)
    pool = np.flatnonzero(mask)
    nearest = []
    for i, _ in (bank.search(topic, k_near, mask) if len(pool) else []):
        if normalize(bank.questions[i].text) not in map(normalize, nearest):
            nearest.append(bank.questions[i].text)

    exemplars = []
    if len(pool) > k_clusters:
        from sklearn.cluster import KMeans  # MiniBatchKMeans / precomputed clusters for very large pools
        vecs = bank.vectors[pool]
        km = KMeans(n_clusters=k_clusters, n_init=3, random_state=0).fit(vecs)
        for c in range(k_clusters):
            members = np.flatnonzero(km.labels_ == c)
            best = members[np.argmax(vecs[members] @ km.cluster_centers_[c])]
            exemplars.append(bank.questions[pool[best]].text)
    exemplars = [e for e in exemplars if e not in nearest]

    angles = Counter()
    for i in pool:
        sig = signature(bank.questions[i].text)
        angles[f"{sig.intent}: {' '.join(sorted(sig.concepts))}"] += 1
    return Context(nearest, exemplars, [a for a, _ in angles.most_common(max_angles)], len(pool))


def render_prompt(topic: str, n: int, ctx: Context, rejected: list[str]) -> str:
    bullet = lambda xs: "\n".join(f"- {x}" for x in xs) or "- (none)"
    return f"""You write exam questions. Topic: {topic}

The question bank already holds {ctx.pool_size} questions in this scope. Do NOT produce any question
that asks the same thing as an existing one, even in different words (e.g. "Define X" duplicates
"What is X?"). Each new question must test a concept or angle not yet covered.

Existing questions closest to this topic:
{bullet(ctx.nearest)}

Representative questions from elsewhere in the bank (shows breadth):
{bullet(ctx.exemplars)}

Angles already covered (intent: concept terms):
{bullet(ctx.covered_angles)}

Candidates already rejected as duplicates - do not repeat or rephrase them:
{bullet(rejected)}

Write {n} new, mutually distinct questions. For each give the question, its intent
(definition / laws / numerical / difference / reason / process / examples / application) and the
specific concept it tests."""


_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["questions"],
    "properties": {"questions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["question", "intent", "concept"],
        "properties": {"question": {"type": "string"}, "intent": {"type": "string"},
                       "concept": {"type": "string"}}}}},
}


class ClaudeQuestionLLM:
    name = "claude"

    def __init__(self, ask_json=None):
        if ask_json is None:
            from common.llm import ask_json
        self.ask_json = ask_json

    def __call__(self, prompt: str, n: int) -> list[str]:
        return [q["question"] for q in self.ask_json(prompt, _SCHEMA)["questions"]][:n]


class ScriptedLLM:
    """Offline stand-in for an LLM that *sometimes ignores instructions*.

    It returns canned candidates in order - including paraphrases of existing
    questions - so the independent safeguard is exercised deterministically.
    """

    name = "scripted-offline"

    def __init__(self, candidates: list[str]):
        self.queue = list(candidates)

    def __call__(self, prompt: str, n: int) -> list[str]:
        out, self.queue = self.queue[:n], self.queue[n:]
        return out


@dataclass
class GenerationReport:
    accepted: list[str] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    review: list[dict] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)


def generate(bank: QuestionBank, topic: str, n: int, llm, tags: dict | None = None,
             detector: DuplicateDetector | None = None, max_rounds: int = 3,
             oversample: float = 1.5) -> GenerationReport:
    """Generate up to ``n`` new questions; accepted ones are added to the bank."""
    detector = detector or DuplicateDetector(bank)
    report = GenerationReport()
    for _ in range(max_rounds):
        need = n - len(report.accepted)
        if need <= 0:
            break
        ctx = build_context(bank, topic, tags)
        prompt = render_prompt(topic, need, ctx, [r["question"] for r in report.rejected][-20:])
        report.prompts.append(prompt)
        candidates = llm(prompt, max(need, round(need * oversample)))
        if not candidates:
            break
        for cand in candidates:
            if len(report.accepted) >= n:
                break
            v = detector.check(cand)  # independent of the prompt above
            entry = {"question": cand, "verdict": v.label, "match": v.match_text, "reason": v.reason}
            if v.is_duplicate:
                report.rejected.append(entry)
            elif v.label == "needs_review":
                report.review.append(entry)  # human queue; not added, not fed back as a reject
            else:
                bank.add(cand, tags, origin="generated")  # later candidates are checked against it too
                report.accepted.append(cand)
    return report
