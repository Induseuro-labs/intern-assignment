"""Independent duplicate detector (the safeguard).

It shares no state with the generator: it only reads the bank and a
candidate string. Decision cascade for each candidate, against its top-k
nearest neighbours:

  1. exact          normalised text hash matches            -> exact_duplicate
  2. signature      same intent AND same concept terms      -> semantic_duplicate
  3. guards         different intents (definition vs laws vs numerical ...)
                    or no shared concept term (reflection vs refraction)
                                                            -> not a duplicate of that neighbour
  4. embedding      cosine >= dup threshold                 -> semantic_duplicate
                    cosine >= review threshold              -> needs_review
  5. judge (opt.)   an LLM resolves needs_review into duplicate / unique / still-ambiguous

The verdict for the candidate is the most severe verdict over its neighbours.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .core import QuestionBank, exact_key, signature

SEVERITY = {"exact_duplicate": 3, "semantic_duplicate": 2, "needs_review": 1, "unique": 0}


@dataclass
class Verdict:
    label: str                      # exact_duplicate | semantic_duplicate | needs_review | unique
    match_id: str | None = None
    match_text: str | None = None
    score: float = 0.0
    reason: str = ""
    neighbours: list[dict] = field(default_factory=list)

    @property
    def is_duplicate(self) -> bool:
        return self.label in ("exact_duplicate", "semantic_duplicate")


_JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["verdict", "reason"],
    "properties": {"verdict": {"type": "string", "enum": ["duplicate", "different", "ambiguous"]},
                   "reason": {"type": "string"}},
}


def llm_judge(candidate: str, existing: str, ask_json=None) -> tuple[str, str]:
    if ask_json is None:
        from common.llm import ask_json
    out = ask_json(
        "Two exam questions. Are they duplicates, i.e. would a student's answer to one fully answer the "
        "other (same concept, same depth), regardless of wording?\n"
        f"A: {existing}\nB: {candidate}\n"
        "Answer 'ambiguous' if it depends on context (e.g. one is a more general version of the other).",
        _JUDGE_SCHEMA, max_tokens=2000)
    return out["verdict"], out["reason"]


class DuplicateDetector:
    def __init__(self, bank: QuestionBank, k: int = 8, judge=None):
        self.bank, self.k, self.judge = bank, k, judge

    def _pair(self, cand_sig, other_text: str, cos: float) -> tuple[str, str]:
        other_sig = signature(other_text)
        if cand_sig == other_sig and cand_sig.concepts:
            return "semantic_duplicate", f"same intent '{cand_sig.intent}' and concepts {sorted(cand_sig.concepts)}"
        known = {cand_sig.intent, other_sig.intent} - {"other"}
        if cand_sig.intent != other_sig.intent and len(known) == 2:
            return "unique", f"different intent ({cand_sig.intent} vs {other_sig.intent})"
        if cand_sig.concepts and other_sig.concepts and not cand_sig.concepts & other_sig.concepts:
            return "unique", "no shared concept term"
        emb = self.bank.embedder
        if cos >= emb.dup_threshold:
            return "semantic_duplicate", f"embedding cosine {cos:.2f} >= {emb.dup_threshold}"
        if cos >= emb.review_threshold:
            return "needs_review", f"embedding cosine {cos:.2f} in review band"
        return "unique", f"embedding cosine {cos:.2f}"

    def check(self, text: str) -> Verdict:
        bank = self.bank
        hits = bank.by_exact.get(exact_key(text))
        if hits:
            q = bank.questions[hits[0]]
            return Verdict("exact_duplicate", q.id, q.text, 1.0, "identical after normalisation")

        cand_sig = signature(text)
        best = Verdict("unique", reason="no similar question")
        for idx, cos in bank.search(text, self.k):
            q = bank.questions[idx]
            label, reason = self._pair(cand_sig, q.text, cos)
            best.neighbours.append({"id": q.id, "text": q.text, "cosine": round(cos, 3), "label": label})
            if (SEVERITY[label], cos) > (SEVERITY[best.label], best.score):
                best = Verdict(label, q.id, q.text, cos, reason, best.neighbours)

        if best.label == "needs_review" and self.judge:
            verdict, why = self.judge(text, best.match_text)
            if verdict == "duplicate":
                best.label, best.reason = "semantic_duplicate", f"LLM judge: {why}"
            elif verdict == "different":
                best.label, best.reason = "unique", f"LLM judge: {why}"
            else:
                best.reason += f"; LLM judge: {why}"
        return best
