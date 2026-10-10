"""Independent duplicate detector.

Runs after generation, on every candidate, against the WHOLE collection
(never filtered by tags), and shares no code path or model with the
generator's prevention layer. Its job is to catch what prevention missed.

Per candidate:
 1. exact layer   — formatting-insensitive key lookup (O(1)) + near-exact
                    character similarity for typos/spacing.
 2. retrieval     — top-k neighbours by bge-small embeddings (a different
                    model from the generator's MiniLM).
 3. pair verdict  — a transparent cascade over signals, each recorded as
                    evidence: numeric givens, answer intent, content-term
                    overlap, bi-encoder cosine, cross-encoder (Quora
                    duplicate-question model) probability.
 4. grey zone     — pairs the cascade cannot settle go to an LLM judge when
                    one is configured, else to human review.

Verdicts: exact_duplicate > semantic_duplicate > needs_review > unique.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from rapidfuzz import fuzz

from . import text as T
from .embeddings import DET_MODEL, Embedder, VectorIndex, cross_encoder
from .llm import chat_json, llm_available

SEVERITY = {"exact_duplicate": 3, "semantic_duplicate": 2, "needs_review": 1, "unique": 0}
# intents that ask for clearly different answers; others (other/process/reason/property) are compatible
SPECIFIC_INTENTS = {"definition", "law", "difference", "calculation", "example", "application"}


@dataclass
class Thresholds:
    """Calibrated on data/calibration_pairs.json (refraction, photosynthesis, motion, ...),
    NOT on the reflection test corpus."""
    near_exact_ratio: float = 97.0     # rapidfuzz ratio on article-free normalised text
    same_terms_cos: float = 0.80       # same concepts + compatible intent → paraphrase
    strong_ce: float = 0.90            # cross-encoder says duplicate ...
    strong_ce_cos: float = 0.90        # ... and embeddings agree
    scope_cos: float = 0.75            # broader/narrower question (subset of terms)
    scope_max_extra: int = 2
    grey_cos: float = 0.90
    grey_ce: float = 0.50
    k: int = 8


@dataclass
class PairVerdict:
    other_id: str
    other_text: str
    label: str
    reason: str
    signals: dict = field(default_factory=dict)


@dataclass
class Verdict:
    text: str
    label: str
    matched_id: str | None
    matched_text: str | None
    reason: str
    pairs: list[PairVerdict]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pairs"] = d["pairs"][:3]
        return d


class DuplicateDetector:
    def __init__(self, store=None, use_llm: bool | None = None, thresholds: Thresholds | None = None,
                 judge_model: str = "anthropic/claude-haiku-5.5"):
        self.store = store
        self.th = thresholds or Thresholds()
        self.embedder = Embedder(DET_MODEL, store)
        self.use_llm = llm_available() if use_llm is None else use_llm
        self.judge_model = judge_model
        self._judge_cache: dict[tuple[str, str], dict] = {}
        self.index: VectorIndex | None = None
        self.texts: dict[str, str] = {}

    # -------------------------------------------------------------- index
    def build(self, questions=None) -> "DuplicateDetector":
        questions = questions if questions is not None else self.store.all()
        self.texts = {q.id: q.text for q in questions}
        self.index = VectorIndex([q.id for q in questions], self.embedder.encode_questions(questions))
        self._keys: dict[str, list[str]] = {}
        for q in questions:
            self._keys.setdefault(T.exact_key(q.text), []).append(q.id)
        return self

    def add(self, qid: str, text: str, vec: np.ndarray | None = None) -> None:
        self.texts[qid] = text
        self._keys.setdefault(T.exact_key(text), []).append(qid)
        self.index.add(qid, vec if vec is not None else self.embedder.encode([text])[0])

    # ------------------------------------------------------------ verdict
    def check(self, text: str) -> Verdict:
        if self.index is None:
            self.build()
        key = T.exact_key(text)
        if self._keys.get(key):
            qid = self._keys[key][0]
            pv = PairVerdict(qid, self.texts[qid], "exact_duplicate", "identical after normalising case, "
                             "punctuation, whitespace and articles", {"exact_key": True})
            return Verdict(text, pv.label, qid, pv.other_text, pv.reason, [pv])
        vec = self.embedder.encode([text])[0]
        hits = self.index.search(vec, self.th.k)
        if not hits:
            return Verdict(text, "unique", None, None, "empty collection", [])
        ce_scores = cross_encoder().predict([(text, self.texts[qid]) for qid, _ in hits], show_progress_bar=False)
        pairs = [self.compare(text, self.texts[qid], qid, cos, float(ce)) for (qid, cos), ce in zip(hits, ce_scores)]
        pairs.sort(key=lambda p: (-SEVERITY[p.label], -p.signals.get("cos", 0)))
        best = pairs[0]
        if best.label == "unique":
            part = self._compound(text)
            if part is not None:
                return Verdict(text, "needs_review", part.other_id, part.other_text, part.reason, [part] + pairs)
            return Verdict(text, "unique", None, None, f"no close match (nearest: {best.other_id}, "
                           f"cos {best.signals['cos']:.2f}: {best.reason})", pairs)
        return Verdict(text, best.label, best.other_id, best.other_text, best.reason, pairs)

    def _compound(self, text: str) -> PairVerdict | None:
        """A multi-part question that contains an existing question as one of its parts."""
        for part in T.sub_questions(text):
            sub = self.check(part)
            if sub.label in ("exact_duplicate", "semantic_duplicate"):
                return PairVerdict(sub.matched_id, sub.matched_text, "needs_review",
                                   f"compound question: its part “{part}” is a {sub.label.replace('_', ' ')} "
                                   f"of {sub.matched_id} ({sub.reason}); the rest is new", {"part": part})
        return None

    def compare(self, a: str, b: str, b_id: str = "", cos: float | None = None, ce: float | None = None) -> PairVerdict:
        """Classify one pair; every signal used is stored in ``signals``."""
        th = self.th
        if cos is None:
            va, vb = self.embedder.encode([a, b])
            cos = float(va @ vb)
        if ce is None:
            ce = float(cross_encoder().predict([(a, b)], show_progress_bar=False)[0])
        ta, tb = T.content_terms(a), T.content_terms(b)
        ia, ib = T.intent(a), T.intent(b)
        na, nb = T.numbers(a), T.numbers(b)
        ratio = fuzz.ratio(" ".join(w for w in T.normalize(a).split() if w not in T.ARTICLES),
                           " ".join(w for w in T.normalize(b).split() if w not in T.ARTICLES))
        sig = {"cos": round(cos, 3), "cross_encoder": round(ce, 3), "terms_a": sorted(ta), "terms_b": sorted(tb),
               "intent_a": ia, "intent_b": ib, "numbers_a": na, "numbers_b": nb, "char_ratio": round(ratio, 1)}

        def v(label, reason):
            return PairVerdict(b_id, b, label, reason, sig)

        if T.exact_key(a) == T.exact_key(b) or (ratio >= th.near_exact_ratio and na == nb):
            return v("exact_duplicate", f"same text up to formatting (char similarity {ratio:.0f})")
        if na != nb:
            if na and nb:
                return v("unique", f"different numeric givens {na} vs {nb}")
            return v("unique", "numerical problem vs conceptual question")
        if ia != ib and ia in SPECIFIC_INTENTS and ib in SPECIFIC_INTENTS and not (ce >= 0.97 and cos >= 0.95):
            return v("unique", f"asks for a different kind of answer ({ia} vs {ib})")
        if ta == tb and ta and cos >= th.same_terms_cos:
            return v("semantic_duplicate", f"same concepts {sorted(ta)} and same kind of answer ({ia}); "
                                           f"wording differs (cos {cos:.2f})")
        if ce >= th.strong_ce and cos >= th.strong_ce_cos:
            return v("semantic_duplicate", f"paraphrase model {ce:.2f} and embeddings {cos:.2f} agree")
        small, big = (ta, tb) if len(ta) < len(tb) else (tb, ta)
        if small and small < big and len(big - small) <= th.scope_max_extra and cos >= th.scope_cos:
            extra = sorted(big - small)
            reason = (f"one question is broader: {sorted(small)} vs {sorted(big)} (extra {extra}); "
                      "equivalent only if the context implies the qualifier")
            return self._judge(a, b, v, reason, scope=True)
        if cos >= th.grey_cos or ce >= th.grey_ce:
            return self._judge(a, b, v, f"close but unconfirmed (cos {cos:.2f}, paraphrase {ce:.2f})")
        return v("unique", f"different concepts {sorted(ta ^ tb)}")

    # ------------------------------------------------------------- judge
    JUDGE_SYSTEM = (
        "You compare two exam questions from a question bank and decide whether the second one would be a "
        "duplicate of the first. Answer with JSON {\"verdict\": \"duplicate\"|\"different\"|\"ambiguous\", "
        "\"reason\": str}. duplicate = a student would give the same answer to both (same concept, same scope, "
        "same kind of answer), even if worded differently. different = they test different knowledge or ask "
        "for different answers (e.g. a definition vs a law, a quantity vs its unit, a broader topic vs a "
        "specific sub-part). ambiguous = one question is a broader or less specific version of the other (for "
        "example it omits a qualifier) so whether they are equivalent depends on the surrounding context.")

    def _judge(self, a: str, b: str, v, reason: str, scope: bool = False) -> PairVerdict:
        if not self.use_llm:
            return v("needs_review", reason + " → human review (no LLM judge configured)")
        key = (a, b)
        if key not in self._judge_cache:
            try:
                self._judge_cache[key] = chat_json(self.JUDGE_SYSTEM, f"Question A: {a}\nQuestion B: {b}",
                                                   model=self.judge_model)
            except Exception as exc:
                return v("needs_review", reason + f" → judge unavailable ({type(exc).__name__})")
        j = self._judge_cache[key]
        verdict = str(j.get("verdict", "ambiguous")).lower()
        label = {"duplicate": "semantic_duplicate", "different": "unique"}.get(verdict, "needs_review")
        if scope and label == "semantic_duplicate":
            # a broader/narrower pair is equivalent only in some contexts; the judge sees no context,
            # so it may rule the pair out but not merge it automatically
            label = "needs_review"
        pv = v(label, f"{reason} → LLM judge: {verdict} ({j.get('reason', '')})")
        pv.signals = {**pv.signals, "llm_judge": verdict}
        return pv
