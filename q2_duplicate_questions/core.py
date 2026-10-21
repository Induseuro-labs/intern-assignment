"""Question bank, text normalisation, question "signatures" and embedders.

Every question gets three content-only representations (tags never enter):
  1. normalised text      -> exact-duplicate hash (formatting/punctuation/case-proof)
  2. signature            -> (intent, concept terms): "what is reflection of light?",
                             "Define the reflection of light." and "Explain the meaning
                             of light reflection." all become (definition, {light, reflect})
  3. dense embedding      -> paraphrases the rules don't anticipate
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- text
_NUM_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
              "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def normalize(text: str) -> str:
    """Lowercase, unify unicode/whitespace/punctuation and units. Used for exact matching."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = t.replace("°", " degrees ").replace("’", "'")
    t = re.sub(r"[^\w\s]", " ", t)
    words = [_NUM_WORDS.get(w, w) for w in t.split()]
    return " ".join(words)


def exact_key(text: str) -> str:
    return hashlib.sha1(normalize(text).encode()).hexdigest()


# Ordered: first match wins. "numerical" first so "30 degrees ... what is" is a calculation.
INTENTS = [
    ("numerical", r"\d+(\.\d+)?\s*(degrees|cm|mm|m|s|kg|n|j|w|v|hz|percent)\b|\b(calculate|compute|find the value)\b"),
    ("difference", r"\bdifferences?\b|\bdifferentiate\b|\bdistinguish\b|\bcompare\b|\bcontrast\b|\bversus\b|\bvs\b|\bdiffer\b"),
    ("laws", r"\blaws?\b|\brules?\b|\bprinciples?\b"),
    ("examples", r"\bexamples?\b|\binstances?\b"),
    ("reason", r"^why\b|\bwhy (do|does|is|are)\b|\breason\b"),
    ("process", r"^how\b|\bhow (is|are|does|do|can)\b"),
    ("application", r"\buses?\b|\bapplications?\b"),
    ("definition", r"^what (is|are|s)\b|\b(define|definition|explain|meaning|meant|describe)\b"
                   r"|\bwhat do you (understand|mean)\b|\bwhat does\b.*\bmean\b"),
]

# Words that frame a question rather than carry its concept.
FRAME_WORDS = set("""
a an the of is are was were be by to in on for with and or what whats which who do does did you your
define definition explain meaning meant mean understand term concept state describe give write briefly
short note brief tell me us please can could would kindly about
between difference differences differentiate distinguish compare contrast versus vs
if will be it its this that then there when at from down into find calculate compute determine value
""".split())


def stem(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    for suffix in ("ation", "ion", "ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


@dataclass(frozen=True)
class Signature:
    intent: str
    concepts: frozenset[str]


def signature(text: str) -> Signature:
    norm = normalize(text)
    intent = next((name for name, pat in INTENTS if re.search(pat, norm)), "other")
    words = [w for w in norm.split() if w not in FRAME_WORDS and not w.isdigit() and w != "degrees"]
    return Signature(intent, frozenset(stem(w) for w in words))


# ------------------------------------------------------------------------ embedders
class SentenceTransformerEmbedder:
    """Semantic embeddings (all-MiniLM-L6-v2, 384-d, CPU friendly)."""

    dup_threshold, review_threshold = 0.92, 0.80

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.name = f"sentence-transformers/{model}"
        self.model = SentenceTransformer(model)

    def fit(self, texts: list[str]) -> None:
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


class TfidfEmbedder:
    """Dependency-light lexical fallback (word + char n-gram TF-IDF).

    Catches reworded/reordered text but not true synonyms; the signature layer
    carries more of the load when this backend is active.
    """

    dup_threshold, review_threshold = 0.90, 0.80
    name = "tfidf-char-word"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion
        self.vec = FeatureUnion([
            ("w", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True)),
            ("c", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5),
                                  sublinear_tf=True)),
        ])
        self.fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vec.fit(list(texts))
        self.fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self.fitted:
            self.fit(texts)
        m = self.vec.transform(list(texts)).toarray().astype(np.float32)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def default_embedder():
    try:
        return SentenceTransformerEmbedder()
    except Exception:  # package or model download unavailable
        return TfidfEmbedder()


# ---------------------------------------------------------------------------- bank
@dataclass
class Question:
    text: str
    tags: dict[str, str] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    origin: str = "seed"  # seed | generated | manual


class QuestionBank:
    """In-memory store with an exact-hash index and a dense vector matrix.

    Brute-force cosine over a float32 matrix handles ~10^5 questions in a few
    ms; swap ``search`` for FAISS/HNSW or pgvector beyond that (same interface).
    """

    def __init__(self, embedder=None):
        self.embedder = embedder or default_embedder()
        self.questions: list[Question] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.by_exact: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return len(self.questions)

    def add_many(self, questions: list[Question]) -> None:
        if not questions:
            return
        if isinstance(self.embedder, TfidfEmbedder):  # refit vocabulary, re-embed everything
            self.embedder.fit([q.text for q in self.questions + questions])
            self.questions.extend(questions)
            self.vectors = self.embedder.encode([q.text for q in self.questions])
        else:
            vecs = self.embedder.encode([q.text for q in questions])
            self.vectors = vecs if len(self.questions) == 0 else np.vstack([self.vectors, vecs])
            self.questions.extend(questions)
        for i in range(len(self.questions) - len(questions), len(self.questions)):
            self.by_exact.setdefault(exact_key(self.questions[i].text), []).append(i)

    def add(self, text: str, tags: dict | None = None, origin: str = "seed", id: str | None = None) -> Question:
        q = Question(text, dict(tags or {}), origin=origin, **({"id": id} if id else {}))
        self.add_many([q])
        return q

    def search(self, text: str, k: int = 10, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        if not self.questions:
            return []
        v = self.embedder.encode([text])[0]
        sims = self.vectors @ v
        if mask is not None:
            sims = np.where(mask, sims, -np.inf)
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top if np.isfinite(sims[i])]

    def filter_mask(self, tags: dict | None) -> np.ndarray:
        """Boolean mask of questions whose tags contain every requested key=value."""
        tags = tags or {}
        return np.array([all(q.tags.get(k) == v for k, v in tags.items()) for q in self.questions], bool)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([q.__dict__ for q in self.questions], indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, embedder=None) -> "QuestionBank":
        bank = cls(embedder)
        bank.add_many([Question(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))])
        return bank
