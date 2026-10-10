"""Text normalisation and light-weight linguistic features for questions.

These are deliberately simple, transparent signals used by the duplicate
detector alongside the neural models:

* ``normalize``   — formatting-insensitive form (case, punctuation, unicode,
                    whitespace, degree signs) → exact-duplicate key.
* ``content_terms`` — stemmed content words without question cue words, so
                    "Define the reflection of light" → {reflect, light}.
* ``numbers``     — numeric givens; questions with different givens are
                    different problems even when worded alike.
* ``intent``      — what kind of answer is being asked for (definition, law,
                    difference, calculation, ...).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

ARTICLES = {"the", "a", "an"}
STOPWORDS = ARTICLES | {
    "of", "is", "are", "was", "were", "be", "to", "in", "on", "at", "for", "by", "with", "and", "or", "its",
    "it", "this", "that", "these", "those", "you", "your", "we", "do", "does", "did", "what", "which", "how",
    "why", "when", "where", "who", "whom", "can", "could", "would", "should", "will", "shall", "please",
    "from", "as", "into", "about", "between", "there", "their", "has", "have", "if", "then", "so", "than",
    "write", "give", "brief", "briefly", "short", "note", "term", "terms", "given", "but", "not", "no", "our",
    "my", "i", "me", "us", "they", "them", "he", "she", "his", "her", "also", "very", "any", "some", "all",
}
# words that only say *how* to ask, not *what* is asked
CUE_WORDS = {"define", "definition", "explain", "meaning", "mean", "meant", "understand", "describe",
             "state", "tell", "concept", "phenomenon", "idea", "notion", "known", "called", "refer", "refers",
             "name", "list", "mention", "say", "says", "find", "calculate", "compute", "determine", "equal",
             "equals", "value", "obtain"}

UNIT_WORDS = {"°": " degrees ", "º": " degrees ", "˚": " degrees ", "%": " percent "}

INTENTS = [
    ("difference", r"\b(difference|differ|differentiate|distinguish|compare|comparison|contrast|versus|vs)\b"),
    ("law", r"\b(laws?|principles?|rules?)\b"),
    ("calculation", r"\b(calculate|compute|find the value|how (much|many|far|long))\b"),
    ("reason", r"^\s*why\b|\breason\b|\bcauses?\b"),
    ("process", r"^\s*how\b|\bsteps?\b|\bprocess\b|\bprocedure\b"),
    ("example", r"\bexamples?\b"),
    ("application", r"\b(uses?|applications?|applied)\b"),
    ("property", r"\b(properties|characteristics|features)\b"),
    ("definition", r"^\s*(what (is|are|do you (understand|mean)|is meant)|define|explain the (meaning|term|concept)|"
                   r"what does .* mean|meaning of)"),
]


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    for k, v in UNIT_WORDS.items():
        t = t.replace(k, v)
    t = t.lower()
    t = re.sub(r"[‘’´`]", "'", t)
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)        # 1,000 → 1000
    t = re.sub(r"[^\w\s.']|(?<!\d)\.|\.(?!\d)", " ", t)  # punctuation, keep decimal points
    t = t.replace("'", "")
    return re.sub(r"\s+", " ", t).strip()


def exact_key(text: str) -> str:
    """Formatting-insensitive identity: normalised text without articles."""
    words = [w for w in normalize(text).split() if w not in ARTICLES]
    return hashlib.sha1(" ".join(words).encode()).hexdigest()


def stem(word: str) -> str:
    for suf, rep in (("ations", "ate"), ("ation", "ate"), ("ions", ""), ("ion", ""), ("ings", ""), ("ing", ""),
                     ("ied", "y"), ("ies", "y"), ("ed", ""), ("es", ""), ("ly", ""), ("s", "")):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)] + rep
    return word


def content_terms(text: str) -> frozenset[str]:
    words = normalize(text).split()
    return frozenset(stem(w) for w in words
                     if w not in STOPWORDS and w not in CUE_WORDS and not re.fullmatch(r"[\d.]+", w)
                     and stem(w) not in CUE_WORDS)


def numbers(text: str) -> tuple[str, ...]:
    return tuple(sorted(re.findall(r"\d+(?:\.\d+)?", normalize(text))))


QUESTION_START = re.compile(r"^(what|why|how|which|when|where|who|state|define|list|name|explain|describe|"
                            r"draw|write|give|distinguish|differentiate|compare|calculate|find|derive|show|prove)\b")


def sub_questions(text: str) -> list[str]:
    """Self-contained parts of a compound question.

    A part counts only if it reads as a question on its own and no earlier
    sentence supplied numeric givens it might depend on ("The angle of incidence
    is 30°. What is the angle of reflection?" yields nothing).
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+(?=[A-Z(])", text.strip()) if s.strip()]
    if len(sentences) < 2:
        return []
    parts, givens = [], False
    for s in sentences:
        n = normalize(s)
        if not givens and len(n.split()) >= 4 and (s.endswith("?") or QUESTION_START.match(n)):
            parts.append(s)
        if numbers(s):
            givens = True
    return parts


def intent(text: str) -> str:
    t = normalize(text)
    if numbers(text):
        return "calculation"
    for name, pattern in INTENTS:
        if re.search(pattern, t):
            return name
    return "other"
