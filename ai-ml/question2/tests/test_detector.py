import pytest

from qgen import evaluate as E
from qgen.detector import DuplicateDetector
from qgen.store import QuestionStore


@pytest.fixture(scope="module")
def corpus_detector():
    store = QuestionStore()
    for q in E.load_corpus():
        if q["id"] not in ("Q8", "Q10"):
            store.add(q["text"], q["tags"], qid=q["id"])
    return DuplicateDetector(store, use_llm=False).build()


def test_assignment_corpus_sequential_offline():
    rows = E.eval_corpus(use_llm=False)
    assert all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
    labels = {r["id"]: (r["label"], r["matched"]) for r in rows}
    assert labels["Q10"] == ("exact_duplicate", "Q2")
    assert labels["Q2"] == ("semantic_duplicate", "Q1")
    assert labels["Q8"] == ("needs_review", "Q1")


@pytest.mark.parametrize("text, label", [
    ("  what is REFLECTION of light ?? ", "exact_duplicate"),
    ("The angle of incidence is 30 degrees. What is the angle of reflection?", "exact_duplicate"),
    ("What is meant by the reflection of light?", "semantic_duplicate"),
    ("If the angle of incidence equals 30°, find the angle of reflection.", "semantic_duplicate"),
    ("The angle of incidence is 55°. What is the angle of reflection?", "unique"),       # new givens
    ("Why do we see our reflection in a still lake but not in a rough sea?", "unique"),
    ("What is reflection?", "needs_review"),                                          # broader scope
    ("State the laws of reflection of light. Give one everyday use of a plane mirror.", "needs_review"),  # compound
])
def test_cases_against_corpus(corpus_detector, text, label):
    assert corpus_detector.check(text).label == label


def test_detection_ignores_tags():
    store = QuestionStore()
    store.add("What is reflection of light?", {"subject": "physics", "chapter": "light"}, qid="Q1")
    det = DuplicateDetector(store, use_llm=False).build()
    # the detector API takes text only; differently-tagged copies are still duplicates ...
    store.add("Define reflection of light.", {"subject": "general-science", "board": "ICSE"}, qid="X1")
    assert det.check("Define reflection of light.").label == "semantic_duplicate"
    # ... and identically-tagged different questions are not
    assert det.check("What is a periscope used for?").label == "unique"


def test_calibration_pairs_offline_is_safe():
    m = E.eval_pairs(use_llm=False)
    assert m["accuracy"] >= 0.85
    assert m["dup_precision"] == 1.0      # never merges two different questions
    assert m["missed_as_unique"] == 0     # never silently passes a duplicate
