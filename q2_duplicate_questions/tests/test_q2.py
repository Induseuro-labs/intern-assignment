"""Tests for question 2. Run against every available embedder."""
import pytest

from q2_duplicate_questions.core import QuestionBank, SentenceTransformerEmbedder, TfidfEmbedder, normalize, signature
from q2_duplicate_questions.data import CORPUS, HELD_OUT, TAGS
from q2_duplicate_questions.detector import DuplicateDetector
from q2_duplicate_questions.generator import ScriptedLLM, build_context, generate


def _embedders():
    out = [pytest.param(TfidfEmbedder, id="tfidf")]
    try:
        import sentence_transformers  # noqa: F401
        out.append(pytest.param(SentenceTransformerEmbedder, id="minilm"))
    except ImportError:
        pass
    return out


@pytest.fixture(scope="module", params=_embedders())
def embedder_cls(request):
    return request.param


def corpus_bank(cls):
    bank = QuestionBank(cls())
    for qid, text, _ in CORPUS:
        bank.add(text, TAGS, id=qid)
    return bank


def test_normalize_and_signature():
    assert normalize("Define  the REFLECTION of light!!") == normalize("define the reflection of light")
    assert signature("What is reflection of light?") == signature("Explain the meaning of light reflection.")
    assert signature("State the laws of reflection of light.").intent == "laws"
    assert signature("The angle of incidence is 30°. What is the angle of reflection?").intent == "numerical"


def test_section6_corpus_in_order(embedder_cls):
    """Each Q checked against the ones before it reproduces the expected classification."""
    bank = QuestionBank(embedder_cls())
    det = DuplicateDetector(bank)
    for qid, text, expected in CORPUS:
        v = det.check(text)
        assert v.label == expected, (qid, v.label, v.reason)
        bank.add(text, id=qid)
    assert det.check("Define the reflection of light.").match_id == "Q2"


def test_held_out_never_auto_rejects_unique(embedder_cls):
    det = DuplicateDetector(corpus_bank(embedder_cls))
    for text, expected in HELD_OUT:
        v = det.check(text)
        if expected == "unique":
            assert not v.is_duplicate, (text, v.label, v.reason)
        else:
            assert v.label != "unique", (text, v.reason)  # every real duplicate is at least flagged


def test_held_out_exact_with_minilm():
    pytest.importorskip("sentence_transformers")
    det = DuplicateDetector(corpus_bank(SentenceTransformerEmbedder))
    misses = [(t, e, v.label) for t, e in HELD_OUT if (v := det.check(t)).label != e]
    assert misses == []


def test_tags_do_not_affect_duplicates():
    bank = QuestionBank(TfidfEmbedder())
    bank.add("What is reflection of light?", {"board": "CBSE", "grade": "10"})
    v = DuplicateDetector(bank).check("what is reflection of light")
    assert v.label == "exact_duplicate"  # different/absent tags, still a duplicate


def test_judge_resolves_review_band():
    bank = corpus_bank(TfidfEmbedder)
    calls = []

    def judge(cand, existing):
        calls.append((cand, existing))
        return "different", "refraction is not reflection"

    v = DuplicateDetector(bank, judge=judge).check("What is refraction of light?")  # review band for TF-IDF
    assert calls and v.label == "unique" and "LLM judge" in v.reason


def test_context_is_bounded_and_scoped():
    bank = corpus_bank(TfidfEmbedder)
    bank.add("State Snell's law.", {"topic": "refraction"})
    for i in range(200):
        bank.add(f"Filler question number {i} about optics?", {"topic": "reflection"})
    ctx = build_context(bank, "reflection of light", {"topic": "reflection"}, k_near=10, k_clusters=5)
    assert len(ctx.nearest) <= 10 and len(ctx.exemplars) <= 5
    assert "State Snell's law." not in ctx.nearest + ctx.exemplars  # other tag scope


def test_generation_loop_rejects_and_feeds_back():
    bank = corpus_bank(TfidfEmbedder)
    llm = ScriptedLLM(["Define reflection of light.", "What is a plane mirror?",
                       "State the laws of reflection of light!", "What is meant by a plane mirror?",
                       "What is a virtual image?", "Why is a periscope useful?"])
    report = generate(bank, "reflection of light", 3, llm, {"topic": "reflection"}, oversample=1.0)
    assert report.accepted == ["What is a plane mirror?", "What is a virtual image?", "Why is a periscope useful?"]
    assert {r["question"] for r in report.rejected} >= {"Define reflection of light.",
                                                        "State the laws of reflection of light!"}
    assert "Define reflection of light." in report.prompts[1]  # rejects fed back to the next round
    assert len(bank) == len(CORPUS) + 3
