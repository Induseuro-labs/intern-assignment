from qgen import evaluate as E
from qgen.detector import DuplicateDetector
from qgen.generator import QuestionGenerator, cluster
from qgen.pipeline import generate_and_store
from qgen.store import QuestionStore


def corpus_store():
    store = QuestionStore()
    for q in E.load_corpus():
        if q["id"] != "Q10":
            store.add(q["text"], q["tags"], qid=q["id"])
    return store


class ScriptedLLM:
    """Returns pre-written batches and records every prompt it receives."""

    def __init__(self, *batches):
        self.batches = list(batches)
        self.prompts = []

    def __call__(self, system, user):
        self.prompts.append(user)
        return {"questions": [{"question": q, "concept": c} for q, c in self.batches.pop(0)]}


def test_context_clusters_paraphrases_and_respects_budget():
    store = corpus_store()
    ctx, rel, _ = QuestionGenerator(store, llm=ScriptedLLM()).select_context("reflection of light")
    # the definition paraphrases Q1-Q4 collapse into one concept; the laws (Q5) stay separate
    biggest = max(ctx.concepts, key=lambda c: c.size)
    assert {"What is reflection of light?", "Define the reflection of light.",
            "What do you understand by the reflection of light?"} <= set(biggest.members)
    assert "State the laws of reflection of light." not in biggest.members
    assert len(ctx.concepts) < ctx.relevant and ctx.shown == len(ctx.concepts)
    # a tight budget truncates the list and says so
    small, _, _ = QuestionGenerator(store, budget_tokens=45, llm=ScriptedLLM()).select_context("reflection of light")
    assert small.shown < len(small.concepts) and "not listed" in small.prompt_block
    assert small.est_tokens <= 45 + 20


def test_paraphrase_rejected_in_loop_and_fed_back():
    store = corpus_store()
    llm = ScriptedLLM(
        [("Explain what is meant by reflection of light.", "definition"),
         ("Why can we read a book under a lamp from any angle?", "diffuse reflection")],
        [("How does a periscope use two plane mirrors?", "periscope"),
         ("A ray hits a plane mirror at 40 degrees incidence; find the angle between incident and reflected rays.",
          "angle between rays")],
    )
    gen = QuestionGenerator(store, llm=llm)
    res = gen.generate("reflection of light", n=3)
    assert [c.question for c in res.rejected] == ["Explain what is meant by reflection of light."]
    assert len(res.accepted) == 3 and res.rounds == 2
    # round 2 prompt tells the model what was rejected and what is already accepted
    assert "Explain what is meant by reflection of light." in llm.prompts[1]
    assert "Why can we read a book under a lamp" in llm.prompts[1]


def test_detector_catches_what_prevention_let_through():
    store = corpus_store()
    llm = ScriptedLLM([("What is meant by reflection of light?", "definition"),  # paraphrase of Q1
                       ("What is a periscope?", "periscope")])
    gen = QuestionGenerator(store, llm=llm, novelty_threshold=1.01)            # prevention disabled
    det = DuplicateDetector(store, use_llm=False).build()
    report = generate_and_store(store, "reflection of light", 2, generator=gen, detector=det,
                                new_tags={"generated_by": "test"})
    assert [v.label for v in report.verdicts] == ["semantic_duplicate", "unique"]
    assert len(report.dropped) == 1 and len(report.stored) == 1
    stored = store.get(report.stored[0])
    assert stored.source == "llm" and stored.tags["generated_by"] == ["test"]


def test_cluster_threshold():
    import numpy as np

    v = np.eye(3, dtype=np.float32)
    v = np.vstack([v, v[0] * 0.99 + v[1] * 0.01])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    groups = sorted(sorted(g) for g in cluster(v))
    assert groups == [[0, 3], [1], [2]]
