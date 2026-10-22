"""Generate → independently check → store.

The detector is built from the full collection (no tag filter) and every
generated question is checked before insertion:
  exact/semantic duplicate → dropped (logged as a prevention miss)
  needs_review             → stored with status 'needs_review' + duplicate_of
  unique                   → stored as active
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .detector import DuplicateDetector, Verdict
from .generator import GenerationResult, QuestionGenerator


@dataclass
class PipelineReport:
    generation: GenerationResult
    verdicts: list[Verdict]
    stored: list[str] = field(default_factory=list)
    review: list[str] = field(default_factory=list)
    dropped: list[Verdict] = field(default_factory=list)


def generate_and_store(store, topic: str, n: int, tags: dict | None = None, new_tags: dict | None = None,
                       generator: QuestionGenerator | None = None,
                       detector: DuplicateDetector | None = None) -> PipelineReport:
    generator = generator or QuestionGenerator(store)
    result = generator.generate(topic, n, tags)
    detector = detector or DuplicateDetector(store).build()
    report = PipelineReport(result, [])
    for cand in result.accepted:
        v = detector.check(cand.question)
        report.verdicts.append(v)
        if v.label in ("exact_duplicate", "semantic_duplicate"):
            report.dropped.append(v)
            continue
        q = store.add(cand.question, {**(new_tags or {}), "concept": cand.concept or "unspecified"},
                      source="llm", status="needs_review" if v.label == "needs_review" else "active",
                      duplicate_of=v.matched_id, detection=v.to_dict())
        detector.add(q.id, q.text)
        (report.review if v.label == "needs_review" else report.stored).append(q.id)
    return report
