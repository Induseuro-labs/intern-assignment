"""Accuracy and performance evaluation for the duplicate detector."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .detector import DuplicateDetector
from .embeddings import VectorIndex
from .store import QuestionStore

DATA = Path(__file__).resolve().parents[1] / "data"
LABELS = ["exact_duplicate", "semantic_duplicate", "needs_review", "unique"]


def load_corpus() -> list[dict]:
    return json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))["questions"]


def load_calibration() -> list[list[str]]:
    return json.loads((DATA / "calibration_pairs.json").read_text(encoding="utf-8"))["pairs"]


EXPECTED_MAP = {"original": "unique", "different": "unique", "ambiguous": "needs_review",
                "semantic_duplicate": "semantic_duplicate", "exact_duplicate": "exact_duplicate"}


def eval_corpus(use_llm: bool) -> list[dict]:
    """Insert Q1..Q10 in order; classify each against the questions before it."""
    store = QuestionStore()
    det = DuplicateDetector(store, use_llm=use_llm).build([])
    rows = []
    for q in load_corpus():
        t = time.perf_counter()
        v = det.check(q["text"])
        ms = (time.perf_counter() - t) * 1000
        expected = EXPECTED_MAP[q["expected"]]
        match_ok = q.get("of") is None or v.matched_id is not None
        rows.append({"id": q["id"], "text": q["text"], "expected": q["expected"], "label": v.label,
                     "matched": v.matched_id, "reason": v.reason, "ok": v.label == expected and match_ok,
                     "ms": round(ms, 1)})
        added = store.add(q["text"], q["tags"], qid=q["id"])
        det.add(added.id, added.text)
    return rows


def eval_pairs(use_llm: bool, pairs=None) -> dict:
    pairs = pairs or load_calibration()
    det = DuplicateDetector(use_llm=use_llm)
    conf = {a: {b: 0 for b in LABELS} for a in LABELS}
    errors = []
    t = time.perf_counter()
    for a, b, gold in pairs:
        pred = det.compare(b, a).label
        conf[gold][pred] += 1
        if pred != gold:
            errors.append({"a": a, "b": b, "gold": gold, "pred": pred})
    ms = (time.perf_counter() - t) * 1000 / len(pairs)
    dup = {"exact_duplicate", "semantic_duplicate"}
    tp = sum(conf[g][p] for g in dup for p in dup)
    fp = sum(conf[g][p] for g in LABELS if g not in dup for p in dup)
    fn = sum(conf[g][p] for g in dup for p in LABELS if p not in dup)
    wrong_unique = sum(conf[g]["unique"] for g in LABELS if g != "unique")
    correct = sum(conf[l][l] for l in LABELS)
    return {"n": len(pairs), "accuracy": correct / len(pairs), "confusion": conf,
            "dup_precision": tp / (tp + fp) if tp + fp else 1.0, "dup_recall": tp / (tp + fn) if tp + fn else 1.0,
            "missed_as_unique": wrong_unique, "ms_per_pair": round(ms, 1), "errors": errors}


def scale_benchmark(sizes=(1_000, 10_000, 100_000), dim: int = 384, queries: int = 200) -> list[dict]:
    """Exact cosine top-10 search cost as the collection grows (random unit vectors)."""
    rng = np.random.default_rng(0)
    out = []
    for n in sizes:
        vecs = rng.normal(size=(n, dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        idx = VectorIndex([str(i) for i in range(n)], vecs)
        qs = vecs[rng.integers(0, n, queries)]
        t = time.perf_counter()
        for q in qs:
            idx.search(q, 10)
        out.append({"n": n, "ms_per_query": round((time.perf_counter() - t) * 1000 / queries, 3),
                    "index_mb": round(vecs.nbytes / 2 ** 20, 1)})
    return out
