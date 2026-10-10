"""Embedding models and a vector index.

Two different bi-encoders are used on purpose:

* ``GEN_MODEL`` (all-MiniLM-L6-v2) — used by the *generator* for context
  selection and its in-loop novelty filter.
* ``DET_MODEL`` (bge-small-en-v1.5) — used by the independent *detector* for
  candidate retrieval, so a blind spot of one model is not shared by both
  layers.

``VectorIndex`` is exact cosine search over a normalised matrix (fast to
~10^5–10^6 vectors on CPU); ``hnswlib``/FAISS can be dropped in behind the
same ``search`` method for larger collections.
"""
from __future__ import annotations

import os
import threading

import numpy as np

GEN_MODEL = os.environ.get("QGEN_GEN_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DET_MODEL = os.environ.get("QGEN_DET_MODEL", "BAAI/bge-small-en-v1.5")
CROSS_MODEL = os.environ.get("QGEN_CROSS_MODEL", "cross-encoder/quora-distilroberta-base")

_lock = threading.Lock()
_models: dict[str, object] = {}


def _st(name: str):
    with _lock:
        if name not in _models:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            from sentence_transformers import SentenceTransformer

            _models[name] = SentenceTransformer(name, device="cpu")
        return _models[name]


def cross_encoder():
    with _lock:
        if CROSS_MODEL not in _models:
            from sentence_transformers import CrossEncoder

            _models[CROSS_MODEL] = CrossEncoder(CROSS_MODEL, device="cpu")
        return _models[CROSS_MODEL]


class Embedder:
    def __init__(self, model: str, store=None):
        self.model = model
        self.store = store

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), np.float32)
        vecs = _st(self.model).encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, np.float32)

    def encode_questions(self, questions) -> np.ndarray:
        """Encode stored questions, reusing vectors cached in the store."""
        if self.store is None:
            return self.encode([q.text for q in questions])
        cached = self.store.get_vectors(self.model)
        missing = [q for q in questions if q.id not in cached]
        if missing:
            vecs = self.encode([q.text for q in missing])
            new = {q.id: v.tobytes() for q, v in zip(missing, vecs)}
            self.store.put_vectors(self.model, new)
            cached.update(new)
        return np.stack([np.frombuffer(cached[q.id], np.float32) for q in questions]) if questions else \
            np.zeros((0, 384), np.float32)


class VectorIndex:
    def __init__(self, ids: list[str], vectors: np.ndarray):
        self.ids = list(ids)
        self.vectors = vectors.astype(np.float32)

    def add(self, qid: str, vec: np.ndarray) -> None:
        self.ids.append(qid)
        self.vectors = np.vstack([self.vectors, vec[None].astype(np.float32)]) if len(self.vectors) else vec[None]

    def search(self, vec: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        sims = self.vectors @ vec
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.ids[i], float(sims[i])) for i in top]
