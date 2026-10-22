"""Open-vocabulary detection with OWLv2 (google/owlv2-base-patch16-ensemble).

OWLv2 is prompted with text queries, so the same model finds an apple in a
photograph and a boy in a flat cartoon — no per-style training. The query
list is configurable; labels are the query text without the article.
"""
from __future__ import annotations

import os
import threading

import numpy as np

MODEL_ID = os.environ.get("IMGCAT_OWL_MODEL", "google/owlv2-base-patch16-ensemble")

DEFAULT_QUERIES = {
    "apple": "an apple",
    "orange": "an orange fruit",
    "banana": "a banana",
    "boy": "a boy",
    "girl": "a girl",
    "man": "a man",
    "woman": "a woman",
    "person": "a person",
    "dog": "a dog",
    "cat": "a cat",
    "car": "a car",
    "ball": "a ball",
    "flower": "a flower",
    "tree": "a tree",
    "cup": "a cup",
    "circle": "a circle",
    "square": "a square",
    "triangle": "a triangle",
    "rectangle": "a rectangle",
    "star": "a star shape",
}

_lock = threading.Lock()
_model = None


def _load():
    global _model
    with _lock:
        if _model is None:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            proc = Owlv2Processor.from_pretrained(MODEL_ID)
            model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
            _model = (proc, model, device)
    return _model


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def detect(rgb: np.ndarray, queries: dict[str, str] | None = None, threshold: float = 0.1,
           nms_iou: float = 0.5) -> list[dict]:
    """Return ``[{label, score, box:[x0,y0,x1,y1]}]`` after class-agnostic NMS."""
    import torch
    from PIL import Image

    queries = queries or DEFAULT_QUERIES
    labels, texts = list(queries), list(queries.values())
    proc, model, device = _load()
    image = Image.fromarray(rgb)
    inputs = proc(text=[texts], images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    side = max(image.size)  # OWLv2 pads to a square at the bottom/right
    res = proc.post_process_grounded_object_detection(
        out, threshold=threshold, target_sizes=torch.tensor([[side, side]]))[0]
    h, w = rgb.shape[:2]
    dets = []
    for score, label, box in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
        x0, y0, x1, y1 = box
        box = [max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)]
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            continue
        dets.append({"label": labels[label], "score": round(float(score), 3), "box": box})
    dets.sort(key=lambda d: -d["score"])
    kept: list[dict] = []
    for d in dets:
        if all(_iou(d["box"], k["box"]) < nms_iou for k in kept):
            kept.append(d)
    return kept
