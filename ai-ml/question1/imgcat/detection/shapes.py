"""Geometric shape fitting that tolerates occlusion.

Each candidate model (circle, rectangle, triangle) is fitted only to the
*true edge* pixels of a region — boundary pixels that touch the background —
and not to the edges it shares with an overlapping shape. A circle half
hidden behind a triangle still has a circular visible edge, so it is
recognised, and the fitted model gives its full (amodal) bounding box.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ShapeFit:
    name: str            # circle | square | rectangle | triangle
    score: float         # 0..1 confidence from fit quality
    bbox: dict           # amodal box of the fitted model
    coverage: float      # visible area / model area (1.0 = not occluded)
    residual: float
    model: np.ndarray | None = field(default=None, repr=False)  # full model mask


def _bbox_from_points(pts: np.ndarray) -> dict:
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    return {"x": int(x0), "y": int(y0), "width": int(x1 - x0 + 1), "height": int(y1 - y0 + 1)}


def _edge_points(mask: np.ndarray, occluders: np.ndarray | None) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), np.float32)
    pts = max(contours, key=cv2.contourArea).reshape(-1, 2)
    h, w = mask.shape
    # drop image-border points (could be cropped) and points shared with occluders
    keep = (pts[:, 0] > 0) & (pts[:, 1] > 0) & (pts[:, 0] < w - 1) & (pts[:, 1] < h - 1)
    if occluders is not None and occluders.any():
        near = cv2.dilate(occluders.astype(np.uint8), np.ones((7, 7), np.uint8))
        keep &= near[pts[:, 1], pts[:, 0]] == 0
    return pts[keep].astype(np.float32)


def _poly_residual(pts: np.ndarray, poly: np.ndarray) -> float:
    poly = poly.reshape(-1, 1, 2).astype(np.float32)
    d = [abs(cv2.pointPolygonTest(poly, (float(x), float(y)), True)) for x, y in pts[:: max(1, len(pts) // 400)]]
    return float(np.mean(d))


def _model_mask(shape, kind: str, params) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    if kind == "circle":
        cx, cy, r = params
        cv2.circle(m, (int(round(cx)), int(round(cy))), int(round(r)), 1, -1)
    else:
        cv2.fillPoly(m, [np.round(params).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def fit_shape(mask: np.ndarray, occluders: np.ndarray | None = None) -> ShapeFit | None:
    """Best geometric model for a binary region, or None if nothing fits well."""
    mask = mask.astype(bool)
    area = int(mask.sum())
    if area < 50:
        return None
    pts = _edge_points(mask, occluders)
    if len(pts) < 20:
        return None
    size = np.sqrt(area)
    candidates = []

    # circle — algebraic (Kasa) least squares on true-edge points
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    r2 = c + cx ** 2 + cy ** 2
    if r2 > 0:
        r = np.sqrt(r2)
        res = float(np.mean(np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)))
        candidates.append(("circle", (cx, cy, r), res))

    # rectangle — minimum-area rotated rectangle of true-edge points
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    candidates.append(("rect", box, _poly_residual(pts, box)))

    # triangle — minimum enclosing triangle of true-edge points
    ok, tri = cv2.minEnclosingTriangle(pts.reshape(-1, 1, 2))
    if ok and tri is not None:
        tri = tri.reshape(-1, 2)
        candidates.append(("triangle", tri, _poly_residual(pts, tri)))

    best = None
    for kind, params, res in candidates:
        model = _model_mask(mask.shape, kind, params)
        model_area = model.sum()
        if model_area == 0:
            continue
        inside = (mask & model).sum() / area           # visible pixels explained by the model
        coverage = (mask & model).sum() / model_area   # how much of the model is visible
        hidden_ok = True
        if coverage < 0.97:
            # missing model area must be explained by an occluder, not by background
            missing = model & ~mask
            explained = (missing & occluders).sum() / max(1, missing.sum()) if occluders is not None else 0
            hidden_ok = explained > 0.8
        rel = res / size
        score = float(np.clip(1 - rel * 25, 0, 1) * min(1.0, inside / 0.98) * (1 if hidden_ok else 0.3))
        if coverage < 0.45:
            score *= 0.3
        if best is None or score > best[1]:
            best = (kind, score, params, coverage, rel, model)
    if best is None or best[1] < 0.6:
        return None
    kind, score, params, coverage, rel, model = best
    if kind == "circle":
        cx, cy, r = params
        bbox = _bbox_from_points(np.array([[cx - r, cy - r], [cx + r, cy + r]]))
    else:
        bbox = _bbox_from_points(np.asarray(params))
    name = kind
    if kind == "rect":
        (w, h) = rect[1]
        name = "square" if min(w, h) / max(w, h) > 0.92 else "rectangle"
    return ShapeFit(name, round(score, 3), bbox, round(float(min(coverage, 1.0)), 3), round(rel, 4), model)
