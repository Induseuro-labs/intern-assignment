"""Detection pipeline: classical segmentation fused with OWLv2.

Why both:
* Segmentation gives pixel-exact, shadow-free boxes and masks on plain
  backgrounds, and geometric labels for flat shapes (including occluded
  ones), but cannot name a photographed apple or a cartoon boy.
* OWLv2 names objects in photos and cartoons alike and works on cluttered
  backgrounds, but its boxes are looser and it confuses "square" with
  "rectangle" or calls colour swatches "squares".

Fusion: segmentation regions get semantic names from overlapping OWLv2
boxes; flat shape layers keep their geometric label; OWLv2 boxes that match
no region are kept when confident (cluttered photos), and dropped when they
cover only background.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import owl
from .colors import NEUTRALS, color_histogram, dominant_color, name_rgb
from .segmentation import (Region, bbox_of, connected_regions, decompose, estimate_background,
                           estimate_style, foreground_mask, trim_contact_shadow)

DETECTOR_VERSION = "fusion-1.0 (segmentation + owlv2-base-ensemble)"
SEG_ONLY_VERSION = "fusion-1.0 (segmentation only)"

CATEGORIES = {
    "fruit": {"apple", "orange", "banana", "mandarin", "pear", "lemon", "grape", "peach"},
    "person": {"boy", "girl", "man", "woman", "person"},
    "shape": {"circle", "square", "triangle", "rectangle", "star"},
    "animal": {"dog", "cat"},
}


def category_of(name: str) -> str:
    for cat, names in CATEGORIES.items():
        if name in names:
            return cat
    return "swatch" if name.endswith("swatch") else "object"


@dataclass
class Detection:
    auto: dict                 # fields stored in objects.auto
    mask: np.ndarray | None    # pixel mask (saved alongside the catalog)


@dataclass
class DetectionResult:
    objects: list[Detection]
    style: str
    background_plain: bool
    notes: list[str] = field(default_factory=list)


def _box_xyxy(b: dict) -> tuple[float, float, float, float]:
    return b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]


def _inter(a, b) -> float:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(a) -> float:
    return (a[2] - a[0]) * (a[3] - a[1])


def _iou(a, b) -> float:
    i = _inter(a, b)
    return i / (_area(a) + _area(b) - i) if i else 0.0


def _xyxy_to_bbox(b) -> dict:
    x0, y0, x1, y1 = [int(round(v)) for v in b]
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


class Detector:
    def __init__(self, use_owl: bool | None = None, queries: dict[str, str] | None = None,
                 owl_threshold: float = 0.1, keep_threshold: float = 0.25):
        if use_owl is None:
            use_owl = os.environ.get("IMGCAT_DETECTOR", "fusion") != "seg"
        self.use_owl = use_owl
        self.queries = queries
        self.owl_threshold = owl_threshold
        self.keep_threshold = keep_threshold

    @property
    def version(self) -> str:
        return DETECTOR_VERSION if self.use_owl else SEG_ONLY_VERSION

    def detect(self, rgb: np.ndarray) -> DetectionResult:
        notes: list[str] = []
        bg = estimate_background(rgb)
        regions: list[Region] = []
        fg = None
        if bg.plain:
            fg = foreground_mask(rgb, bg)
            for r in connected_regions(fg):
                parts = decompose(rgb, r)
                if len(parts) == 1 and parts[0].shape is None and estimate_style(rgb, r.mask) == "photo":
                    parts = [trim_contact_shadow(rgb, r, bg)]
                regions.extend(parts)
        else:
            notes.append(f"background not plain (uniformity {bg.uniform:.2f}); segmentation skipped")
        style = estimate_style(rgb, fg)
        dets = owl.detect(rgb, self.queries, self.owl_threshold) if self.use_owl else []
        used = [False] * len(dets)

        objects: list[Detection] = []
        shape_objs: list[tuple[Region, Detection]] = []
        for r in regions:
            rb = _box_xyxy(r.bbox)
            matches = [i for i, d in enumerate(dets)
                       if _iou(rb, d["box"]) >= 0.4
                       or (_inter(rb, d["box"]) / _area(d["box"]) >= 0.7 and _inter(rb, d["box"]) / _area(rb) >= 0.3)]
            if r.shape is not None:
                fit = r.shape
                conf = fit.score
                owl_label = None
                for i in matches:
                    used[i] = True
                    if dets[i]["label"] == fit.name and owl_label is None:
                        conf = 1 - (1 - fit.score) * (1 - dets[i]["score"])
                        owl_label = dets[i]["label"]
                color = name_rgb(r.color_rgb)
                det = Detection({
                    "name": fit.name, "display_name": f"{color} {fit.name}", "category": "shape",
                    "confidence": round(conf, 3), "bbox": fit.bbox, "color": color,
                    "colors": [{"name": color, "hex": "#%02x%02x%02x" % r.color_rgb, "fraction": 1.0}],
                    "shape": fit.name,
                    "attributes": {"visible_bbox": r.bbox, "visible_fraction": fit.coverage,
                                   "occluded": fit.coverage < 0.97, "fit_residual": fit.residual,
                                   "owl_agrees": owl_label is not None},
                    "detector": "geometry" + ("+owlv2" if owl_label else ""),
                }, r.mask)
                shape_objs.append((r, det))
                objects.append(det)
                continue

            strong = [i for i in matches if dets[i]["score"] >= self.keep_threshold]
            distinct = []
            for i in sorted(strong, key=lambda i: -dets[i]["score"]):
                if all(_iou(dets[i]["box"], dets[j]["box"]) < 0.3 for j in distinct):
                    distinct.append(i)
            for i in matches:
                used[i] = True
            if len(distinct) >= 2:  # one blob holding several touching objects → split by OWL boxes
                for i in distinct:
                    x0, y0, x1, y1 = [int(round(v)) for v in dets[i]["box"]]
                    sub = np.zeros_like(r.mask)
                    sub[y0:y1, x0:x1] = r.mask[y0:y1, x0:x1]
                    if sub.sum() < 0.2 * r.area / len(distinct):
                        continue
                    objects.append(self._semantic(rgb, dets[i]["label"], dets[i]["score"], sub,
                                                  bbox_of(sub), "segmentation+owlv2"))
            elif matches:
                best = max(matches, key=lambda i: dets[i]["score"])
                objects.append(self._semantic(rgb, dets[best]["label"], dets[best]["score"], r.mask,
                                              r.bbox, "segmentation+owlv2"))
            else:
                det = self._semantic(rgb, "unknown object", 0.2, r.mask, r.bbox, "segmentation")
                det.auto["attributes"]["needs_review"] = True
                objects.append(det)

        # mark which shape layers sit on top of which
        for r, det in shape_objs:
            det.auto["attributes"]["occluded_by"] = [
                shape_objs[j][1].auto["display_name"] for j in r.occluded_by if j < len(shape_objs)]
        self._swatches([d for _, d in shape_objs])

        h, w = rgb.shape[:2]
        for i, d in enumerate(dets):
            if used[i] or d["score"] < self.keep_threshold:
                continue
            box = d["box"]
            if bg.plain:
                x0, y0, x1, y1 = [int(round(v)) for v in box]
                if fg is None or fg[y0:y1, x0:x1].mean() < 0.15 or _area(box) > 0.9 * w * h:
                    notes.append(f"dropped OWLv2 '{d['label']}' {d['score']}: box is background")
                    continue
            mask = self._grabcut(rgb, box)
            objects.append(self._semantic(rgb, d["label"], d["score"], mask, _xyxy_to_bbox(box), "owlv2"))

        objects.sort(key=lambda o: (o.auto["bbox"]["x"], o.auto["bbox"]["y"]))
        return DetectionResult(objects, style, bg.plain, notes)

    # ------------------------------------------------------------ helpers
    def _semantic(self, rgb, label, score, mask, bbox, detector) -> Detection:
        colors = color_histogram(rgb, mask)
        color = dominant_color(colors)
        color = str(color) if color else None
        cat = category_of(label)
        attributes: dict = {}
        if cat == "person":
            # clothing: body below the head, minus skin, outlines and white highlights
            body = mask.copy()
            body[: bbox["y"] + int(0.22 * bbox["height"])] = False
            clothing = color_histogram(rgb, body, exclude={"skin", "black", "white", "gray"})
            attributes["clothing_colors"] = [str(c["name"]) for c in clothing if c["fraction"] >= 0.08]
            color = attributes["clothing_colors"][0] if attributes["clothing_colors"] else color
        display = f"{color} {label}" if color and color not in NEUTRALS and color != label and cat == "fruit" else label
        return Detection({
            "name": label, "display_name": display, "category": cat, "confidence": round(float(score), 3),
            "bbox": bbox, "color": color, "colors": colors, "shape": None,
            "attributes": attributes, "detector": detector,
        }, mask)

    @staticmethod
    def _grabcut(rgb, box) -> np.ndarray:
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        mask = np.zeros((h, w), np.uint8)
        rect = (max(0, x0), max(0, y0), max(1, x1 - x0), max(1, y1 - y0))
        try:
            bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
            cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
            out = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
            if out.sum() > 0:
                return out
        except cv2.error:
            pass
        out = np.zeros((h, w), bool)
        out[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = True
        return out

    @staticmethod
    def _swatches(shapes: list[Detection]) -> None:
        """Rows/columns of ≥3 equal, unoccluded rectangles in different colours are swatches."""
        rects = [d for d in shapes if d.auto["shape"] in ("square", "rectangle")
                 and not d.auto["attributes"]["occluded"]]
        if len(rects) < 3:
            return
        ws = np.array([d.auto["bbox"]["width"] for d in rects], float)
        hs = np.array([d.auto["bbox"]["height"] for d in rects], float)
        ys = np.array([d.auto["bbox"]["y"] for d in rects], float)
        xs = np.array([d.auto["bbox"]["x"] for d in rects], float)
        same_size = (np.abs(ws - np.median(ws)) <= 0.1 * np.median(ws)).all() and \
                    (np.abs(hs - np.median(hs)) <= 0.1 * np.median(hs)).all()
        aligned = np.ptp(ys) <= 0.1 * np.median(hs) or np.ptp(xs) <= 0.1 * np.median(ws)
        distinct = len({d.auto["color"] for d in rects}) == len(rects)
        if same_size and aligned and distinct:
            for d in rects:
                d.auto.update(name="color swatch", category="swatch",
                              display_name=f"{d.auto['color']} swatch")
                d.auto["attributes"]["swatch_shape"] = d.auto["shape"]
