"""Object detection backends.

ClassicalDetector  - offline, deterministic. Background subtraction + colour
                     segmentation + geometric shape fitting. Strong on
                     illustrations, product shots and flat graphics; reports
                     an explicit failure on cluttered photographic backgrounds.
ClaudeVisionDetector - hosted vision model. Uses the classical regions as
                     pixel-accurate proposals and lets the model name them
                     (boy vs girl, fruit type, ...). Falls back to model-drawn
                     boxes when no proposals exist (photos, clutter).

Both return a DetectionResult; the catalog never cares which one ran.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, width, height in source pixels
    color: str | None = None
    attributes: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    status: str  # "detected" | "no_objects" | "failed"
    backend: str
    objects: list[Detection] = field(default_factory=list)
    error: str | None = None


# ----------------------------------------------------------------------------
# colour naming
# ----------------------------------------------------------------------------
COLOR_NAMES = ["black", "white", "gray", "skin", "red", "orange", "brown",
               "yellow", "green", "blue", "purple", "pink"]
_IDX = {n: i for i, n in enumerate(COLOR_NAMES)}


def name_colors(rgb: np.ndarray) -> np.ndarray:
    """Vectorised HSV-bucket colour naming. Returns an index into COLOR_NAMES per pixel."""
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h = hsv[:, 0].astype(float) * 2  # OpenCV hue is 0..179
    s = hsv[:, 1] / 255.0
    v = hsv[:, 2] / 255.0
    out = np.full(len(h), _IDX["purple"], dtype=np.int8)
    rules = [  # first matching rule wins, so order matters
        (v < 0.2, "black"),
        ((s < 0.15) & (v > 0.85), "white"),
        (s < 0.15, "gray"),
        ((h >= 5) & (h < 40) & (s >= 0.15) & (s < 0.5) & (v > 0.6), "skin"),
        ((h < 15) | (h >= 345), "red"),
        ((h < 40) & (v < 0.6), "brown"),
        (h < 40, "orange"),
        (h < 70, "yellow"),
        (h < 170, "green"),
        (h < 255, "blue"),
        ((h >= 290) & (v > 0.75), "pink"),
    ]
    done = np.zeros(len(h), dtype=bool)
    for cond, name in rules:
        hit = cond & ~done
        out[hit] = _IDX[name]
        done |= hit
    return out.reshape(rgb.shape[:2])


# ----------------------------------------------------------------------------
# shape fitting
# ----------------------------------------------------------------------------
def classify_shape(mask: np.ndarray) -> tuple[str, float]:
    """Name the geometric primitive a binary mask best matches, plus a fit score 0..1.

    Uses the convex hull so partially occluded shapes still classify.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "blob", 0.0
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    area = cv2.contourArea(hull)
    if area < 1:
        return "blob", 0.0
    perimeter = cv2.arcLength(hull, True)
    circularity = 4 * math.pi * area / (perimeter ** 2)
    approx = cv2.approxPolyDP(hull, 0.04 * perimeter, True)
    solidity = float(mask.sum()) / area  # how much of the hull the region really fills

    if len(approx) == 3:
        _, tri = cv2.minEnclosingTriangle(hull)
        fit = area / max(cv2.contourArea(tri), 1)
        return "triangle", min(1.0, fit * solidity)
    if len(approx) == 4:
        (_, _), (w, h), _ = cv2.minAreaRect(hull)
        fit = area / max(w * h, 1)
        ratio = min(w, h) / max(w, h)
        return ("square" if ratio > 0.9 else "rectangle"), min(1.0, fit * solidity)
    if circularity > 0.8:
        (_, _), r = cv2.minEnclosingCircle(hull)
        fit = area / (math.pi * r * r)
        return "circle", min(1.0, fit * solidity)
    return "blob", min(1.0, circularity * solidity)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


# ----------------------------------------------------------------------------
# classical detector
# ----------------------------------------------------------------------------
def estimate_background(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Median border colour and how uniform the border is (mean distance from it)."""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(float)
    bg = np.median(border, axis=0)
    spread = float(np.linalg.norm(border - bg, axis=1).mean())
    return bg, spread


def foreground_mask(rgb: np.ndarray, bg: np.ndarray, threshold: float = 60.0) -> np.ndarray:
    dist = np.linalg.norm(rgb.astype(float) - bg, axis=2)
    mask = (dist > threshold).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


FRUIT_BY_COLOR = {"red": "apple", "orange": "orange"}


class ClassicalDetector:
    name = "classical-cv-v1"

    def __init__(self, min_area_frac: float = 0.002, region_frac: float = 0.08,
                 max_bg_spread: float = 25.0):
        self.min_area_frac = min_area_frac   # ignore specks smaller than this share of the image
        self.region_frac = region_frac       # colour regions smaller than this share of an object are details
        self.max_bg_spread = max_bg_spread   # border noise above this => background can't be modelled

    def detect(self, image: Image.Image) -> DetectionResult:
        rgb = np.asarray(image.convert("RGB"))
        bg, spread = estimate_background(rgb)
        if spread > self.max_bg_spread:
            return DetectionResult("failed", self.name, error=(
                f"non-uniform background (border spread {spread:.0f} > {self.max_bg_spread:.0f}); "
                "classical segmentation unreliable - add objects manually or use the vision backend"))

        fg = foreground_mask(rgb, bg)
        colors = name_colors(rgb)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        min_area = max(50, self.min_area_frac * fg.size)
        objects: list[Detection] = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            objects.extend(self._describe_component(labels == i, colors))
        objects.sort(key=lambda o: (o.bbox[0], o.bbox[1]))
        return DetectionResult("detected" if objects else "no_objects", self.name, objects)

    def _color_regions(self, comp: np.ndarray, colors: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Significant single-colour connected regions inside a component, largest first."""
        regions = []
        area = comp.sum()
        for idx in np.unique(colors[comp]):
            cmask = (comp & (colors == idx)).astype(np.uint8)
            k, lab, st, _ = cv2.connectedComponentsWithStats(cmask, connectivity=8)
            for j in range(1, k):
                if st[j, cv2.CC_STAT_AREA] >= self.region_frac * area:
                    regions.append((COLOR_NAMES[idx], lab == j))
        return sorted(regions, key=lambda r: -r[1].sum())

    def _describe_component(self, comp: np.ndarray, colors: np.ndarray) -> list[Detection]:
        regions = self._color_regions(comp, colors)
        palette = [c for c, _ in regions]
        dominant = palette[0] if palette else COLOR_NAMES[np.bincount(colors[comp]).argmax()]

        # 1) Figure: a skin-toned region inside a multi-coloured blob => a person.
        if "skin" in palette and len(regions) >= 3:
            clothing = sorted({c for c in palette if c not in ("skin",)})
            return [Detection("person", 0.6, _bbox(comp), dominant,
                              {"colors": clothing, "method": "skin+multi-region heuristic"})]

        # 2) Several clean geometric colour regions touching => overlapping shapes.
        if len(regions) >= 2:
            fitted = [(c, m, *classify_shape(m)) for c, m in regions]
            if all(shape != "blob" and fit >= 0.6 for _, _, shape, fit in fitted):
                return [Detection(shape, round(fit, 2), _bbox(m), c,
                                  {"overlapping": True, "method": "colour split + shape fit"})
                        for c, m, shape, fit in fitted]

        # 3) Single object: geometric primitive or round fruit. A perfectly flat
        #    single-colour disc is a drawn circle; a round object with detail
        #    (stem, leaf, shading) is treated as fruit.
        shape, fit = classify_shape(comp)
        attrs = {"shape": shape, "colors": palette, "method": "shape fit"}
        flat = (colors[comp] == _IDX[dominant]).mean() > 0.995
        if shape in ("circle", "blob") and dominant in FRUIT_BY_COLOR and not flat:
            return [Detection(FRUIT_BY_COLOR[dominant], round(0.5 + 0.35 * fit, 2), _bbox(comp), dominant, attrs)]
        if shape == "blob":
            return [Detection("object", round(0.3 * fit, 2), _bbox(comp), dominant, attrs)]
        return [Detection(shape, round(fit, 2), _bbox(comp), dominant, attrs)]


# ----------------------------------------------------------------------------
# hosted vision detector
# ----------------------------------------------------------------------------
_VISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["objects"],
    "properties": {"objects": {"type": "array", "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "color", "confidence", "proposal_index", "bbox"],
        "properties": {
            "label": {"type": "string"},
            "color": {"type": "string"},
            "confidence": {"type": "number"},
            "proposal_index": {"type": "integer"},
            "bbox": {"type": "object", "additionalProperties": False,
                     "required": ["x", "y", "w", "h"],
                     "properties": {k: {"type": "integer"} for k in "xywh"}},
        },
    }}},
}


class ClaudeVisionDetector:
    """Labels classical proposals with a vision LLM; draws its own boxes when there are none."""

    name = "claude-vision-v1"

    def __init__(self, proposer: ClassicalDetector | None = None, ask_json=None):
        self.proposer = proposer or ClassicalDetector()
        if ask_json is None:
            from common.llm import ask_json
        self.ask_json = ask_json

    def detect(self, image: Image.Image) -> DetectionResult:
        proposals = self.proposer.detect(image)
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        listing = "\n".join(f"{i}: bbox={o.bbox} colour={o.color} heuristic_label={o.label}"
                            for i, o in enumerate(proposals.objects)) or "(none - draw your own boxes)"
        prompt = (
            f"Image size: {image.width}x{image.height}. Identify every distinct object.\n"
            f"Pixel-accurate region proposals from a segmenter:\n{listing}\n\n"
            "For each object return a specific noun label (e.g. 'boy', 'girl', 'apple', 'circle'), its main "
            "colour, a confidence 0-1, the proposal_index it corresponds to (-1 if none) and a bbox in pixels "
            "(x, y, w, h). Correct the heuristic labels where they are wrong. Return an empty list if the "
            "image contains no objects.")
        try:
            data = self.ask_json([
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.standard_b64encode(buf.getvalue()).decode()}},
                {"type": "text", "text": prompt},
            ], _VISION_SCHEMA)
        except Exception as exc:  # network, auth, refusal: surface as a failed run, never crash ingestion
            return DetectionResult("failed", self.name, error=f"vision backend error: {exc}")

        objects = []
        for o in data["objects"]:
            idx = o["proposal_index"]
            if 0 <= idx < len(proposals.objects):
                bbox = proposals.objects[idx].bbox  # keep segmenter's exact box
                method = "vision label on cv proposal"
            else:
                b = o["bbox"]
                bbox = (b["x"], b["y"], b["w"], b["h"])
                method = "vision box"
            objects.append(Detection(o["label"].lower(), round(float(o["confidence"]), 2), bbox,
                                     o["color"].lower(), {"method": method}))
        return DetectionResult("detected" if objects else "no_objects", self.name, objects)


def default_detector():
    from common.llm import available
    return ClaudeVisionDetector() if available() else ClassicalDetector()
