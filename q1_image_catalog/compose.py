"""Natural-language composite generation from existing catalog assets.

Pipeline:  parse request -> resolve each item to a catalog object -> cut it
out (alpha mask) -> scale / position / layer -> save as a new catalog entry.

Guarantee: pixels only ever come from catalog images. If a requested item
has no matching asset the request fails with MissingAssetError listing what
is missing - nothing is silently generated or substituted.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image

from .catalog import Catalog, CatalogError
from .detect import COLOR_NAMES, estimate_background, foreground_mask, name_colors

NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
           "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
COLORS = set(COLOR_NAMES) | {"grey", "violet"}
FILLER = {"the", "some", "of", "colored", "coloured", "different", "few", "several", "image", "picture"}
LEAD_IN = re.compile(r"^(please\s+)?(create|make|generate|compose|draw|build|render)\s+"
                     r"(an?\s+)?(new\s+)?(image|picture|composite|scene)?\s*"
                     r"(containing|with|of|showing|that has|having)?\s*")

# Relative on-canvas height per category: people are tall, fruit is small.
SIZE_PRIOR = {"person": 1.0, "boy": 1.0, "girl": 1.0, "man": 1.0, "woman": 1.0,
              "apple": 0.42, "orange": 0.42}
DEFAULT_SIZE = 0.45


class MissingAssetError(CatalogError):
    def __init__(self, missing: list[str], available: list[str]):
        self.missing, self.available = missing, available
        super().__init__(f"no catalog asset for: {', '.join(missing)}. "
                         f"Add or label one first (available: {', '.join(available)})")


@dataclass
class Item:
    term: str               # noun as requested, e.g. "apple", "shapes"
    count: int | None       # None = "all of them" (bare plural)
    color: str | None = None


def _singular(word: str) -> str:
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("es") and word[:-2].endswith(("sh", "ch", "x", "s")):
        return word[:-2]
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def parse_request(text: str) -> list[Item]:
    """Rule-based parser: 'one red apple and one orange' -> [Item(apple,1,red), Item(orange,1)]."""
    text = LEAD_IN.sub("", text.lower().strip().rstrip("."))
    items = []
    for chunk in re.split(r",|\band\b|\bplus\b|&|\bwith\b", text):
        words = [w for w in re.findall(r"[a-z0-9]+", chunk) if w not in FILLER]
        if not words:
            continue
        count = None
        if words[0] in NUMBERS or words[0].isdigit():
            count = NUMBERS.get(words[0]) or int(words[0])
            words = words[1:]
        colors = [w for w in words if w in COLORS]
        nouns = [w for w in words if w not in COLORS]
        if not nouns and colors:  # "one orange": the colour word *is* the noun
            nouns = [colors.pop()]
        if not nouns:
            continue
        term = " ".join(nouns)
        if count is None and _singular(term) == term:
            count = 1  # bare singular ("boy") means one
        items.append(Item(term, count, colors[0] if colors else None))
    return items


_PARSE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["term", "count", "color"],
        "properties": {"term": {"type": "string"},
                       "count": {"type": ["integer", "null"]},
                       "color": {"type": ["string", "null"]}}}}},
}


def parse_request_llm(text: str, vocabulary: list[str], ask_json=None) -> list[Item]:
    """LLM parser for free-form phrasing. Terms are constrained to catalog vocabulary."""
    if ask_json is None:
        from common.llm import ask_json
    data = ask_json(
        f"Catalog vocabulary (object labels and image names): {', '.join(vocabulary)}\n"
        f"Request: {text!r}\n"
        "List each item the image must contain. term = the closest vocabulary entry (keep the user's "
        "word if nothing fits), count = number requested (null when they mean all of a group), "
        "color = requested colour or null.", _PARSE_SCHEMA)
    return [Item(i["term"].lower(), i["count"], i["color"]) for i in data["items"]]


# -----------------------------------------------------------------------------
# resolution
# -----------------------------------------------------------------------------
def _sources(catalog: Catalog) -> list[dict]:
    return [img for img in catalog.list() if img["auto"]["kind"] == "upload" and img["objects"]]


def vocabulary(catalog: Catalog) -> list[str]:
    words = set()
    for img in _sources(catalog):
        words.add(img["name"])
        words.update(o["label"] for o in img["objects"])
    return sorted(words)


def resolve(catalog: Catalog, items: list[Item]) -> list[tuple[Item, dict, dict]]:
    """Map each requested item to concrete (image, object) assets. Raises MissingAssetError."""
    images = _sources(catalog)
    picks, missing = [], []
    for item in items:
        names = {item.term, _singular(item.term)}
        # 1) objects whose label/display name matches, e.g. 'apple' or 'red square'
        candidates = []
        for img in images:
            for obj in img["objects"]:
                labels = {obj["label"].lower(), (obj["display_name"] or "").lower()}
                if names & labels and (item.color is None or obj["color"] == item.color):
                    rank = (img["name"] not in names,          # dedicated image named after it
                            len(img["objects"]),                 # fewer objects = cleaner asset
                            not obj["edited_by_user"],           # user-verified first
                            -(obj["confidence"] or 0))
                    candidates.append((rank, img, obj))
        if candidates:
            _, img, obj = min(candidates, key=lambda c: c[0])
            picks += [(item, img, obj)] * (item.count or 1)
            continue
        # 2) an image named after the term, e.g. 'shapes' or 'boy'
        img = next((i for i in images if i["name"] in names | {item.term + "s"}), None)
        if img:
            if item.count is None:  # the whole group, in its original left-to-right order
                chosen = img["objects"]
            else:
                chosen = [max(img["objects"], key=lambda o: o["bbox"]["w"] * o["bbox"]["h"])] * item.count
            picks += [(item, img, o) for o in chosen]
            continue
        missing.append(item.term)
    if missing:
        raise MissingAssetError(missing, vocabulary(catalog))
    return picks


# -----------------------------------------------------------------------------
# rendering
# -----------------------------------------------------------------------------
def cutout(catalog: Catalog, img: dict, obj: dict) -> Image.Image:
    """RGBA crop of one object with its background removed."""
    rgb = np.asarray(catalog.load_pil(img["id"]))
    b = obj["bbox"]
    crop = rgb[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]]
    bg, spread = estimate_background(rgb)
    if spread > 25:  # cluttered source: no reliable matte, keep the rectangle
        alpha = np.full(crop.shape[:2], 255, np.uint8)
    else:
        mask = foreground_mask(crop, bg)
        if obj["attributes"].get("overlapping") and obj["color"] in COLOR_NAMES:
            # split from overlapping shapes: keep only this object's own colour
            mask &= (name_colors(crop) == COLOR_NAMES.index(obj["color"])).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n > 1:  # drop fragments of neighbouring objects that leak into the bbox
            mask = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        alpha = mask * 255
    return Image.fromarray(np.dstack([crop, alpha]), "RGBA")


def compose(catalog: Catalog, request: str, name: str | None = None, unit: int = 300,
            gap: int = 40, margin: int = 40, use_llm: bool | None = None, ask_json=None) -> dict:
    """Build a composite for ``request`` and store it as a new catalog entry."""
    if use_llm is None:
        from common.llm import available
        use_llm = available()
    items = (parse_request_llm(request, vocabulary(catalog), ask_json) if use_llm
             else parse_request(request))
    if not items:
        raise CatalogError(f"could not find any items in request {request!r}")
    picks = resolve(catalog, items)

    # Scale to a category height, keeping aspect ratio. All pieces of one
    # requested item share a scale so a group (e.g. 'shapes') keeps its
    # relative sizes.
    pieces = [(item, img, obj, cutout(catalog, img, obj)) for item, img, obj in picks]
    item_scale = {}
    for item, img, obj, piece in pieces:
        key = item.term if item.term in SIZE_PRIOR else obj["label"]
        target_h = unit * SIZE_PRIOR.get(key, DEFAULT_SIZE)
        item_scale[id(item)] = min(item_scale.get(id(item), float("inf")), target_h / piece.height)
    layers = []
    for item, img, obj, piece in pieces:
        scale = item_scale[id(item)]
        size = (max(1, round(piece.width * scale)), max(1, round(piece.height * scale)))
        layers.append((item, img, obj, piece.resize(size, Image.LANCZOS), scale))

    # row layout on a shared ground line; tallest items drawn first (behind)
    width = margin * 2 + sum(p.width for *_, p, _ in layers) + gap * (len(layers) - 1)
    height = margin * 2 + max(p.height for *_, p, _ in layers)
    canvas = Image.new("RGB", (width, height), "white")
    placements, x = [], margin
    for idx, (item, img, obj, piece, scale) in enumerate(layers):
        y = height - margin - piece.height
        placements.append((idx, item, img, obj, piece, scale, x, y))
        x += piece.width + gap
    ordered = sorted(placements, key=lambda p: -p[4].height)  # painter's order
    for idx, item, img, obj, piece, scale, px, py in ordered:
        canvas.paste(piece, (px, py), piece)

    provenance = {
        "request": request,
        "parser": "llm" if use_llm else "rules",
        "items": [asdict(i) for i in items],
        "layout": "row, bottom-aligned, size prior per category",
        "generated_pixels": False,  # every pixel comes from a catalog asset
        "sources": [{"image_id": img["id"], "image_name": img["name"], "object_id": obj["id"],
                     "label": obj["label"], "source_bbox": obj["bbox"], "scale": round(scale, 3),
                     "placed_bbox": {"x": px, "y": py, "w": piece.width, "h": piece.height},
                     "z": z}
                    for z, (idx, item, img, obj, piece, scale, px, py) in enumerate(ordered)],
    }
    name = name or "composite_" + "_".join(i.term.replace(" ", "-") for i in items)
    base, n = name, 2
    while catalog.db.execute("SELECT 1 FROM images WHERE name = ?", (name,)).fetchone():
        name, n = f"{base}_{n}", n + 1
    entry = catalog.add_image(canvas, name, description=request, detect=False, kind="composite",
                              provenance=provenance, tags={"generated": "composite"})
    # Object metadata for a composite is known exactly from the layout - no detection needed.
    with catalog.db:
        for idx, item, img, obj, piece, scale, px, py in placements:
            catalog._insert_object(entry["id"], "composer", "compose-v1", obj["label"], obj["color"], 1.0,
                                   (px, py, piece.width, piece.height),
                                   {"source_image": img["name"], "source_object": obj["id"]},
                                   display_name=item.term if item.count else obj["display_name"])
        catalog.db.execute("UPDATE images SET detection_status = 'detected', detection_backend = "
                           "'compose-v1' WHERE id = ?", (entry["id"],))
    return catalog.get(entry["id"])
