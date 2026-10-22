"""Composite generation from existing catalog assets.

Pipeline: plan (LLM or rules) → resolve every item to a catalog object
(fail loudly if any is missing — never synthesise a substitute) → cut the
object out of its source image with its stored mask → scale with real-world
size priors → lay out (row / grid groups / overlapping layers) → render →
save as a new catalog entry whose objects and provenance are exact.
"""
from __future__ import annotations

import hashlib
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..detection.colors import PROTOTYPES
from ..detection.segmentation import estimate_background, foreground_mask
from ..store import CatalogObject
from .planner import Plan, PlanItem, make_plan

COMPOSER_VERSION = "composer-1.0"

# Typical real-world heights (cm) so a boy is taller than an apple; graphics get a nominal size.
SIZE_PRIORS = {"apple": 8, "orange": 8, "banana": 18, "boy": 120, "girl": 115, "man": 175,
               "woman": 165, "person": 165, "dog": 50, "cat": 30}
CATEGORY_PRIORS = {"fruit": 8, "person": 120, "shape": 36, "swatch": 36, "animal": 40, "object": 30}
TALLEST_PX = 420
SYNONYMS = {"kid": "person", "child": "person", "ball": "circle"}


class MissingAssetError(LookupError):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(
            "the catalog has no asset for: " + ", ".join(missing) +
            ". Add an image containing it first; the composer never substitutes generated approximations.")


@dataclass
class Placement:
    obj: CatalogObject
    item_index: int
    rgba: Image.Image = None
    mask_source: str = ""
    crop_sha256: str = ""
    scale: float = 1.0
    x: int = 0
    y: int = 0
    z: int = 0
    source_style: str | None = None


# ------------------------------------------------------------- inventory
def inventory(svc) -> dict:
    inv: dict[str, dict] = {}
    for img in svc.store.list_images():
        if img.kind != "original":
            continue
        for o in img.live_objects:
            e = o.effective
            if not e.get("name") or e["name"] == "unknown object":
                continue
            v = inv.setdefault(e["name"], {"category": e.get("category") or "object", "colors": set(), "count": 0})
            if e.get("color"):
                v["colors"].add(e["color"])
            v["count"] += 1
    return inv


def _matches(e: dict, item: PlanItem) -> bool:
    noun = SYNONYMS.get(item.noun, item.noun)
    if item.is_category:
        ok = e.get("category") == noun
    else:
        ok = noun in (e.get("name"), e.get("display_name")) or (noun == "person" and e.get("category") == "person")
    if ok and item.color:
        ok = e.get("color") == item.color or any(
            c["name"] == item.color and c["fraction"] >= 0.3 for c in (e.get("colors") or []))
    return ok


def resolve(svc, plan: Plan) -> list[tuple[int, list[CatalogObject]]]:
    images = [i for i in svc.store.list_images() if i.kind == "original"]
    resolved, missing = [], []
    for idx, item in enumerate(plan.items):
        cands = []
        for img in images:
            n_live = len(img.live_objects)
            for o in img.live_objects:
                e = o.effective
                if _matches(e, item):
                    occluded = bool((e.get("attributes") or {}).get("occluded"))
                    verified = img.objects_source == "manual"
                    cands.append(((occluded, n_live, not verified, -(e.get("confidence") or 0)), img.id, o))
        if not cands:
            missing.append((f"{item.color} " if item.color else "") + item.noun + (" (category)" if item.is_category else ""))
            continue
        cands.sort(key=lambda c: c[0])
        if item.count == 0:  # every distinct member of the category, from the best single image
            by_img: dict[str, list] = {}
            for c in cands:
                by_img.setdefault(c[1], []).append(c)
            def img_score(cs):
                names = {(c[2].effective["name"], c[2].effective.get("color")) for c in cs if not c[0][0]}
                return (len(names), -min(c[0][1] for c in cs))
            best = max(by_img.values(), key=img_score)
            seen, chosen = set(), []
            for c in best:
                key = (c[2].effective["name"], c[2].effective.get("color"))
                if key not in seen:
                    seen.add(key)
                    chosen.append(c[2])
            resolved.append((idx, chosen))
        else:
            distinct = [c[2] for c in cands]
            resolved.append((idx, [distinct[k % len(distinct)] if k < len(distinct) else distinct[0]
                                   for k in range(item.count)]))
    if missing:
        raise MissingAssetError(missing)
    return resolved


# --------------------------------------------------------------- cutouts
def cutout(svc, obj: CatalogObject) -> tuple[Image.Image, str, str]:
    """RGBA cut of the object's own pixels from its source image."""
    rgb = svc.load_rgb(obj.image_id)
    h, w = rgb.shape[:2]
    b = obj.effective["bbox"]
    x0, y0 = max(0, b["x"]), max(0, b["y"])
    x1, y1 = min(w, b["x"] + b["width"]), min(h, b["y"] + b["height"])
    box = np.zeros((h, w), bool)
    box[y0:y1, x0:x1] = True
    mask = svc.object_mask(obj)
    if mask is not None:
        source = "stored detection mask"
        mask = mask & box
    else:
        bg = estimate_background(rgb)
        if bg.plain:
            mask = foreground_mask(rgb, bg) & box
            source = "background subtraction inside bbox"
        else:
            gc = np.zeros((h, w), np.uint8)
            bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
            cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, (x0, y0, x1 - x0, y1 - y0), bgd, fgd, 4,
                        cv2.GC_INIT_WITH_RECT)
            mask = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD))
            source = "grabcut inside bbox"
        if mask.sum() < 0.05 * (x1 - x0) * (y1 - y0):  # mask failed → use the whole box
            mask = box
            source = "bbox rectangle"
    ys, xs = np.nonzero(mask)
    cy0, cy1, cx0, cx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    alpha = Image.fromarray((mask[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
    rgba = Image.fromarray(rgb[cy0:cy1, cx0:cx1]).convert("RGBA")
    rgba.putalpha(alpha)
    digest = hashlib.sha256(rgba.tobytes()).hexdigest()[:16]
    return rgba, source, digest


# ---------------------------------------------------------------- layout
def _prior(e: dict) -> float:
    return SIZE_PRIORS.get(e.get("name"), CATEGORY_PRIORS.get(e.get("category") or "object", 30))


def _scale_to_prior(p: Placement, px_per_unit: float) -> None:
    e = p.obj.effective
    w, h = p.rgba.size
    graphic = e.get("category") in ("shape", "swatch")
    ref = max(w, h) if graphic else h
    p.scale = max(0.05, _prior(e) * px_per_unit / ref)
    p.rgba = p.rgba.resize((max(1, round(w * p.scale)), max(1, round(h * p.scale))), Image.LANCZOS)


def layout(placements: list[Placement], plan: Plan) -> tuple[int, int]:
    tallest = max(_prior(p.obj.effective) for p in placements)
    ppu = TALLEST_PX / tallest
    for p in placements:
        _scale_to_prior(p, ppu)
    gap = int(0.12 * TALLEST_PX)
    margin = int(0.15 * TALLEST_PX)

    # group placements by plan item: category groups of 3+ are packed as a 2-row grid
    groups: list[list[Placement]] = []
    for p in placements:
        if groups and groups[-1][0].item_index == p.item_index:
            groups[-1].append(p)
        else:
            groups.append([p])

    if plan.layout == "overlap":
        x, z = margin, 0
        for p in placements:
            p.x, p.z = x, z
            x += int(p.rgba.width * 0.7)
            z += 1
        width = max(p.x + p.rgba.width for p in placements) + margin
        height = max(p.rgba.height for p in placements) + 2 * margin
        for p in placements:
            p.y = height - margin - p.rgba.height
        return width, height

    blocks = []  # (group, block_w, block_h, local offsets)
    for g in groups:
        if len(g) >= 3 or plan.layout == "grid":
            cols = math.ceil(len(g) / 2) if plan.layout != "grid" else math.ceil(math.sqrt(len(g)))
            rows = math.ceil(len(g) / cols)
            cw = max(p.rgba.width for p in g)
            ch = max(p.rgba.height for p in g)
            cg = int(gap * 0.5)
            offs = []
            for k, p in enumerate(g):
                r, c = divmod(k, cols)
                offs.append((c * (cw + cg) + (cw - p.rgba.width) // 2, r * (ch + cg) + (ch - p.rgba.height)))
            blocks.append((g, cols * cw + (cols - 1) * cg, rows * ch + (rows - 1) * cg, offs))
        else:
            offs, x = [], 0
            bh = max(p.rgba.height for p in g)
            for p in g:
                offs.append((x, bh - p.rgba.height))
                x += p.rgba.width + gap
            blocks.append((g, x - gap, bh, offs))

    height = max(b[2] for b in blocks) + 2 * margin
    ground = height - margin
    x = margin
    z = 0
    for g, bw, bh, offs in blocks:
        for p, (ox, oy) in zip(g, offs):
            p.x, p.y, p.z = x + ox, ground - bh + oy, z
            z += 1
        x += bw + gap
    return x - gap + margin, height


def render(placements: list[Placement], size: tuple[int, int], background: str) -> Image.Image:
    bg = PROTOTYPES.get(background, [(255, 255, 255)])[0] if background != "white" else (255, 255, 255)
    canvas = Image.new("RGBA", size, bg + (255,))
    for p in sorted(placements, key=lambda p: p.z):
        if p.source_style == "photo":  # soft contact shadow so photographed objects sit on the ground
            sh = Image.new("RGBA", size, (0, 0, 0, 0))
            w, h = p.rgba.size
            ImageDraw.Draw(sh).ellipse((p.x + w * 0.1, p.y + h * 0.93, p.x + w * 0.9, p.y + h * 1.03),
                                       fill=(0, 0, 0, 70))
            canvas = Image.alpha_composite(canvas, sh.filter(ImageFilter.GaussianBlur(max(2, w * 0.03))))
        canvas.alpha_composite(p.rgba, (p.x, p.y))
    return canvas.convert("RGB")


# ------------------------------------------------------------ entry point
def compose(svc, instruction: str, display_name: str | None = None, use_llm: bool | None = None,
            verify: bool = False):
    inv = inventory(svc)
    plan, notes = make_plan(instruction, inv, use_llm)
    resolved = resolve(svc, plan)

    placements: list[Placement] = []
    for idx, objs in resolved:
        for obj in objs:
            rgba, src, digest = cutout(svc, obj)
            style = svc.store.get_image(obj.image_id).style
            placements.append(Placement(obj, idx, rgba, src, digest, source_style=style))
    size = layout(placements, plan)
    image = render(placements, size, plan.background)

    provenance = {
        "composer": COMPOSER_VERSION,
        "instruction": instruction,
        "plan": plan.to_dict(),
        "notes": notes,
        "canvas": {"width": size[0], "height": size[1], "background": plan.background},
        "generative_fill": False,  # every pixel of every object is copied from a catalog asset
        "placements": [],
    }
    objects = []
    for p in placements:
        e = p.obj.effective
        src_img = svc.store.get_image(p.obj.image_id)
        bbox = {"x": p.x, "y": p.y, "width": p.rgba.width, "height": p.rgba.height}
        provenance["placements"].append({
            "object_id": p.obj.id, "source_image_id": src_img.id, "source_image": src_img.display_name,
            "source_bbox": e["bbox"], "mask_source": p.mask_source, "crop_sha256": p.crop_sha256,
            "scale": round(p.scale, 4), "bbox": bbox, "z_index": p.z,
        })
        objects.append({
            "name": e["name"], "display_name": e.get("display_name"), "category": e.get("category"),
            "color": e.get("color"), "colors": e.get("colors"), "shape": e.get("shape"),
            "confidence": 1.0, "bbox": bbox, "detector": "composition",
            "attributes": {"source_object_id": p.obj.id, "source_image_id": src_img.id, "z_index": p.z,
                           **({"clothing_colors": e["attributes"]["clothing_colors"]}
                              if (e.get("attributes") or {}).get("clothing_colors") else {})},
        })

    if verify:
        provenance["verification"] = _verify(svc, image, objects)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "composite.png"
        image.save(path)
        image_id = svc.store.add_image_file(path, display_name or None, kind="composite",
                                            provenance=provenance, tags=["composite"])
    if not display_name:
        svc.store.update_image_user_fields(image_id, display_name=f"composite-{image_id[:6]}")
    svc.store.set_image_auto(image_id, objects_source="composition",
                             style=placements[0].source_style if len({p.source_style for p in placements}) == 1 else "mixed")
    for auto in objects:
        svc.store.insert_object(image_id, "composition", auto=auto)
    return svc.store.get_image(image_id)


def _verify(svc, image: Image.Image, objects: list[dict]) -> dict:
    """Run the detector on the composite and check that every placed object is found."""
    from ..detection.pipeline import _iou, _box_xyxy

    result = svc.detector.detect(np.array(image))
    found = []
    for o in objects:
        hit = max((( _iou(_box_xyxy(o["bbox"]), _box_xyxy(d.auto["bbox"])), d.auto["name"]) for d in result.objects),
                  default=(0, None))
        found.append({"placed": o["name"], "detected_as": hit[1], "iou": round(hit[0], 3),
                      "ok": hit[0] >= 0.5 and hit[1] in (o["name"], o.get("category"))})
    return {"detector": svc.detector.version, "all_found": all(f["ok"] for f in found), "objects": found}
