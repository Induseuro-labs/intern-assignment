"""Q1 vision layer: background removal + connected components + color/shape heuristics.

No model weights, no API keys. Works on cartoon AND photo-style assets because
the catalog uses plain backgrounds; anything it gets wrong is fixed by manual
correction in catalog.py, which stays the source of truth.
"""
import numpy as np
from scipy.ndimage import label as paint_label

BG_THRESH = 235       # channels above this count as (near-)white background
MIN_AREA = 150        # smaller regions are anti-aliasing noise
SPLIT_SHARE = 0.15    # color share that counts as a "major" color

PALETTE = {  # name -> rgb; auto color naming snaps to the nearest entry
    "red": (230, 30, 30), "orange": (245, 130, 20), "yellow": (240, 210, 30),
    "green": (40, 160, 60), "blue": (40, 100, 230), "purple": (140, 60, 200),
    "pink": (240, 150, 200), "brown": (150, 90, 40), "skin": (255, 214, 170),
    "black": (30, 30, 30), "gray": (150, 150, 150), "navy": (40, 40, 90),
}
AUTO_FIELDS = ("bbox", "color", "colors", "confidence", "shape", "label")
USER_FIELDS = ("name", "color", "bbox", "confidence")  # user-editable subset


def color_name(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return min(PALETTE, key=lambda n: sum((a - b) ** 2 for a, b in zip(PALETTE[n], rgb)))


def _shape_of(area, bbox):
    x, y, w, h = bbox
    fill = area / max(1, w * h)
    squarish = abs(w - h) <= max(4, 0.12 * max(w, h))
    if fill > 0.88:
        return "square" if squarish else "rectangle"
    if (w > 1.6 * h or h > 1.6 * w) and fill < 0.9:
        return "figure"  # tall/wide articulated thing (person), not a shape
    if fill > 0.62:
        return "circle"
    if fill > 0.38:
        return "triangle"
    return "figure"


def detect_objects(image):
    """Return [dict(label,color,colors,bbox,confidence,shape)] for a PIL image."""
    arr = np.asarray(image.convert("RGB"))
    H, W, _ = arr.shape
    fg = ~(arr > BG_THRESH).all(axis=2)
    if fg.sum() < max(MIN_AREA, 0.005 * H * W):
        return []  # no detectable objects
    out, cands = [], []
    # One foreground component = one object (touching parts, e.g. a cartoon
    # figure, stay merged; separated items stay separate).
    lab, n = paint_label(fg, structure=np.ones((3, 3)))
    for i in range(1, n + 1):
        m = lab == i
        if int(m.sum()) >= MIN_AREA:
            cands.append(m)
    biggest = max(int(m.sum()) for m in cands)
    for m in cands:
        area = int(m.sum())
        if area < max(MIN_AREA, 0.15 * biggest):
            continue  # satellite speck (leaf, stem): part of the main object
        ys, xs = np.where(m)
        box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
               int(ys.max() - ys.min() + 1))
        mean = arr[m].mean(axis=0)
        # Major colors inside this object (for clothing palettes / overlaps).
        # Regions are per-color *connected* components, so disjoint same-color
        # parts (head vs legs) get their own boxes instead of one giant one.
        q = (arr // 64 * 64)
        uniq, counts = np.unique(q[m].reshape(-1, 3), axis=0, return_counts=True)
        majors = []
        for c, k in sorted(zip(uniq, counts), key=lambda t: -t[1]):
            if k < SPLIT_SHARE * area or (c > BG_THRESH).all():
                continue
            sm = m & (q == c).all(axis=2)
            rl, rn = paint_label(sm, structure=np.ones((3, 3)))
            for r in range(1, rn + 1):
                rm = rl == r
                if int(rm.sum()) < MIN_AREA:
                    continue
                sy, sx = np.where(rm)
                rb = (int(sx.min()), int(sy.min()), int(sx.max() - sx.min() + 1),
                      int(sy.max() - sy.min() + 1))
                majors.append((color_name(arr[rm].mean(axis=0)), rb, rm))
        if len(majors) >= 2:
            # Occlusion (split) vs adjacency/detail (merge): split only if two
            # regions overlap substantially yet neither contains the other.
            # A leaf inside an apple box is ~100% contained -> detail, merged.
            boxes = [b for _, b, _ in majors]
            split = False
            for k in range(len(boxes)):
                for l in range(k + 1, len(boxes)):
                    a, b = boxes[k], boxes[l]
                    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                    inter = max(0, iw) * max(0, ih)
                    small = min(a[2] * a[3], b[2] * b[3])
                    if inter > 0.005 * H * W and inter / small < 0.7:
                        split = True
            if split:  # truly overlapping -> one object per color
                bycol = {}
                for col, rb, rm in majors:
                    bycol.setdefault(col, []).append((rb, rm))
                for col, parts in bycol.items():
                    xs0 = min(p[0] for p, _ in parts)
                    ys0 = min(p[1] for p, _ in parts)
                    xs1 = max(p[0] + p[2] for p, _ in parts)
                    ys1 = max(p[1] + p[3] for p, _ in parts)
                    sb = (xs0, ys0, xs1 - xs0, ys1 - ys0)
                    shape = _shape_of(sum(int(x.sum()) for _, x in parts), sb)
                    out.append({"label": f"{col} {shape}", "color": col,
                                "colors": [col], "bbox": sb, "confidence": 0.55,
                                "shape": shape})
                continue
        cols = [c for c, _, _ in majors] or [color_name(mean)]
        shape = _shape_of(area, box)
        solid = area / (box[2] * box[3])
        conf = round(min(0.95, 0.45 + 0.45 * solid + 0.05 * (area / fg.sum())), 2)
        out.append({"label": f"{cols[0]} {shape}", "color": cols[0],
                    "colors": cols, "bbox": box, "confidence": conf, "shape": shape})
    return out
