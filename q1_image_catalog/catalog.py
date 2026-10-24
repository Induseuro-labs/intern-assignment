"""Q1 catalog layer: entries + manual correction + composite generation.

Manual edits win until rerun_detection() is explicitly called. Auto-only vs
user-editable fields are enforced in edit_object(). Composites reuse real
catalog pixels; missing assets raise instead of being AI-substituted.
"""
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from detector import (AUTO_FIELDS, BG_THRESH, PALETTE, USER_FIELDS,
                      detect_objects)


@dataclass
class DetObj:
    id: int
    name: str                      # USER-EDITABLE (defaults to auto label)
    auto_label: str                # auto snapshot, never user-edited
    color: str
    colors: list = field(default_factory=list)
    bbox: tuple = (0, 0, 0, 0)
    confidence: float = 0.0
    shape: str = ""
    corrected: bool = False        # True once a human touches it
    auto_snapshot: dict = field(default_factory=dict)


@dataclass
class Entry:
    name: str
    image: Image.Image
    objects: list = field(default_factory=list)
    status: str = "detected"       # detected|no_objects|manual|composite
    source: dict = field(default_factory=dict)  # composite provenance


class Catalog:
    """Reusable image collection. Manual edits win until rerun_detection()."""

    def __init__(self):
        self.items = {}
        self._next_id = 1

    def add_image(self, name, image, object_names=None, source=None):
        det = detect_objects(image)
        objs = []
        for j, d in enumerate(det):
            uname = (object_names or {}).get(j, d["label"]) if isinstance(
                object_names, dict) else (object_names[j] if object_names and j < len(object_names) else d["label"])
            objs.append(DetObj(self._next_id + j, uname, d["label"], d["color"],
                               d["colors"], d["bbox"], d["confidence"], d["shape"],
                               auto_snapshot={k: d[k] for k in AUTO_FIELDS}))
        self._next_id += len(objs)
        e = Entry(name, image.convert("RGB"), objs,
                  "detected" if objs else "no_objects", source or {})
        self.items[name] = e
        return e

    def _obj(self, entry, oid):
        for o in self.items[entry].objects:
            if o.id == oid:
                return o
        raise KeyError(f"object {oid} not in {entry}")

    def add_object(self, entry, obj_name, bbox, color="gray", confidence=1.0):
        e = self.items[entry]  # manual add when detection missed something
        o = DetObj(self._next_id, obj_name, obj_name + " (manual)", color,
                   [color], tuple(bbox), confidence, "manual",
                   corrected=True, auto_snapshot={})
        self._next_id += 1
        e.objects.append(o)
        e.status = "manual"
        return o

    def edit_object(self, entry, oid, **kw):
        o = self._obj(entry, oid)  # only USER_FIELDS are writable
        bad = set(kw) - set(USER_FIELDS)
        if bad:
            raise ValueError(f"auto-only fields, not editable: {sorted(bad)}")
        for k, v in kw.items():
            setattr(o, k, tuple(v) if k == "bbox" else v)
        o.corrected = True
        self.items[entry].status = "manual"
        return o

    def remove_object(self, entry, oid):
        e = self.items[entry]
        e.objects = [o for o in e.objects if o.id != oid]
        e.status = "manual"

    def rerun_detection(self, entry):
        e = self.items[entry]  # explicit rerun discards manual corrections
        fresh = self.add_image(entry + "__tmp", e.image)
        del self.items[entry + "__tmp"]
        e.objects = fresh.objects
        e.status = "detected" if e.objects else "no_objects"
        return e

    # ---- composite generation: natural language -> reuse of catalog assets ----
    def generate_composite(self, prompt, new_name, canvas_bg="white"):
        wants = self._parse(prompt)
        missing = self._unknown_nps(prompt)
        if not wants and not missing:
            raise ValueError(f"no catalog assets mentioned in: {prompt!r}")
        assets = []
        for count, color, noun in wants:
            hit = [n for n in self.items
                   if n == noun or noun in n or n in noun] or ([noun] if noun in self.items else [])
            if color and not any(o.color == color for n in hit for o in self.items[n].objects):
                missing.append(f"{color} {noun}")  # asked color we don't have
                continue
            if color:
                hit = [n for n in hit if any(
                    o.color == color for o in self.items[n].objects)] or hit
            (assets.extend([hit[0]] * count) if hit else missing.append(noun))
        if missing:  # never silently swap in a generated approximation
            raise ValueError(f"missing catalog assets (add them first): {missing}")
        cell, per = 250, 300
        canvas = Image.new("RGB", (cell * len(assets), per), canvas_bg)
        transforms = []
        for k, a in enumerate(assets):
            im = self.items[a].image
            fg = ~(np.asarray(im.convert("RGB")) > BG_THRESH).all(axis=2)
            alpha = Image.fromarray((fg * 255).astype("uint8"))
            im = im.copy()
            im.thumbnail((cell - 30, per - 30))
            x = k * cell + (cell - im.width) // 2
            y = (per - im.height) // 2
            canvas.paste(im, (x, y), alpha.resize(im.size))
            transforms.append({"asset": a, "pos": (x, y), "size": im.size})
        e = self.add_image(new_name, canvas, source={"prompt": prompt,
                           "sources": assets, "transforms": transforms})
        e.status = "composite"
        return e

    def _parse(self, prompt):
        """Extract (count, color, catalog-noun) triples from free text."""
        out, seen = [], set()
        for noun in sorted(self.items, key=len, reverse=True):
            for m in re.finditer(
                    r"(?:\b(one|two|three|\d+|a|an)\b\s+)?(?:\b(\w+)\b\s+)?\b"
                    + re.escape(noun) + r"s?\b", prompt, re.I):
                if m.span() in seen:
                    continue
                seen.add(m.span())
                cw, adj = (m.group(1) or "one").lower(), (m.group(2) or "").lower()
                n = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}.get(cw)
                out.append((m.start(), int(cw) if n is None else n,
                            adj if adj in PALETTE else None, noun))
        return [(c, col, n) for _, c, col, n in sorted(out)]

    def _unknown_nps(self, prompt):
        """Noun phrases after a count-word that match no catalog entry.

        Catches requests like 'one dragon fruit': the parser can only resolve
        known nouns, so anything else that looks like an asset request must be
        reported, never silently dropped or AI-regenerated.
        """
        ADJ = set(PALETTE) | {"big", "small", "single", "colored", "cartoon",
                              "standing", "fresh", "whole", "sitting"}
        STOP = {"image", "picture", "photo", "scene", "containing", "composition",
                "collage", "side", "version", "copy", "new", "one", "two", "three"}
        out = []
        for m in re.finditer(
                r"\b(?:one|two|three|\d+|a|an)\b((?:\s+[A-Za-z]+){1,2})", prompt):
            words = m.group(1).split()
            phrase = " ".join(words)
            head = next((w.lower() for w in words if w.lower() not in ADJ), None)
            if not head or head in STOP:
                continue
            stem = lambda s: s[:-1] if s.endswith("s") and not s.endswith("ss") else s
            if not any(stem(head) == stem(n) or stem(head) in stem(n)
                       or stem(n) in stem(head) for n in self.items):
                out.append(phrase)
        return out

    def save(self, name, path):
        self.items[name].image.save(path)
