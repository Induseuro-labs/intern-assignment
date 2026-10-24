"""Q1 runnable check: all 7 test scenarios from the brief + visual proof.

Run:  python demo.py   (from this folder, after pip install -r requirements.txt)
Writes annotated detections + composites into outputs/.
"""
import os

import numpy as np
from PIL import ImageDraw

from catalog import Catalog
from data import make_test_images
from detector import PALETTE


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    iw, ih = min(ax + aw, bx + bw) - ix, min(ay + ah, by + bh) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


def annotate(entry):
    """Copy of the image with every detected object's bbox + label drawn."""
    im = entry.image.copy()
    d = ImageDraw.Draw(im)
    for o in entry.objects:
        x, y, w, h = o.bbox
        d.rectangle([x, y, x + w, y + h], outline="red", width=3)
        d.text((x + 2, max(0, y - 12)), f"{o.name} {o.confidence}", fill="red")
    return im


def demo():
    """One runnable check covering all 7 test scenarios in the brief."""
    cat, imgs = Catalog(), make_test_images()
    for n, im in imgs.items():
        if n != "blank":
            cat.add_image(n, im)
    # 1. successful detection: names, colors, confidences, bboxes
    assert len(cat.items["apple"].objects) == 1
    assert cat.items["apple"].objects[0].color == "red"
    assert len(cat.items["shapes"].objects) == 4, len(cat.items["shapes"].objects)
    assert len(cat.items["colors"].objects) == 5
    assert len(cat.items["apple_orange"].objects) == 2
    assert len(cat.items["boy"].objects) == 1 and "blue" in cat.items["boy"].objects[0].colors
    assert len(cat.items["girl"].objects) == 1
    ov = cat.items["overlapping_shapes"].objects
    assert len(ov) >= 2 and any(_iou(a.bbox, b.bbox) > 0 for a in ov for b in ov if a is not b)
    print("1. detection ok:", {n: [(o.color, o.shape, o.bbox) for o in e.objects]
                               for n, e in cat.items.items()})
    # 2/3. failed detection (blank) -> manual add; then edit
    cat.add_image("blank", imgs["blank"])
    assert cat.items["blank"].objects == [] and cat.items["blank"].status == "no_objects"
    o = cat.add_object("blank", "apple", (140, 150, 121, 141), "red")
    cat.edit_object("blank", o.id, name="red apple", confidence=0.99,
                    bbox=(140, 150, 121, 141))
    assert cat.items["blank"].objects[0].name == "red apple"
    try:
        cat.edit_object("blank", o.id, shape="circle")  # auto-only -> refused
        raise AssertionError("auto field edit should fail")
    except ValueError:
        pass
    print("2/3. failed detection + manual add/edit ok")
    # 4. corrections persist until explicit rerun
    before = (cat.items["blank"].objects[0].name, cat.items["blank"].objects[0].bbox)
    cat.add_image("apple2", imgs["apple"])  # unrelated ingest changes nothing
    assert (cat.items["blank"].objects[0].name, cat.items["blank"].objects[0].bbox) == before
    assert cat.items["blank"].objects[0].corrected
    cat.rerun_detection("blank")  # explicit rerun discards the manual fix
    assert cat.items["blank"].objects == []
    del cat.items["apple2"]
    print("4. manual corrections persist until rerun ok")
    # 5. composite from existing assets, using the real pixels
    comp = cat.generate_composite(
        "Create an image containing one red apple and one orange", "comp1")
    assert comp.source["sources"] == ["apple", "orange"]
    px = np.asarray(comp.image.convert("RGB")).reshape(-1, 3)
    assert ((abs(px - PALETTE["red"]) < 30).all(axis=1)).any()
    assert ((abs(px - PALETTE["orange"]) < 30).all(axis=1)).any()
    try:  # missing assets are reported, never silently AI-generated
        cat.generate_composite("one apple and one dragon fruit", "bad")
        raise AssertionError("missing asset should fail")
    except ValueError as e:
        assert "dragon" in str(e)
    print("5. apple+orange composite ok:", comp.source["sources"])
    # 6. boy + girl + shapes composite
    comp2 = cat.generate_composite(
        "a scene with a boy, a girl and colored shapes", "comp2")
    assert comp2.source["sources"] == ["boy", "girl", "shapes"], comp2.source["sources"]
    assert len(comp2.objects) >= 3  # detection also runs on composites
    print("6. boy+girl+shapes composite ok:", comp2.source["sources"])
    # 7. overlapping + empty covered above (assertions on overlapping_shapes/blank)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out, exist_ok=True)
    for n, e in cat.items.items():
        annotate(e).save(os.path.join(out, f"det_{n}.png"))  # visual proof
    for n in ("comp1", "comp2"):
        cat.save(n, os.path.join(out, f"{n}.png"))
    print("7. overlapping + empty ok; proof saved in outputs/")


if __name__ == "__main__":
    demo()
