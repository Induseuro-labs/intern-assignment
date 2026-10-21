"""End-to-end walkthrough of every test scenario in question 1, section 6.

    python -m q1_image_catalog.demo            # offline, deterministic
    python -m q1_image_catalog.demo --claude   # use the Claude vision/parse backends

Writes annotated detections, composites, the catalog dump and a log to
examples/q1/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from PIL import ImageDraw

from .catalog import Catalog, ManualCorrectionsExist
from .compose import MissingAssetError, compose
from .detect import ClassicalDetector, ClaudeVisionDetector
from .testdata import DESCRIPTIONS, write_images

OUT = Path("examples/q1")
log_lines: list[str] = []


def log(text: str = "") -> None:
    print(text)
    log_lines.append(text)


def annotate(cat: Catalog, ref: str, path: Path) -> None:
    img = cat.load_pil(ref)
    d = ImageDraw.Draw(img)
    for o in cat.get(ref)["objects"]:
        b = o["bbox"]
        color = {"user": "magenta", "composer": "blue"}.get(o["origin"], "lime")
        if o["edited_by_user"]:
            color = "orange"
        d.rectangle((b["x"], b["y"], b["x"] + b["w"] - 1, b["y"] + b["h"] - 1), outline=color, width=3)
        d.text((b["x"] + 4, b["y"] + 4), f"{o['display_name']} {o['confidence']:.2f}", fill="black")
    img.save(path)


def objects_table(img: dict) -> None:
    det = img["auto"]["detection"]
    log(f"  status={det['status']} backend={det['backend']}" + (f" error={det['error']}" if det["error"] else ""))
    for o in img["objects"]:
        b = o["bbox"]
        log(f"  - {o['label']:10s} color={str(o['color']):7s} conf={o['confidence']:.2f} "
            f"bbox=({b['x']},{b['y']},{b['w']},{b['h']}) origin={o['origin']}"
            f"{' edited' if o['edited_by_user'] else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude", action="store_true", help="use Claude vision + LLM parsing")
    args = ap.parse_args()

    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "detections").mkdir(parents=True)
    root = Path(tempfile.mkdtemp()) / "catalog"
    cat = Catalog(root, ClaudeVisionDetector() if args.claude else ClassicalDetector())

    log("# Question 1 demo log\n")
    log("## Scenario 1 - detect objects (names, colours, confidence, bounding boxes)\n")
    log("```")
    for name, path in write_images(root.parent / "testdata").items():
        img = cat.add_image(path, name, DESCRIPTIONS[name], tags={"set": "section-6"})
        log(f"{name}: {DESCRIPTIONS[name]}")
        objects_table(img)
        annotate(cat, name, OUT / "detections" / f"{name}.png")
    log("```\n")

    log("## Scenario 2 - manually add an object the detector missed (`faint`)\n")
    log("```")
    obj = cat.add_object("faint", "circle", (100, 100, 201, 201), color="white", confidence=1.0)
    log(f"added {obj['label']} origin={obj['origin']} bbox={obj['bbox']}")
    obj = cat.add_object("cluttered", "apple", (110, 100, 181, 211), color="red")
    log(f"added {obj['label']} to 'cluttered' (detector failed on noisy background)")
    for n in ("faint", "cluttered"):
        objects_table(cat.get(n)); annotate(cat, n, OUT / "detections" / f"{n}_corrected.png")
    log("```\n")

    log("## Scenario 3 - edit name / colour / confidence / bbox\n")
    log("```")
    for name, label in (("boy", "boy"), ("girl", "girl")):
        o = cat.get(name)["objects"][0]
        edited = cat.update_object(o["id"], label=label, display_name=label, confidence=0.99)
        log(f"{name}: '{o['label']}' -> '{edited['label']}', auto_original kept: {edited['auto_original']}")
        annotate(cat, name, OUT / "detections" / f"{name}_corrected.png")
    o = cat.get("apple")["objects"][0]
    edited = cat.update_object(o["id"], bbox=(85, 75, 231, 261), color="dark red")
    log(f"apple: bbox {o['bbox']} -> {edited['bbox']}, color red -> {edited['color']}")
    log("```\n")

    log("## Scenario 4 - corrections persist until detection is explicitly rerun\n")
    log("```")
    try:
        cat.run_detection("boy")
    except ManualCorrectionsExist as exc:
        log(f"run_detection('boy') refused: {exc}")
    log(f"redetect_all(): {json.dumps(cat.redetect_all())}")
    log(f"boy label after batch reprocess: {cat.get('boy')['objects'][0]['label']}")
    tmp = cat.run_detection("apple", force=True, keep_manual=True)
    log(f"apple rerun with keep_manual=True -> objects: {[(o['label'], o['origin'], o['edited_by_user']) for o in tmp['objects']]}")
    log("```\n")

    log("## Scenario 5 - composite from the existing apple and orange assets\n")
    log("```")
    c1 = compose(cat, "Create an image containing one red apple and one orange", "apple_and_orange",
                 use_llm=args.claude)
    cat.load_pil(c1["id"]).save(OUT / "composite_apple_orange.png")
    log(json.dumps(c1["auto"]["provenance"], indent=2))
    log("```\n")

    log("## Scenario 6 - composite with a boy, a girl and coloured shapes\n")
    log("```")
    c2 = compose(cat, "Create an image with a boy, a girl and colored shapes", "kids_and_shapes",
                 use_llm=args.claude)
    cat.load_pil(c2["id"]).save(OUT / "composite_boy_girl_shapes.png")
    annotate(cat, c2["id"], OUT / "detections" / "kids_and_shapes.png")
    for s in c2["auto"]["provenance"]["sources"]:
        log(f"  {s['label']:10s} from {s['image_name']:8s} scale={s['scale']} placed={s['placed_bbox']} z={s['z']}")
    log("```\n")

    log("## Scenario 7 - overlapping objects, empty image, and a missing asset\n")
    log("```")
    objects_table(cat.get("overlapping_shapes"))
    objects_table(cat.get("blank"))
    try:
        compose(cat, "an apple and a banana", use_llm=args.claude)
    except MissingAssetError as exc:
        log(f"compose('an apple and a banana') refused: {exc}")
    log("```\n")

    (OUT / "catalog.json").write_text(json.dumps(cat.list(), indent=2))
    (OUT / "DEMO_LOG.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    cat.close()


if __name__ == "__main__":
    main()
