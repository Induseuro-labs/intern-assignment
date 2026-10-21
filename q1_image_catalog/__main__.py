"""Command-line interface.

  python -m q1_image_catalog --root data/catalog seed            # load the section-6 test images
  python -m q1_image_catalog --root data/catalog list
  python -m q1_image_catalog --root data/catalog show apple
  python -m q1_image_catalog --root data/catalog add path.png --name dog
  python -m q1_image_catalog --root data/catalog add-object faint --label circle --bbox 100 100 201 201 --color white
  python -m q1_image_catalog --root data/catalog edit-object <id> --label boy
  python -m q1_image_catalog --root data/catalog delete-object <id>
  python -m q1_image_catalog --root data/catalog detect boy [--force] [--keep-manual]
  python -m q1_image_catalog --root data/catalog compose "one red apple and one orange"
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile

from .catalog import Catalog, CatalogError
from .compose import compose
from .detect import ClassicalDetector, ClaudeVisionDetector
from .testdata import DESCRIPTIONS, write_images


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="q1_image_catalog")
    p.add_argument("--root", default="data/catalog")
    p.add_argument("--detector", choices=["auto", "classical", "claude"], default="auto")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed")
    sub.add_parser("list")
    s = sub.add_parser("show"); s.add_argument("image")
    s = sub.add_parser("add"); s.add_argument("path"); s.add_argument("--name", required=True)
    s.add_argument("--description", default="")
    s = sub.add_parser("detect"); s.add_argument("image")
    s.add_argument("--force", action="store_true"); s.add_argument("--keep-manual", action="store_true")
    s = sub.add_parser("add-object"); s.add_argument("image"); s.add_argument("--label", required=True)
    s.add_argument("--bbox", nargs=4, type=int, required=True, metavar=("X", "Y", "W", "H"))
    s.add_argument("--color"); s.add_argument("--confidence", type=float, default=1.0)
    s = sub.add_parser("edit-object"); s.add_argument("object_id")
    s.add_argument("--label"); s.add_argument("--display-name"); s.add_argument("--color")
    s.add_argument("--confidence", type=float)
    s.add_argument("--bbox", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    s = sub.add_parser("delete-object"); s.add_argument("object_id")
    s = sub.add_parser("compose"); s.add_argument("request"); s.add_argument("--name")
    a = p.parse_args(argv)

    detector = {"classical": ClassicalDetector, "claude": ClaudeVisionDetector}.get(a.detector, lambda: None)()
    cat = Catalog(a.root, detector)
    try:
        if a.cmd == "seed":
            for name, path in write_images(tempfile.mkdtemp()).items():
                img = cat.add_image(path, name, DESCRIPTIONS[name])
                print(f"{name:20s} {img['auto']['detection']['status']:10s} "
                      f"{[o['display_name'] for o in img['objects']]}")
        elif a.cmd == "list":
            for img in cat.list():
                print(f"{img['id']}  {img['name']:28s} {img['auto']['kind']:9s} "
                      f"{img['auto']['detection']['status']:10s} objects={len(img['objects'])}"
                      f"{'  [corrected]' if img['corrected'] else ''}")
        elif a.cmd == "show":
            print(json.dumps(cat.get(a.image), indent=2))
        elif a.cmd == "add":
            print(json.dumps(cat.add_image(a.path, a.name, a.description), indent=2))
        elif a.cmd == "detect":
            print(json.dumps(cat.run_detection(a.image, a.force, a.keep_manual), indent=2))
        elif a.cmd == "add-object":
            print(json.dumps(cat.add_object(a.image, a.label, a.bbox, a.color, a.confidence), indent=2))
        elif a.cmd == "edit-object":
            fields = {k: v for k, v in {"label": a.label, "display_name": a.display_name, "color": a.color,
                                        "confidence": a.confidence, "bbox": a.bbox}.items() if v is not None}
            print(json.dumps(cat.update_object(a.object_id, **fields), indent=2))
        elif a.cmd == "delete-object":
            cat.delete_object(a.object_id); print("deleted")
        elif a.cmd == "compose":
            img = compose(cat, a.request, a.name)
            print(json.dumps(img, indent=2))
    except CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        cat.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
