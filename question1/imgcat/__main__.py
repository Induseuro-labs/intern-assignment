"""Command-line interface.

    python -m imgcat add test_images/apple.png --name apple
    python -m imgcat list
    python -m imgcat show apple
    python -m imgcat detect apple [--force]
    python -m imgcat add-object camouflage square 320 220 160 160 --color gray
    python -m imgcat edit-object <object_id> --name mandarin --bbox 530 130 310 320
    python -m imgcat delete-object <object_id>
    python -m imgcat compose "Create an image containing one red apple and one orange" [--verify]
    python -m imgcat serve [--port 8000]

The catalog lives in $IMGCAT_ROOT (default runtime/catalog).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .composite import MissingAssetError
from .service import CatalogService, ProtectedObjectsError


def _print_image(img) -> None:
    d = img.to_dict()
    a = d["auto"]
    print(f"{img.id}  {d['user']['display_name']}  [{a['kind']}] {a['width']}x{a['height']} "
          f"status={a['detection_status']} objects_source={a['objects_source']} style={a['style']}")
    for o in d["objects"]:
        e = o["effective"]
        b = e["bbox"]
        conf = f"{e['confidence']:.2f}" if e.get("confidence") is not None else "-"
        edited = f" edited={','.join(o['edited_fields'])}" if o["edited_fields"] else ""
        print(f"   {o['id']}  {e['display_name']:<20} name={e['name']:<14} color={e.get('color') or '-':<7} "
              f"conf={conf} bbox=({b['x']},{b['y']},{b['width']},{b['height']}) origin={o['origin']}{edited}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="imgcat")
    ap.add_argument("--root", default=os.environ.get("IMGCAT_ROOT", "runtime/catalog"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add"); p.add_argument("path"); p.add_argument("--name"); p.add_argument("--no-detect", action="store_true")
    sub.add_parser("list")
    p = sub.add_parser("show"); p.add_argument("image"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("detect"); p.add_argument("image"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("add-object"); p.add_argument("image"); p.add_argument("name")
    p.add_argument("bbox", nargs=4, type=int, metavar=("X", "Y", "W", "H")); p.add_argument("--color")
    p.add_argument("--confidence", type=float, default=1.0)
    p = sub.add_parser("edit-object"); p.add_argument("object_id"); p.add_argument("--name"); p.add_argument("--display-name")
    p.add_argument("--color"); p.add_argument("--confidence", type=float)
    p.add_argument("--bbox", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    p = sub.add_parser("delete-object"); p.add_argument("object_id")
    p = sub.add_parser("compose"); p.add_argument("instruction"); p.add_argument("--name")
    p.add_argument("--verify", action="store_true"); p.add_argument("--no-llm", action="store_true")
    p = sub.add_parser("serve"); p.add_argument("--port", type=int, default=8000); p.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)

    svc = CatalogService(args.root)
    try:
        if args.cmd == "add":
            _print_image(svc.add_image(args.path, args.name, detect=not args.no_detect))
        elif args.cmd == "list":
            for img in svc.store.list_images():
                print(f"{img.id}  {img.display_name:<22} {img.kind:<9} {img.detection_status:<10} "
                      f"{img.objects_source:<11} {len(img.live_objects)} objects")
        elif args.cmd == "show":
            img = svc.get(args.image)
            print(json.dumps(img.to_dict(include_deleted=True), indent=2)) if args.json else _print_image(img)
        elif args.cmd == "detect":
            _print_image(svc.run_detection(svc.get(args.image).id, force=args.force))
        elif args.cmd == "add-object":
            x, y, w, h = args.bbox
            img = svc.get(args.image)
            svc.add_object(img.id, args.name, {"x": x, "y": y, "width": w, "height": h},
                           color=args.color, confidence=args.confidence)
            _print_image(svc.get(img.id))
        elif args.cmd == "edit-object":
            fields = {k: v for k, v in {"name": args.name, "display_name": args.display_name, "color": args.color,
                                        "confidence": args.confidence}.items() if v is not None}
            if args.bbox:
                fields["bbox"] = dict(zip(("x", "y", "width", "height"), args.bbox))
            obj = svc.update_object(args.object_id, **fields)
            _print_image(svc.get(obj.image_id))
        elif args.cmd == "delete-object":
            obj = svc.store.get_object(args.object_id)
            svc.delete_object(args.object_id)
            _print_image(svc.get(obj.image_id))
        elif args.cmd == "compose":
            img = svc.compose(args.instruction, args.name, use_llm=False if args.no_llm else None, verify=args.verify)
            _print_image(img)
            print("file:", svc.store.image_path(img.id))
        elif args.cmd == "serve":
            import uvicorn

            from .api import create_app

            uvicorn.run(create_app(service=svc), host=args.host, port=args.port)
    except ProtectedObjectsError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except MissingAssetError as exc:
        print(f"missing assets: {exc}", file=sys.stderr)
        return 3
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
