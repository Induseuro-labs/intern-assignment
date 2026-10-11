"""
Question 1 CLI Entry Point: Image Catalog, Object Detection & Composite Engine.
Usage:
    python -m question_1_image_catalog list
    python -m question_1_image_catalog add <image_path> [--name <name>]
    python -m question_1_image_catalog detect <entry_id_or_path> [--force]
    python -m question_1_image_catalog edit <entry_id> <obj_id> [--label <new_label>]
    python -m question_1_image_catalog compose "<prompt>" [--verify] [--name <comp_name>]
    python -m question_1_image_catalog report
    python -m question_1_image_catalog serve
"""

from __future__ import annotations
import os
import sys
import argparse
import subprocess

from .catalog.storage import ImageCatalog
from .catalog.models import ProtectedObjectsError
from .detector.detector import HybridObjectDetector
from .compositor.engine import CompositeEngine

def get_catalog():
    catalog = ImageCatalog()
    if len(catalog.list_entries()) == 0:
        # Auto-seed standard test dataset
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        detector = HybridObjectDetector(use_yolo=False)
        seed_files = [
            ("apple", "apple.png"),
            ("orange", "orange.png"),
            ("boy", "boy.png"),
            ("girl", "girl.png"),
            ("shapes", "shapes.png"),
            ("colors", "colors.png"),
            ("apple_orange", "apple_orange.png"),
            ("overlapping_shapes", "overlapping_shapes.png"),
            ("empty_canvas", "empty_canvas.png")
        ]
        from PIL import Image
        from .catalog.models import CatalogEntry
        for slug, fname in seed_files:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                objs = detector.detect(fpath)
                im = Image.open(fpath)
                catalog.add_entry(CatalogEntry(
                    id=slug,
                    image_name=fname,
                    file_path=fpath,
                    width=im.width,
                    height=im.height,
                    format=im.format or "PNG",
                    file_size_bytes=os.path.getsize(fpath),
                    detection_status="completed" if objs else "no_objects_found",
                    objects_source="auto",
                    objects=objs
                ))
    return catalog

def cmd_list(args):
    catalog = get_catalog()
    entries = catalog.list_entries()
    print(f"\n==================== IMAGE CATALOG ({len(entries)} entries) ====================")
    print(f"{'ID':<15} | {'Image Name':<20} | {'Status':<16} | {'Source':<10} | {'Active Objs'}")
    print("-" * 75)
    for e in entries:
        active = e.get_active_objects()
        labels = ", ".join([o.label for o in active])
        print(f"{e.id:<15} | {e.image_name:<20} | {e.detection_status:<16} | {e.objects_source:<10} | {len(active)} [{labels}]")
    print("=" * 75)

def cmd_add(args):
    catalog = get_catalog()
    img_path = os.path.abspath(args.path)
    if not os.path.exists(img_path):
        print(f"Error: File not found at '{img_path}'")
        return
    name = args.name or os.path.splitext(os.path.basename(img_path))[0]
    
    detector = HybridObjectDetector(use_yolo=True)
    detected = detector.detect(img_path)
    
    from PIL import Image
    im = Image.open(img_path)
    from .catalog.models import CatalogEntry
    entry = CatalogEntry(
        id=name,
        image_name=os.path.basename(img_path),
        file_path=img_path,
        width=im.width,
        height=im.height,
        format=im.format or "PNG",
        file_size_bytes=os.path.getsize(img_path),
        detection_status="completed" if detected else "no_objects_found",
        objects_source="auto",
        objects=detected
    )
    catalog.add_entry(entry)
    print(f"Added '{name}' to catalog. Detected {len(detected)} object(s):")
    for o in detected:
        print(f"  - [{o.label}] conf={o.confidence} colors={o.colors} bbox={o.bbox.to_tuple() if o.bbox else None}")

def cmd_detect(args):
    catalog = get_catalog()
    entry = catalog.get_entry(args.target)
    if not entry:
        print(f"Error: Entry '{args.target}' not found in catalog.")
        return
    detector = HybridObjectDetector(use_yolo=True)
    detected = detector.detect(entry.file_path)
    try:
        catalog.update_detection(entry.id, detected, force=args.force)
        print(f"Detection succeeded on '{entry.id}'. Found {len(detected)} object(s).")
    except ProtectedObjectsError as e:
        print(f"\n[PROTECTION ALERT] {e}")
        print("To overwrite manual corrections, run with --force flag.")

def cmd_edit(args):
    catalog = get_catalog()
    res = catalog.edit_object(args.entry, args.obj_id, label=args.label, confidence=args.conf)
    if res:
        print(f"Successfully edited object '{args.obj_id}' in '{args.entry}'.")
        print(f"Effective label is now: '{res.label}', source_of_truth='manual'")
    else:
        print(f"Error: Object '{args.obj_id}' or entry '{args.entry}' not found.")

def cmd_compose(args):
    catalog = get_catalog()
    engine = CompositeEngine(catalog)
    try:
        print(f"\nParsing prompt: '{args.prompt}'...")
        comp = engine.generate_composite(args.prompt, composite_name=args.name, verify=args.verify)
        print(f"Composite generated successfully: {comp.file_path}")
        print(f"Dimensions: {comp.width}x{comp.height}")
        print(f"Objects placed ({len(comp.objects)}):")
        for o in comp.objects:
            print(f"  - [{o.label}] bbox={o.bbox.to_tuple() if o.bbox else None}")
        if comp.composite_metadata:
            print(f"Provenance verification: {comp.composite_metadata.verification_status}")
            if comp.composite_metadata.verification_iou:
                print(f"Average IoU alignment: {comp.composite_metadata.verification_iou}")
    except Exception as e:
        print(f"Composite generation failed: {e}")

def cmd_report(args):
    print("Running scenarios and compiling comprehensive REPORT.md...")
    script_path = os.path.join(os.path.dirname(__file__), "generate_report.py")
    subprocess.run([sys.executable, script_path], check=True)

def cmd_serve(args):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(root_dir, "app", "streamlit_app.py")
    print(f"Launching Streamlit UI from {app_path}...")
    subprocess.run(["streamlit", "run", app_path])

def main():
    parser = argparse.ArgumentParser(description="Image Catalog & Object Detection Service")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    subparsers.add_parser("list", help="List all catalog entries")

    # add
    p_add = subparsers.add_parser("add", help="Add image to catalog with auto-detection")
    p_add.add_argument("path", help="Path to image file")
    p_add.add_argument("--name", help="Custom catalog entry name")

    # detect
    p_det = subparsers.add_parser("detect", help="Run detection on a catalog entry")
    p_det.add_argument("target", help="Catalog entry ID")
    p_det.add_argument("--force", action="store_true", help="Force overwrite manual corrections")

    # edit
    p_edit = subparsers.add_parser("edit", help="Edit object metadata manually")
    p_edit.add_argument("entry", help="Catalog entry ID")
    p_edit.add_argument("obj_id", help="Object ID")
    p_edit.add_argument("--label", help="New label")
    p_edit.add_argument("--conf", type=float, help="New confidence score")

    # compose
    p_comp = subparsers.add_parser("compose", help="Generate composite from natural language prompt")
    p_comp.add_argument("prompt", help="Natural language prompt")
    p_comp.add_argument("--name", help="Name for the composite entry")
    p_comp.add_argument("--verify", action="store_true", help="Verify composite with detector pass")

    # report
    subparsers.add_parser("report", help="Generate comprehensive markdown report of all test scenarios")

    # serve
    subparsers.add_parser("serve", help="Launch interactive web browser UI")

    args = parser.parse_args()
    cmds = {
        "list": cmd_list,
        "add": cmd_add,
        "detect": cmd_detect,
        "edit": cmd_edit,
        "compose": cmd_compose,
        "report": cmd_report,
        "serve": cmd_serve,
    }
    cmds[args.command](args)

if __name__ == "__main__":
    main()
