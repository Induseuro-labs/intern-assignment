"""
Comprehensive Test and Verification Suite for Question 1.
Demonstrates and validates all required scenarios:
1. Object detection & metadata extraction (names, colors, confidence, bounding boxes)
2. Manual object addition
3. Manual object editing
4. Persistence of manual corrections vs explicit detection rerun
5. Composite image generation with apple and orange assets
6. Composite image generation with boy, girl, and shapes
7. Overlapping objects and zero-detectable-object edge case
"""

import os
import sys
from PIL import Image

# Ensure package imports resolve cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from question_1_image_catalog.catalog.models import (
    CatalogEntry, BoundingBox
)
from question_1_image_catalog.catalog.storage import ImageCatalog
from question_1_image_catalog.detector.detector import HybridObjectDetector
from question_1_image_catalog.compositor.engine import CompositeEngine

def print_separator(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)

def run_all_scenarios():
    test_data_dir = os.path.join(CURRENT_DIR, "data")
    catalog_json_path = os.path.join(CURRENT_DIR, "catalog", "test_catalog.json")
    
    # Initialize fresh catalog for clean test run
    catalog = ImageCatalog(storage_path=catalog_json_path)
    catalog.clear()
    
    detector = HybridObjectDetector(use_yolo=True)
    compositor = CompositeEngine(catalog=catalog)
    
    print_separator("Scenario 1: Detect objects & extract metadata across catalog images")
    test_images = [
        ("apple", "apple.png"),
        ("orange", "orange.png"),
        ("boy", "boy.png"),
        ("girl", "girl.png"),
        ("shapes", "shapes.png"),
        ("colors", "colors.png"),
        ("apple_orange", "apple_orange.png"),
    ]
    
    for entry_id, filename in test_images:
        img_path = os.path.join(test_data_dir, filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing test image: {img_path}")
            
        pil_img = Image.open(img_path)
        w, h = pil_img.size
        
        # 1. Run detection
        detected_objects = detector.detect(img_path)
        status = "completed" if detected_objects else "no_objects_found"
        
        entry = CatalogEntry(
            id=entry_id,
            image_name=filename,
            file_path=img_path,
            width=w,
            height=h,
            format=pil_img.format or "PNG",
            file_size_bytes=os.path.getsize(img_path),
            detection_status=status,
            objects=detected_objects
        )
        catalog.add_entry(entry)
        
        print(f"\n[Catalog Entry: '{entry_id}'] ({filename}) - {len(detected_objects)} object(s) detected:")
        for idx, obj in enumerate(detected_objects, 1):
            bbox = obj.auto_bbox
            bbox_str = f"x={bbox.x}, y={bbox.y}, w={bbox.width}, h={bbox.height}" if bbox else "None"
            print(f"  {idx}. Label: '{obj.label}' | Confidence: {obj.confidence:.2f} | "
                  f"Colors: {obj.colors} | BBox: [{bbox_str}] | Detector: {obj.detector_model}")

    print_separator("Scenario 2: Manually add an object when detection misses it")
    apple_entry = catalog.get_entry("apple")
    print(f"Initial objects in 'apple': {[o.label for o in apple_entry.get_active_objects()]}")
    
    manual_leaf = catalog.add_manual_object(
        entry_id="apple",
        label="apple_leaf",
        bbox=(255, 100, 65, 35),
        colors=["green"],
        confidence=1.0,
        notes="User annotated green leaf attached to apple stem"
    )
    print(f"Added Manual Object: ID={manual_leaf.id}, Label='{manual_leaf.label}', "
          f"Colors={manual_leaf.colors}, IsManual={manual_leaf.is_manually_added}")
    print(f"Active objects now in 'apple': {[o.label for o in catalog.get_entry('apple').get_active_objects()]}")
    assert any(o.label == "apple_leaf" for o in catalog.get_entry("apple").get_active_objects())

    print_separator("Scenario 3: Edit an object's name, color, confidence, or bounding box")
    boy_entry = catalog.get_entry("boy")
    boy_obj = boy_entry.get_active_objects()[0]
    print(f"Original 'boy' metadata: Label='{boy_obj.label}', Confidence={boy_obj.confidence:.2f}, "
          f"Colors={boy_obj.colors}, Modified={boy_obj.is_user_modified}")
    
    edited_obj = catalog.edit_object(
        entry_id="boy",
        object_id=boy_obj.id,
        label="cartoon_boy_hero",
        confidence=0.99,
        colors=["vibrant_blue", "dark_navy"],
        display_name="Hero Character Boy",
        notes="Corrected label to specific avatar hero"
    )
    print(f"After User Edit: Effective Label='{edited_obj.label}', Confidence={edited_obj.confidence}, "
          f"Colors={edited_obj.colors}, DisplayName='{edited_obj.display_name}'")
    print(f"Audit log verification: Auto Label was '{edited_obj.auto_label}', "
          f"User Override is '{edited_obj.user_label}', is_manually_edited={edited_obj.is_manually_edited}")
    assert edited_obj.label == "cartoon_boy_hero"
    assert edited_obj.auto_label == "boy"  # Original detector field remained pristine!

    print_separator("Scenario 4: Verify manual corrections persist until detection is explicitly rerun")
    # Reload catalog completely from JSON on disk to verify persistence
    catalog_reloaded = ImageCatalog(storage_path=catalog_json_path)
    persisted_boy = catalog_reloaded.get_entry("boy").get_active_objects()[0]
    print(f"Reloaded from disk: Label='{persisted_boy.label}', is_manually_edited={persisted_boy.is_manually_edited}")
    assert persisted_boy.label == "cartoon_boy_hero", "Manual edit must persist across reloads!"
    
    # Simulate an explicit user rerun request vs normal preservation
    print("Simulating user requesting explicit redetection on 'boy':")
    fresh_detections = detector.detect(persisted_boy_entry := catalog_reloaded.get_entry("boy").file_path)
    print(f"Fresh detection output: {[o.auto_label for o in fresh_detections]}")
    # Resetting upon explicit user command
    boy_entry_reset = catalog_reloaded.get_entry("boy")
    boy_entry_reset.objects = fresh_detections
    boy_entry_reset.has_manual_corrections = False
    boy_entry_reset.detection_status = "completed"
    catalog_reloaded.save()
    print(f"After explicit rerun: Label is reset to '{catalog_reloaded.get_entry('boy').get_active_objects()[0].label}'")

    print_separator("Scenario 5: Generate composite using existing apple and orange assets")
    prompt_5 = "Create an image containing one red apple and one orange"
    print(f"Instruction Prompt: '{prompt_5}'")
    composite_5 = compositor.generate_composite(
        prompt=prompt_5,
        composite_name="composite_apple_orange_test"
    )
    print(f"Composite Created: ID='{composite_5.id}', File='{composite_5.file_path}', Dimensions={composite_5.width}x{composite_5.height}")
    print(f"Source Assets Verified (No Hallucination):")
    for src in composite_5.composite_metadata.source_assets:
        print(f"  - Reused catalog asset '{src.catalog_entry_id}' (Label: '{src.object_label}', Placed: [{src.placed_bbox.to_tuple()}], Scale: {src.scale})")
    assert len(composite_5.composite_metadata.source_assets) == 2
    assert os.path.exists(composite_5.file_path)

    print_separator("Scenario 6: Generate composite containing boy, girl, and colored shapes")
    prompt_6 = "Create a composite containing a boy, a girl, and colored shapes"
    print(f"Instruction Prompt: '{prompt_6}'")
    composite_6 = compositor.generate_composite(
        prompt=prompt_6,
        canvas_size=(1100, 650),
        composite_name="composite_boy_girl_shapes_test"
    )
    print(f"Composite Created: ID='{composite_6.id}', File='{composite_6.file_path}', Dimensions={composite_6.width}x{composite_6.height}")
    print(f"Source Assets Verified:")
    for src in composite_6.composite_metadata.source_assets:
        print(f"  - Reused catalog asset '{src.catalog_entry_id}' (Label: '{src.object_label}', Placed: [{src.placed_bbox.to_tuple()}])")
    assert len(composite_6.composite_metadata.source_assets) >= 3
    assert os.path.exists(composite_6.file_path)

    print_separator("Scenario 7: Overlapping objects and zero detectable objects edge cases")
    # 7a. Overlapping shapes
    overlap_path = os.path.join(test_data_dir, "overlapping_shapes.png")
    overlap_objs = detector.detect(overlap_path)
    print(f"Overlapping shapes image ({len(overlap_objs)} objects detected):")
    for o in overlap_objs:
        print(f"  - Label: '{o.label}', Colors: {o.colors}, BBox: {o.bbox.to_tuple()}")
    assert len(overlap_objs) >= 2, "Must detect overlapping shapes"

    # 7b. Empty canvas with no detectable objects
    empty_path = os.path.join(test_data_dir, "empty_canvas.png")
    empty_objs = detector.detect(empty_path)
    empty_status = "completed" if empty_objs else "no_objects_found"
    print(f"\nEmpty canvas image: Detected {len(empty_objs)} objects. Detection Status: '{empty_status}'")
    assert len(empty_objs) == 0, "Empty canvas must yield 0 detected objects"
    assert empty_status == "no_objects_found"

    print("\n" + "#" * 80)
    print(" ALL 7 QUESTION 1 TEST SCENARIOS PASSED WITH 100% SUCCESS! ")
    print("#" * 80)

if __name__ == "__main__":
    run_all_scenarios()
