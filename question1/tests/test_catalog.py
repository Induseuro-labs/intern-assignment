import json

import pytest

from conftest import IMAGES
from imgcat.detection import Detector
from imgcat.service import CatalogService, ProtectedObjectsError


def test_add_image_runs_detection_and_separates_auto_from_user(svc):
    img = svc.add_image(IMAGES / "shapes.png", "shapes")
    d = img.to_dict()
    assert d["auto"]["detection_status"] == "detected"
    assert d["auto"]["width"] == 800 and d["auto"]["height"] == 600
    assert set(d["user"]) == {"display_name", "description", "tags"}
    o = d["objects"][0]
    assert o["origin"] == "detector" and o["user"] == {} and o["auto"]["confidence"] > 0.9
    assert o["effective"]["bbox"] == o["auto"]["bbox"]


def test_manual_add_when_detection_misses(svc):
    img = svc.add_image(IMAGES / "camouflage.png", "camouflage")
    assert img.detection_status == "no_objects"
    obj = svc.add_object(img.id, "square", {"x": 320, "y": 220, "width": 160, "height": 160}, color="gray")
    assert obj.origin == "user" and obj.auto is None
    assert obj.effective["name"] == "square" and obj.effective["category"] == "shape"
    img = svc.get(img.id)
    assert img.objects_source == "manual" and len(img.live_objects) == 1


def test_edit_name_colour_confidence_bbox_keeps_detector_value(svc):
    img = svc.add_image(IMAGES / "shapes.png", "shapes")
    circle = next(o for o in img.objects if o.effective["name"] == "circle")
    auto_before = dict(circle.auto)
    new_box = {"x": 55, "y": 75, "width": 210, "height": 210}
    obj = svc.update_object(circle.id, name="ball", color="crimson", confidence=0.5, bbox=new_box)
    assert obj.auto == auto_before                      # detector output untouched
    assert obj.effective["name"] == "ball" and obj.effective["display_name"] == "ball"
    assert obj.effective["color"] == "crimson" and obj.effective["confidence"] == 0.5
    assert obj.effective["bbox"] == new_box
    assert obj.edited_fields == ["bbox", "color", "confidence", "name"]  # "ball" is no known label: category kept
    reverted = svc.revert_field(obj.id, "color")
    assert reverted.effective["color"] == "red"


@pytest.mark.parametrize("fields", [
    {"bbox": {"x": 700, "y": 0, "width": 200, "height": 10}},   # outside image
    {"confidence": 1.5},
    {"detector": "me"},                                          # auto-only field
    {"name": "  "},
])
def test_invalid_edits_rejected(svc, fields):
    img = svc.add_image(IMAGES / "shapes.png", "shapes")
    with pytest.raises(ValueError):
        svc.update_object(img.objects[0].id, **fields)


def test_corrections_are_source_of_truth_until_forced(svc, tmp_path):
    img = svc.add_image(IMAGES / "colors.png", "colors")
    purple = next(o for o in img.objects if o.effective["color"] == "purple")
    red = next(o for o in img.objects if o.effective["color"] == "red")
    svc.delete_object(purple.id)
    svc.update_object(red.id, display_name="brand red")

    with pytest.raises(ProtectedObjectsError):
        svc.run_detection(img.id)

    # restart: a new service on the same files still sees the corrections
    svc2 = CatalogService(svc.store.root, detector=Detector(use_owl=False))
    img2 = svc2.get(img.id)
    assert img2.objects_source == "manual"
    assert len(img2.live_objects) == 4
    assert any(o.effective["display_name"] == "brand red" for o in img2.live_objects)

    forced = svc2.run_detection(img.id, force=True)
    assert forced.objects_source == "auto" and len(forced.live_objects) == 5
    runs = svc2.store.runs(img.id)
    assert [r["forced"] for r in runs] == [0, 1]
    replaced = json.loads(runs[-1]["replaced"])
    assert any(o["user"].get("display_name") == "brand red" for o in replaced)  # edits archived


def test_deleted_detector_object_is_tombstoned(svc):
    img = svc.add_image(IMAGES / "shapes.png", "shapes")
    oid = img.objects[0].id
    svc.delete_object(oid)
    assert svc.store.get_object(oid).deleted
    assert len(svc.get(img.id).live_objects) == 3
    svc.restore_object(oid)
    assert len(svc.get(img.id).live_objects) == 4


def test_failed_detection_reports_and_keeps_objects(svc):
    bad = svc.add_image(IMAGES / "truncated.png", "truncated")
    assert bad.detection_status == "failed" and "truncated" in bad.detection_error

    img = svc.add_image(IMAGES / "shapes.png", "shapes")
    svc.add_object(img.id, "note", {"x": 1, "y": 1, "width": 10, "height": 10})

    class Broken(Detector):
        def detect(self, rgb):
            raise RuntimeError("model crashed")

    svc._detector = Broken(use_owl=False)
    after = svc.run_detection(img.id, force=True)
    assert after.detection_status == "failed"
    assert len(after.live_objects) == 5  # nothing lost
