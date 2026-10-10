import numpy as np
import pytest
from PIL import Image

from conftest import IMAGES
from imgcat.composite import MissingAssetError, RulePlanner
from imgcat.service import ProtectedObjectsError


@pytest.mark.parametrize("text, expected", [
    ("Create an image containing one red apple and one orange",
     [("apple", 1, "red", False), ("orange", 1, None, False)]),
    ("Create an image containing a boy, a girl, and colored shapes",
     [("boy", 1, None, False), ("girl", 1, None, False), ("shape", 0, None, True)]),
    ("two apples next to an orange", [("apple", 2, None, False), ("orange", 1, None, False)]),
])
def test_rule_planner(text, expected):
    plan = RulePlanner().plan(text)
    assert [(i.noun, i.count, i.color, i.is_category) for i in plan.items] == expected


def test_rule_planner_layout_and_background():
    plan = RulePlanner().plan("overlapping red circle and blue square on a yellow background")
    assert plan.layout == "overlap" and plan.background == "yellow"


@pytest.fixture
def shapes_catalog(svc):
    svc.add_image(IMAGES / "shapes.png", "shapes")
    svc.add_image(IMAGES / "overlapping_shapes.png", "overlapping_shapes")
    return svc


def test_composite_uses_existing_pixels_and_is_saved(shapes_catalog):
    svc = shapes_catalog
    comp = svc.compose("a red circle and a blue square", use_llm=False)
    assert comp.kind == "composite" and comp.objects_source == "composition"
    prov = comp.to_dict()["auto"]["provenance"]
    assert prov["generative_fill"] is False
    assert [p["source_image"] for p in prov["placements"]] == ["shapes", "shapes"]  # unoccluded copies preferred
    assert [o.effective["name"] for o in comp.objects] == ["circle", "square"]

    rgb = np.array(Image.open(svc.store.image_path(comp.id)).convert("RGB"))
    for o, expect in zip(comp.objects, [(220, 40, 40), (40, 90, 220)]):
        b = o.effective["bbox"]
        centre = rgb[b["y"] + b["height"] // 2, b["x"] + b["width"] // 2]
        assert tuple(int(v) for v in centre) == expect  # exact source colour, not a re-render
    with pytest.raises(ProtectedObjectsError):
        svc.run_detection(comp.id)


def test_category_request_takes_all_distinct_shapes(shapes_catalog):
    comp = shapes_catalog.compose("colored shapes", use_llm=False)
    assert sorted(o.effective["name"] for o in comp.objects) == ["circle", "rectangle", "square", "triangle"]


def test_missing_asset_is_an_error_not_a_substitute(shapes_catalog):
    before = len(shapes_catalog.store.list_images())
    with pytest.raises(MissingAssetError) as exc:
        shapes_catalog.compose("a red circle and a banana", use_llm=False)
    assert exc.value.missing == ["banana"]
    assert len(shapes_catalog.store.list_images()) == before


def test_user_corrections_feed_composites(svc):
    img = svc.add_image(IMAGES / "camouflage.png", "camouflage")
    svc.add_object(img.id, "square", {"x": 320, "y": 220, "width": 160, "height": 160}, color="gray")
    comp = svc.compose("a gray square", use_llm=False)
    p = comp.to_dict()["auto"]["provenance"]["placements"][0]
    assert p["source_image"] == "camouflage" and p["mask_source"] != "stored detection mask"


def test_overlap_layout_layers(shapes_catalog):
    comp = shapes_catalog.compose("overlapping red circle, blue square and green triangle", use_llm=False)
    boxes = [o.effective["bbox"] for o in comp.objects]
    zs = [o.effective["attributes"]["z_index"] for o in comp.objects]
    assert zs == [0, 1, 2]
    assert boxes[1]["x"] < boxes[0]["x"] + boxes[0]["width"]  # they overlap
