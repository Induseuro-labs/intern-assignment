"""Tests for every scenario in question 1, section 6 (offline, classical detector)."""
import numpy as np
import pytest

from q1_image_catalog.catalog import Catalog, CatalogError, ManualCorrectionsExist
from q1_image_catalog.compose import MissingAssetError, compose, parse_request
from q1_image_catalog.detect import ClassicalDetector, ClaudeVisionDetector
from q1_image_catalog.testdata import make_images


@pytest.fixture(scope="module")
def images():
    return make_images()


@pytest.fixture
def cat(tmp_path, images):
    c = Catalog(tmp_path / "cat", ClassicalDetector())
    for name, img in images.items():
        c.add_image(img, name)
    yield c
    c.close()


def labels(img):
    return sorted((o["label"], o["color"]) for o in img["objects"])


# --- scenario 1: detection -------------------------------------------------
@pytest.mark.parametrize("name, expected", [
    ("apple", [("apple", "red")]),
    ("orange", [("orange", "orange")]),
    ("apple_orange", [("apple", "red"), ("orange", "orange")]),
    ("shapes", [("circle", "red"), ("rectangle", "purple"), ("square", "blue"), ("triangle", "green")]),
    ("colors", [("square", c) for c in sorted(["red", "blue", "green", "yellow", "purple"])]),
    ("overlapping_shapes", [("circle", "red"), ("square", "blue"), ("triangle", "green")]),
])
def test_detection(cat, name, expected):
    img = cat.get(name)
    assert img["auto"]["detection"]["status"] == "detected"
    assert labels(img) == expected
    for o in img["objects"]:
        assert 0 < o["confidence"] <= 1
        b = o["bbox"]
        assert b["x"] + b["w"] <= img["auto"]["width"] and b["y"] + b["h"] <= img["auto"]["height"]


def test_bbox_is_tight(cat):
    (o,) = cat.get("orange")["objects"]
    assert o["bbox"] == {"x": 90, "y": 100, "w": 221, "h": 221}  # drawn as ellipse(90,100,310,320)


def test_cartoon_people_detected_with_clothing_colors(cat):
    for name in ("boy", "girl"):
        (o,) = cat.get(name)["objects"]
        assert o["label"] == "person" and "skin" not in o["attributes"]["colors"]
    assert "blue" in cat.get("boy")["objects"][0]["attributes"]["colors"]
    assert "pink" in cat.get("girl")["objects"][0]["attributes"]["colors"]


def test_overlapping_boxes_overlap(cat):
    boxes = [o["bbox"] for o in cat.get("overlapping_shapes")["objects"]]
    a, b = boxes[0], boxes[1]
    assert a["x"] + a["w"] > b["x"] and a["y"] + a["h"] > b["y"]


# --- scenario 7 / failure paths ---------------------------------------------
def test_empty_and_failed(cat):
    assert cat.get("blank")["auto"]["detection"]["status"] == "no_objects"
    assert cat.get("faint")["objects"] == []  # missed: too close to background
    failed = cat.get("cluttered")["auto"]["detection"]
    assert failed["status"] == "failed" and "background" in failed["error"]


# --- scenarios 2-4: manual correction ----------------------------------------
def test_manual_add_edit_delete_persist(cat):
    added = cat.add_object("faint", "circle", (100, 100, 201, 201), color="white")
    assert added["origin"] == "user"

    o = cat.get("boy")["objects"][0]
    edited = cat.update_object(o["id"], label="boy", color="blue", confidence=0.95, bbox=(118, 44, 165, 336))
    assert edited["edited_by_user"] and edited["auto_original"]["label"] == "person"
    assert edited["bbox"] == {"x": 118, "y": 44, "w": 165, "h": 336}

    victim = cat.get("shapes")["objects"][0]
    cat.delete_object(victim["id"])

    # automatic / batch reprocessing must not touch any of these
    for name in ("faint", "boy", "shapes"):
        with pytest.raises(ManualCorrectionsExist):
            cat.run_detection(name)
    report = cat.redetect_all()
    assert report["boy"].startswith("skipped") and report["apple"] == "detected"
    assert cat.get("boy")["objects"][0]["label"] == "boy"
    assert len(cat.get("shapes")["objects"]) == 3

    # explicit rerun keeping manual objects
    kept = cat.run_detection("faint", force=True, keep_manual=True)
    assert [o["origin"] for o in kept["objects"]] == ["user"]
    # explicit full rerun discards corrections
    fresh = cat.run_detection("boy", force=True)
    assert fresh["objects"][0]["label"] == "person" and not fresh["corrected"]


def test_edit_validation(cat):
    o = cat.get("apple")["objects"][0]
    with pytest.raises(CatalogError):
        cat.update_object(o["id"], bbox=(300, 300, 500, 500))  # outside image
    with pytest.raises(CatalogError):
        cat.update_object(o["id"], confidence=1.5)
    with pytest.raises(CatalogError):
        cat.update_object(o["id"], detector="hacked")  # system field


def test_auto_vs_user_fields_are_separated(cat):
    img = cat.get("apple")
    assert set(img["auto"]) >= {"width", "height", "sha256", "detection"}
    assert set(img["user"]) == {"display_name", "description", "tags"}
    with pytest.raises(CatalogError):
        cat.update_image("apple", width=10)
    assert cat.update_image("apple", tags={"fruit": "yes"})["user"]["tags"] == {"fruit": "yes"}


# --- scenarios 5-6: composites ------------------------------------------------
def test_parse_request():
    items = parse_request("Create an image containing one red apple and one orange")
    assert [(i.term, i.count, i.color) for i in items] == [("apple", 1, "red"), ("orange", 1, None)]
    items = parse_request("a boy, a girl and colored shapes")
    assert [(i.term, i.count) for i in items] == [("boy", 1), ("girl", 1), ("shapes", None)]
    assert parse_request("three apples")[0].count == 3


def test_composite_uses_catalog_pixels(cat):
    c = compose(cat, "Create an image containing one red apple and one orange", use_llm=False)
    prov = c["auto"]["provenance"]
    assert c["auto"]["kind"] == "composite" and prov["generated_pixels"] is False
    assert [s["image_name"] for s in prov["sources"]] == ["apple", "orange"]  # dedicated assets chosen
    assert sorted(o["label"] for o in c["objects"]) == ["apple", "orange"]

    # the pasted apple must be the catalog apple, pixel for pixel (modulo resampling)
    out = np.asarray(cat.load_pil(c["id"])).astype(int)
    pb = c["objects"][0]["bbox"]
    patch = out[pb["y"]:pb["y"] + pb["h"], pb["x"]:pb["x"] + pb["w"]]
    red = (patch[..., 0] > 180) & (patch[..., 1] < 60)
    assert red.mean() > 0.5


def test_composite_boy_girl_shapes(cat):
    for name in ("boy", "girl"):
        o = cat.get(name)["objects"][0]
        cat.update_object(o["id"], label=name)
    c = compose(cat, "Create an image with a boy, a girl and colored shapes", use_llm=False)
    got = [o["label"] for o in sorted(c["objects"], key=lambda o: o["bbox"]["x"])]
    assert got == ["boy", "girl", "circle", "square", "triangle", "rectangle"]
    boy = next(o for o in c["objects"] if o["label"] == "boy")
    circle = next(o for o in c["objects"] if o["label"] == "circle")
    assert boy["bbox"]["h"] > circle["bbox"]["h"]  # size prior: people taller than shapes
    assert cat.get(c["id"])["name"] in [i["name"] for i in cat.list()]


def test_composite_falls_back_to_image_name(cat):
    c = compose(cat, "a boy and an orange", use_llm=False)  # boy still labelled 'person'
    assert [s["image_name"] for s in c["auto"]["provenance"]["sources"]] == ["boy", "orange"]


def test_missing_asset_is_never_invented(cat):
    before = len(cat.list())
    with pytest.raises(MissingAssetError) as exc:
        compose(cat, "an apple and a banana", use_llm=False)
    assert exc.value.missing == ["banana"]
    assert len(cat.list()) == before


# --- hosted backends, exercised with a fake model ------------------------------
def test_vision_detector_keeps_proposal_boxes(images):
    def fake(content, schema):
        return {"objects": [{"label": "Boy", "color": "blue", "confidence": 0.93, "proposal_index": 0,
                             "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}}]}
    r = ClaudeVisionDetector(ask_json=fake).detect(images["boy"])
    assert r.status == "detected" and r.objects[0].label == "boy"
    assert r.objects[0].bbox == (120, 46, 161, 333)  # segmenter box, not the model's


def test_vision_detector_error_is_a_failed_run(images):
    def boom(content, schema):
        raise ConnectionError("offline")
    r = ClaudeVisionDetector(ask_json=boom).detect(images["apple"])
    assert r.status == "failed" and "offline" in r.error
