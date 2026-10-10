import json

import numpy as np
from PIL import Image

from conftest import IMAGES, iou
from imgcat.detection import Detector
from imgcat.detection.segmentation import estimate_style
from imgcat.detection.shapes import fit_shape

TRUTH = json.loads((IMAGES / "ground_truth.json").read_text())


def rgb(name):
    return np.array(Image.open(IMAGES / f"{name}.png").convert("RGB"))


def detect(name, use_owl=False):
    return Detector(use_owl=use_owl).detect(rgb(name))


def assert_matches_truth(name, objects, min_iou=0.95):
    autos = [o.auto for o in objects]
    assert len(autos) == len(TRUTH[name])
    for gt in TRUTH[name]:
        best = max(autos, key=lambda a: iou(a["bbox"], gt["bbox"]))
        assert iou(best["bbox"], gt["bbox"]) >= min_iou, (gt, best["bbox"])
        assert best["color"] == gt["color"]
        assert gt["name"] in (best["name"], best["display_name"]) or gt["name"].endswith(best["name"])


def test_shapes_names_colours_boxes():
    r = detect("shapes")
    assert_matches_truth("shapes", r.objects)
    assert {o.auto["name"] for o in r.objects} == {"circle", "square", "triangle", "rectangle"}
    assert all(o.auto["confidence"] > 0.9 for o in r.objects)


def test_colour_swatches():
    r = detect("colors")
    assert_matches_truth("colors", r.objects, min_iou=0.98)
    assert {o.auto["display_name"] for o in r.objects} == {
        "red swatch", "blue swatch", "green swatch", "yellow swatch", "purple swatch"}


def test_overlapping_shapes_get_full_boxes_and_occlusion_order():
    r = detect("overlapping_shapes")
    assert_matches_truth("overlapping_shapes", r.objects)
    by = {o.auto["name"]: o.auto for o in r.objects}
    assert by["square"]["attributes"]["occluded_by"] == ["red circle"]
    assert by["circle"]["attributes"]["occluded_by"] == ["green triangle"]
    assert by["triangle"]["attributes"]["occluded"] is False
    assert iou(by["square"]["bbox"], by["circle"]["bbox"]) > 0  # boxes overlap
    # the amodal circle box is larger than its visible part
    vis = by["circle"]["attributes"]["visible_bbox"]
    assert vis["width"] * vis["height"] <= by["circle"]["bbox"]["width"] * by["circle"]["bbox"]["height"]


def test_images_without_objects():
    assert detect("blank").objects == []
    noise = detect("noise")
    assert noise.objects == [] and not noise.background_plain


def test_occluded_circle_fit():
    m = np.zeros((300, 300), bool)
    yy, xx = np.mgrid[:300, :300]
    circle = (xx - 150) ** 2 + (yy - 150) ** 2 <= 100 ** 2
    occluder = xx > 200
    fit = fit_shape(circle & ~occluder, occluder)
    assert fit.name == "circle"
    assert abs(fit.bbox["width"] - 201) <= 3  # full diameter recovered
    assert fit.coverage < 0.97


def test_style_estimation():
    # measured over the foreground, so a photo on a smooth backdrop still counts as a photo
    assert detect("shapes").style == "illustration"
    assert detect("boy").style == "illustration"
    assert detect("apple").style == "photo"
    assert estimate_style(rgb("noise")) == "photo"


def test_semantic_labels_from_owl_with_tight_segmentation_boxes(fake_owl):
    img = rgb("apple_orange")
    fake_owl[img.shape] = [
        {"label": "orange", "score": 0.74, "box": [530.0, 133.0, 836.0, 442.0]},
        {"label": "apple", "score": 0.72, "box": [191.0, 126.0, 488.0, 445.0]},
    ]
    r = Detector(use_owl=True).detect(img)
    names = [o.auto["name"] for o in r.objects]
    assert names == ["apple", "orange"]
    apple, orange = (o.auto for o in r.objects)
    assert apple["color"] == "red" and orange["color"] == "orange"
    assert apple["detector"] == "segmentation+owlv2"
    # contact shadow trimmed: the box ends at the fruit, not the shadow below it
    assert apple["bbox"]["y"] + apple["bbox"]["height"] < 450


def test_owl_box_on_plain_background_is_dropped(fake_owl):
    img = rgb("blank")
    fake_owl[img.shape] = [{"label": "square", "score": 0.3, "box": [0.0, 0.0, 800.0, 600.0]}]
    r = Detector(use_owl=True).detect(img)
    assert r.objects == [] and any("background" in n for n in r.notes)


def test_person_clothing_colours(fake_owl):
    img = rgb("boy")
    fake_owl[img.shape] = [{"label": "boy", "score": 0.76, "box": [426.0, 31.0, 600.0, 540.0]}]
    r = Detector(use_owl=True).detect(img)
    (boy,) = [o.auto for o in r.objects]
    assert boy["name"] == "boy" and boy["category"] == "person"
    assert boy["attributes"]["clothing_colors"][:2] == ["blue", "green"]
