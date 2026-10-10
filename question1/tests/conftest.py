import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Fast, deterministic tests: segmentation-only detector (OWLv2 is stubbed where needed), no LLM calls.
os.environ["IMGCAT_DETECTOR"] = "seg"
os.environ["IMGCAT_OFFLINE"] = "1"

IMAGES = ROOT / "test_images"


@pytest.fixture
def svc(tmp_path):
    from imgcat.detection import Detector
    from imgcat.service import CatalogService

    return CatalogService(tmp_path / "catalog", detector=Detector(use_owl=False))


@pytest.fixture
def fake_owl(monkeypatch):
    """Replace OWLv2 with canned boxes so semantic labelling is testable offline."""
    from imgcat.detection import owl

    canned: dict = {}

    def detect(rgb, queries=None, threshold=0.1, nms_iou=0.5):
        return canned.get(rgb.shape, [])

    monkeypatch.setattr(owl, "detect", detect)
    return canned


def iou(a, b):
    ax1, ay1, bx1, by1 = a["x"] + a["width"], a["y"] + a["height"], b["x"] + b["width"], b["y"] + b["height"]
    iw = max(0, min(ax1, bx1) - max(a["x"], b["x"]))
    ih = max(0, min(ay1, by1) - max(a["y"], b["y"]))
    inter = iw * ih
    return inter / (a["width"] * a["height"] + b["width"] * b["height"] - inter)
