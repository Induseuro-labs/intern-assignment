"""
Bounding box and label visualizer.
Draws detected object annotations on catalog images for display in the Streamlit UI.
"""

from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Tuple
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from question_1_image_catalog.catalog.models import CatalogEntry, DetectedObject

# Color palette for object boxes
BOX_COLORS = [
    (255, 59, 48),    # Red
    (30, 144, 255),   # Blue
    (52, 199, 89),    # Green
    (255, 149, 0),    # Orange
    (71, 85, 105),    # Slate
    (90, 200, 250),   # Cyan
    (255, 214, 10),   # Yellow
    (255, 45, 85),    # Pink
]


def draw_detections(image_path: str, entry: CatalogEntry, scale_to: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    Draws bounding boxes and labels on a catalog image.
    Returns a PIL Image with annotations.
    """
    img = Image.open(image_path).convert("RGB")

    if scale_to:
        img = img.resize(scale_to, Image.Resampling.LANCZOS)
        sx = scale_to[0] / entry.width
        sy = scale_to[1] / entry.height
    else:
        sx, sy = 1.0, 1.0

    draw = ImageDraw.Draw(img, "RGBA")
    active_objects = entry.get_active_objects()

    for idx, obj in enumerate(active_objects):
        bbox = obj.bbox
        if not bbox:
            continue

        color = BOX_COLORS[idx % len(BOX_COLORS)]
        alpha_fill = (*color, 35)

        x1 = int(bbox.x * sx)
        y1 = int(bbox.y * sy)
        x2 = int((bbox.x + bbox.width) * sx)
        y2 = int((bbox.y + bbox.height) * sy)

        # Filled rectangle (semi-transparent)
        draw.rectangle([x1, y1, x2, y2], fill=alpha_fill, outline=color, width=2)

        # Label background
        label = obj.label
        conf = f"{obj.confidence:.0%}"
        is_manual = "[Manual] " if obj.is_user_modified else ""
        label_text = f" {is_manual}{label} ({conf}) "

        label_x = x1
        label_y = max(0, y1 - 22)

        draw.rectangle([label_x, label_y, label_x + len(label_text) * 7, label_y + 20], fill=(*color, 220))
        draw.text((label_x + 4, label_y + 3), label_text.strip(), fill=(255, 255, 255))

    return img


def draw_composite_sources(image_path: str, entry: CatalogEntry) -> Image.Image:
    """
    Draws source asset bounding boxes for composite images.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    if not entry.is_composite or not entry.composite_metadata:
        return img

    for idx, src in enumerate(entry.composite_metadata.source_assets):
        bbox = src.placed_bbox
        if not bbox:
            continue

        color = BOX_COLORS[idx % len(BOX_COLORS)]
        x1, y1 = bbox.x, bbox.y
        x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height

        draw.rectangle([x1, y1, x2, y2], fill=(*color, 25), outline=color, width=2)

        label_text = f" {src.object_label} (from: {src.catalog_entry_id}) "
        draw.rectangle([x1, max(0, y1 - 22), x1 + len(label_text) * 7, max(0, y1)], fill=(*color, 200))
        draw.text((x1 + 4, max(0, y1 - 19)), label_text.strip(), fill=(255, 255, 255))

    return img
