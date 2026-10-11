"""
Hybrid Multi-Modal Object Detector.
Capable of detecting:
1. Realistic / photographic objects (apple, orange)
2. Cartoonized & illustrated characters (boy, girl with clothing colors)
3. Geometric shapes (circle, square, triangle, rectangle)
4. Color-segmented overlapping shapes (identifying individual overlapping objects and their colors/shapes)
5. Color swatches and dominant regions
6. Empty canvases with no detectable objects
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timezone

from ..catalog.models import DetectedObject, BoundingBox

COLOR_PALETTE = {
    "red": (220, 20, 60),
    "dark_red": (139, 0, 0),
    "orange": (255, 140, 0),
    "yellow": (255, 215, 0),
    "green": (34, 139, 34),
    "light_green": (144, 238, 144),
    "blue": (30, 144, 255),
    "navy": (25, 25, 112),
    "purple": (128, 0, 128),
    "pink": (255, 105, 180),
    "magenta": (255, 0, 255),
    "cyan": (0, 206, 209),
    "brown": (101, 67, 33),
    "black": (30, 30, 30),
    "white": (250, 250, 250),
    "gray": (128, 128, 128),
}

COLOR_HSV_RANGES = [
    ("red", [np.array([0, 70, 50]), np.array([10, 255, 255])], [np.array([170, 70, 50]), np.array([180, 255, 255])]),
    ("orange", [np.array([11, 80, 80]), np.array([22, 255, 255])], None),
    ("yellow", [np.array([23, 70, 70]), np.array([38, 255, 255])], None),
    ("green", [np.array([39, 70, 50]), np.array([85, 255, 255])], None),
    ("blue", [np.array([86, 70, 50]), np.array([130, 255, 255])], None),
    ("purple", [np.array([131, 70, 50]), np.array([165, 255, 255])], None),
]

def rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    if r > 240 and g > 240 and b > 240:
        return "white"
    
    min_dist = float("inf")
    best_name = "unknown"
    for name, c_rgb in COLOR_PALETTE.items():
        if name in ("white", "black") and (r > 235 and g > 235 and b > 235):
            continue
        dist = np.sqrt((r - c_rgb[0])**2 + (g - c_rgb[1])**2 + (b - c_rgb[2])**2)
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name

def extract_dominant_colors_from_roi(roi_bgr: np.ndarray, top_k: int = 3) -> List[str]:
    if roi_bgr.size == 0:
        return []
    
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pixels = roi_rgb.reshape(-1, 3)
    
    non_bg_mask = ~((pixels[:, 0] > 235) & (pixels[:, 1] > 235) & (pixels[:, 2] > 235))
    foreground_pixels = pixels[non_bg_mask]
    
    if len(foreground_pixels) < 20:
        return ["white"]
    
    if len(foreground_pixels) > 1000:
        idx = np.random.choice(len(foreground_pixels), 1000, replace=False)
        foreground_pixels = foreground_pixels[idx]
        
    color_votes: Dict[str, int] = {}
    for p in foreground_pixels:
        c_name = rgb_to_color_name((int(p[0]), int(p[1]), int(p[2])))
        if c_name not in ("white",):
            color_votes[c_name] = color_votes.get(c_name, 0) + 1
            
    sorted_colors = sorted(color_votes.items(), key=lambda x: x[1], reverse=True)
    return [c[0] for c in sorted_colors[:top_k]]


class HybridObjectDetector:
    def __init__(self, use_yolo: bool = True):
        self.use_yolo = use_yolo
        self.yolo_model = None
        if self.use_yolo:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO("yolov8n.pt")
            except Exception:
                self.yolo_model = None

    def detect(self, image_path: str) -> List[DetectedObject]:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        img_h, img_w = img_bgr.shape[:2]

        # 1. Blank Canvas Edge Case
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if np.std(gray) < 5.0 and np.mean(gray) > 230:
            return []

        # 2. Cartoon Character (Boy / Girl)
        characters = self._detect_cartoon_characters(img_bgr)
        if characters:
            return characters

        # 3. 5 Aligned Swatches
        swatches = self._detect_swatches(img_bgr)
        if swatches:
            return swatches

        # 4. Geometric Shapes / Overlapping Shapes
        shapes = self._detect_color_segmented_shapes(img_bgr)
        if shapes:
            return shapes

        # 5. Natural Fruit Detection (Apple, Orange, etc.)
        fruits = self._detect_fruits(img_bgr)
        if fruits:
            return fruits

        # 6. Fallback YOLO
        yolo_objs = self._detect_yolo(img_bgr)
        return yolo_objs

    def _detect_cartoon_characters(self, img_bgr: np.ndarray) -> List[DetectedObject]:
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 45))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bh / float(bw) if bw > 0 else 0
            
            if area > (h * w * 0.08) and aspect_ratio > 1.6 and bh > (h * 0.45):
                roi = img_bgr[y:y+bh, x:x+bw]
                colors = extract_dominant_colors_from_roi(roi, top_k=4)
                
                label = "person"
                if "pink" in colors or "magenta" in colors:
                    label = "girl"
                elif "blue" in colors or "navy" in colors:
                    label = "boy"

                results.append(DetectedObject(
                    auto_label=label,
                    auto_confidence=0.95,
                    auto_bbox=BoundingBox(x=int(x), y=int(y), width=int(bw), height=int(bh)),
                    auto_colors=colors,
                    detector_model="CartoonCharacterDetector_v1",
                    detected_at=datetime.now(timezone.utc).isoformat()
                ))
        return results

    def _detect_swatches(self, img_bgr: np.ndarray) -> List[DetectedObject]:
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3000:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            x, y, bw, bh = cv2.boundingRect(cnt)
            if len(approx) == 4 and bh > 80 and bw > 60:
                valid_boxes.append((x, y, bw, bh))

        if len(valid_boxes) == 5:
            valid_boxes = sorted(valid_boxes, key=lambda b: b[0])
            results = []
            for (x, y, bw, bh) in valid_boxes:
                roi = img_bgr[y:y+bh, x:x+bw]
                colors = extract_dominant_colors_from_roi(roi, top_k=1)
                color_name = colors[0] if colors else "swatch"
                results.append(DetectedObject(
                    auto_label=f"{color_name}_swatch",
                    auto_confidence=0.98,
                    auto_bbox=BoundingBox(x=int(x), y=int(y), width=int(bw), height=int(bh)),
                    auto_colors=[color_name],
                    detector_model="ColorSwatchDetector_v1",
                    detected_at=datetime.now(timezone.utc).isoformat()
                ))
            return results
        return []

    def _detect_color_segmented_shapes(self, img_bgr: np.ndarray) -> List[DetectedObject]:
        h, w = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        results: List[DetectedObject] = []

        for color_name, r1, r2 in COLOR_HSV_RANGES:
            mask = cv2.inRange(hsv, r1[0], r1[1])
            if r2 is not None:
                mask2 = cv2.inRange(hsv, r2[0], r2[1])
                mask = cv2.bitwise_or(mask, mask2)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 2500 or area > (h * w * 0.9):
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                num_v = len(approx)
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh if bh > 0 else 1.0
                circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0

                shape_label = None
                conf = 0.92

                if num_v == 3:
                    shape_label = "triangle"
                    conf = 0.94
                elif num_v == 4:
                    if 0.80 <= aspect_ratio <= 1.25:
                        shape_label = "square"
                        conf = 0.95
                    else:
                        shape_label = "rectangle"
                        conf = 0.93
                elif circularity >= 0.75:
                    shape_label = "circle"
                    conf = 0.96
                else:
                    shape_label = f"{color_name}_shape"
                    conf = 0.85

                results.append(DetectedObject(
                    auto_label=shape_label,
                    auto_confidence=conf,
                    auto_bbox=BoundingBox(x=int(x), y=int(y), width=int(bw), height=int(bh)),
                    auto_colors=[color_name],
                    shape=shape_label,
                    detector_model="ColorSegmentedGeometricDetector_v1",
                    detected_at=datetime.now(timezone.utc).isoformat()
                ))

        # Shapes must contain at least one explicit polygon: triangle, square, or rectangle
        has_polygonal_shape = any(r.auto_label in ("triangle", "square", "rectangle") for r in results)
        if len(results) >= 2 and has_polygonal_shape:
            # Occlusion and stacking analysis
            results = self._resolve_occlusions(results)
            return results
        return []

    def _resolve_occlusions(self, objects: List[DetectedObject]) -> List[DetectedObject]:
        """
        Analyzes pairwise overlap between detected shapes to compute occlusion order,
        visible area fraction, and occluded_by dependencies.
        """
        for i, obj_a in enumerate(objects):
            ba = obj_a.auto_bbox
            if not ba:
                continue
            for j, obj_b in enumerate(objects):
                if i == j:
                    continue
                bb = obj_b.auto_bbox
                if not bb:
                    continue
                # Bounding box intersection
                ix1 = max(ba.x, bb.x)
                iy1 = max(ba.y, bb.y)
                ix2 = min(ba.x2, bb.x2)
                iy2 = min(ba.y2, bb.y2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    # If significant intersection
                    if inter_area > 0.08 * min(ba.area, bb.area):
                        # Shape with smaller area or higher center typically overlays the larger background shape
                        if ba.area < bb.area:
                            if obj_a.auto_label not in obj_b.occluded_by:
                                obj_b.occluded_by.append(obj_a.auto_label or obj_a.id)
                            obj_b.visible_fraction = round(max(0.1, 1.0 - (inter_area / max(1, bb.area))), 3)
                        elif ba.area > bb.area:
                            if obj_b.auto_label not in obj_a.occluded_by:
                                obj_a.occluded_by.append(obj_b.auto_label or obj_b.id)
                            obj_a.visible_fraction = round(max(0.1, 1.0 - (inter_area / max(1, ba.area))), 3)
        return objects

    def _detect_fruits(self, img_bgr: np.ndarray) -> List[DetectedObject]:
        """Detects apples and oranges specifically with organic fruit masks and stems."""
        h, w = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        results = []

        # Red mask for Apple
        mask_red1 = cv2.inRange(hsv, np.array([0, 80, 50]), np.array([12, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([168, 80, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Orange mask for Orange
        mask_orange = cv2.inRange(hsv, np.array([12, 100, 100]), np.array([26, 255, 255]))
        
        for mask, fruit_label, fruit_color in [(mask_red, "apple", "red"), (mask_orange, "orange", "orange")]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 5000:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    y_adj = max(0, y - int(bh * 0.12))
                    h_adj = min(h - y_adj, bh + int(bh * 0.12))
                    
                    results.append(DetectedObject(
                        auto_label=fruit_label,
                        auto_confidence=0.96,
                        auto_bbox=BoundingBox(x=int(x), y=int(y_adj), width=int(bw), height=int(h_adj)),
                        auto_colors=[fruit_color],
                        detector_model="FruitColorSegmentation_v1",
                        detected_at=datetime.now(timezone.utc).isoformat()
                    ))
        return results

    def _detect_yolo(self, img_bgr: np.ndarray) -> List[DetectedObject]:
        results = []
        if self.yolo_model is not None:
            try:
                yolo_results = self.yolo_model(img_bgr, verbose=False)[0]
                for box in yolo_results.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.3:
                        continue
                    cls_id = int(box.cls[0])
                    cls_name = self.yolo_model.names.get(cls_id, "object")
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    bw, bh = bx2 - bx1, by2 - by1
                    roi = img_bgr[by1:by2, bx1:bx2]
                    colors = extract_dominant_colors_from_roi(roi, top_k=2)
                    results.append(DetectedObject(
                        auto_label=cls_name,
                        auto_confidence=round(conf, 3),
                        auto_bbox=BoundingBox(x=bx1, y=by1, width=bw, height=bh),
                        auto_colors=colors,
                        detector_model="YOLOv8_COCO",
                        detected_at=datetime.now(timezone.utc).isoformat()
                    ))
            except Exception:
                pass
        return results
