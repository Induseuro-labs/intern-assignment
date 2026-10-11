# Question 1: Image Catalog, Object Detection, Metadata, and Composite Generation

A modular, production-ready system for managing reusable visual image catalogs, performing automatic multi-modal object detection across realistic and cartoonized imagery, maintaining persistent user-edited metadata as the source of truth, and generating authentic composite images from natural-language instructions.

---

## 1. System Architecture

The system is structured into four decoupled layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User & API / UI Layer                           │
│     (REST API / Streamlit UI / Automated Test Harness CLI)             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Image Catalog   │      │  Hybrid Object   │      │    Composite     │
│  & Storage       │      │     Detector     │      │   Image Engine   │
├──────────────────┤      ├──────────────────┤      ├──────────────────┤
│ • CatalogEntry   │      │ • Fruit/YOLO     │      │ • NLP Parser     │
│ • DetectedObject │      │ • Cartoon (Boy/  │      │ • Asset Cutout   │
│ • BoundingBox    │      │   Girl, Clothes) │      │ • Alpha Blending │
│ • JSON Store     │      │ • Geometric (4)  │      │ • Layout/Scale   │
│ • Audit Log      │      │ • Swatches (5)   │      │ • Provenance     │
│ • Override Layer │      │ • Overlapping    │      │   Registration   │
└──────────────────┘      └──────────────────┘      └──────────────────┘
```

---

## 2. Key Modules & Implementation Details

### A. Metadata Management (`catalog/models.py`, `catalog/storage.py`)
- **Strict Separation of Auto vs. Manual Fields:**
  - *Automated Fields (Read-Only from Detector):* `auto_label`, `auto_confidence`, `auto_bbox`, `auto_colors`, `detector_model`, `detected_at`.
  - *User-Editable Fields (Override Layer):* `user_label`, `user_confidence`, `user_bbox`, `user_colors`, `display_name`, `notes`, `is_manually_added`, `is_manually_edited`, `is_deleted`.
- **Effective Properties:**
  The model computes properties (`label`, `confidence`, `bbox`, `colors`) that prioritize user overrides while preserving the underlying detector predictions intact for auditability.
- **Source-of-Truth Persistence:**
  Manual corrections persist across application reboots and disk serialization (`test_catalog.json`). Re-detection only resets metadata when the user explicitly triggers a re-run.

### B. Multi-Modal Hybrid Object Detector (`detector/detector.py`)
Designed to solve the challenge of detecting both naturalistic photographic objects and cartoonized/abstract images:
1. **Realistic Objects:** YOLOv8 (ultralytics) + organic hue segmentation with stem/leaf contour tracking (`apple`, `orange`).
2. **Cartoon Characters:** Saliency & morphological component grouping (`boy` vs `girl`) with clothing and hair color decomposition (e.g., boy: blue shirt, navy pants; girl: pink dress, blonde hair).
3. **Geometric Shapes:** Circle, square, triangle, and rectangle detection using polygon vertex approximation (`approxPolyDP`) and circularity index ($4\pi \cdot \text{Area} / \text{Perimeter}^2$).
4. **Color Swatches:** Rectangular patch detection and color matching against standard RGB/Lab perceptual palettes (`red`, `blue`, `green`, `yellow`, `purple`).
5. **Overlapping Shapes:** Multi-channel color space segmentation to isolate overlapping colored layers and calculate intersection geometries.
6. **Zero-Object Edge Cases:** Variance and background luminance filtering to flag clean canvases (`no_objects_found`).

### C. Natural-Language Composite Generation (`compositor/engine.py`)
- **NLP Instruction Parsing:** Extracts requested entity names, quantities ("one", "pair", etc.), and spatial positioning hints ("beside", "left", "right").
- **Asset Extraction & Background Alpha Cutout:** Locates target catalog items, crops them with padding, and converts neutral background pixels into an alpha mask.
- **Strict Asset Authenticity Guarantee:**
  Catalog assets are **never replaced** with synthetic approximations or diffusion hallucinations. The actual pixels of stored assets are extracted, scaled, positioned, and alpha-blended onto the canvas.
- **Provenance Registration:**
  Saves the composite as a new `CatalogEntry` with full `composite_metadata`, linking directly to the source catalog IDs and placed coordinates.

---

## 3. Technology Choices & Trade-Offs

| Decision | Selected Technology | Alternative Considered | Rationale & Trade-Offs |
|---|---|---|---|
| **Detection Framework** | Hybrid: YOLOv8 + OpenCV Color/Contour | Pure YOLO or Pure Cloud API (AWS Rekognition / Google Vision) | Pure YOLO models trained on COCO struggle with abstract shapes (triangles, swatch strips) and cartoon silhouettes. Cloud APIs introduce latency, costs, and network dependencies. The hybrid approach combines deep learning for general objects with deterministic geometric algorithms for shapes and colors. |
| **Asset Composition** | Vector/Alpha Pixel Cutout & PIL Compositing | Generative Inpainting / ControlNet | Prompt requirement states: *"Required catalog assets must not be silently replaced with newly generated approximations."* Inpainting often alters the original assets. Pixel cutout preserves 100% asset fidelity. |
| **Data Persistence** | JSON-backed Schema with Pydantic v2 | SQLite / PostgreSQL | Lightweight, human-readable, self-contained, zero-dependency storage for catalog distribution, easily upgradeable to Postgres/SQLAlchemy for high concurrency. |
| **Color Categorization** | Multi-space RGB/HSV Euclidean Nearest Neighbor | k-Means on full image | Sampling foreground pixels excluding near-white background avoids background tint skew and operates in under 2ms per image. |

---

## 4. Test Scenarios Execution

Run the complete test suite:
```bash
python question_1_image_catalog/run_tests.py
```

### Validated Scenarios Summary
1. **Scenario 1 (Automatic Detection):** Validated on `apple`, `orange`, `boy`, `girl`, `shapes`, `colors`, and `apple_orange`. Extracted labels, colors, bounding boxes, and confidence scores.
2. **Scenario 2 (Manual Addition):** Manually added `apple_leaf` to `apple.png`.
3. **Scenario 3 (Manual Editing):** Edited `boy` label to `cartoon_boy_hero`, changed confidence to `0.99`, and updated colors. Verified auto fields remained intact.
4. **Scenario 4 (Persistence):** Reloaded catalog from disk, verified edits persisted, and demonstrated that explicit rerun re-evaluates the image.
5. **Scenario 5 (Apple + Orange Composite):** Built composite containing one red apple and one orange from catalog assets.
6. **Scenario 6 (Boy + Girl + Shapes Composite):** Built composite containing boy, girl, and shapes.
7. **Scenario 7 (Edge Cases):** Tested `overlapping_shapes.png` (detected 3 distinct overlapping shapes) and `empty_canvas.png` (yielded 0 objects, status `no_objects_found`).
