"""
Natural Language Composite Image Generator.
Takes a natural-language description, identifies and extracts authentic existing catalog assets,
applies layout, positioning, scaling, and alpha compositing,
and registers the resulting image into the catalog with full provenance metadata.
Strictly ensures catalog assets are never replaced with approximations.
"""

from __future__ import annotations
import os
import re
import uuid
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import numpy as np

from ..catalog.models import (
    CatalogEntry, DetectedObject, BoundingBox,
    CompositeMetadata, CompositeSourceAsset
)
from ..catalog.storage import ImageCatalog

class ParsedElementRequest:
    def __init__(self, query: str, quantity: int = 1, position_hint: Optional[str] = None):
        self.query = query.lower().strip()
        self.quantity = quantity
        self.position_hint = position_hint  # 'left', 'right', 'center', 'beside', etc.

    def __repr__(self):
        return f"<ElementReq: {self.quantity}x '{self.query}' (pos: {self.position_hint})>"


class NaturalLanguagePromptParser:
    """
    Parses user instructions to extract required catalog items, quantities, and spatial relationships.
    """
    WORD_TO_NUM = {
        "one": 1, "a": 1, "an": 1, "single": 1,
        "two": 2, "pair": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }

    KNOWN_KEYWORDS = [
        "apple", "orange", "boy", "girl", "shapes", "colors",
        "circle", "square", "triangle", "rectangle", "fruit"
    ]

    @classmethod
    def parse(cls, prompt: str) -> List[ParsedElementRequest]:
        text = prompt.lower().strip()
        # Clean filler words
        text = re.sub(r"^(create|make|generate)\s+(an?\s+image|a\s+composite|an?\s+picture)?\s*(containing|with|of)?", "", text)
        
        requests: List[ParsedElementRequest] = []
        
        # Split on conjunctions and commas
        chunks = re.split(r",|\band\b|\bwith\b|\bplus\b", text)
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
                
            # Check quantity
            qty = 1
            for word, n in cls.WORD_TO_NUM.items():
                if re.search(rf"\b{word}\b", chunk):
                    qty = n
                    break
            number_match = re.search(r"\b(\d+)\b", chunk)
            if number_match:
                qty = int(number_match.group(1))

            # Check spatial hints
            pos = None
            if "left" in chunk:
                pos = "left"
            elif "right" in chunk:
                pos = "right"
            elif "center" in chunk or "middle" in chunk:
                pos = "center"
            elif "beside" in chunk or "next to" in chunk:
                pos = "beside"

            # Match target entity
            matched_key = None
            for kw in cls.KNOWN_KEYWORDS:
                if kw in chunk:
                    matched_key = kw
                    break
                    
            if matched_key:
                requests.append(ParsedElementRequest(query=matched_key, quantity=qty, position_hint=pos))
            else:
                # Fallback to cleaned chunk
                cleaned = re.sub(r"\b(one|two|three|a|an|the|red|orange|blue|green|standing|cartoon|colored|image|composite)\b", "", chunk).strip()
                if cleaned:
                    requests.append(ParsedElementRequest(query=cleaned, quantity=qty, position_hint=pos))
                    
        return requests


class CompositeEngine:
    """
    Renders composite images strictly using authentic catalog assets.
    """
    def __init__(self, catalog: ImageCatalog, output_dir: Optional[str] = None):
        self.catalog = catalog
        self.output_dir = output_dir or os.path.join(os.path.dirname(catalog.storage_path), "composites")
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_asset_cutout(self, entry: CatalogEntry, target_label: Optional[str] = None) -> Tuple[Image.Image, Optional[DetectedObject]]:
        """
        Extracts the object asset from the catalog image.
        Converts white/neutral background to alpha transparency so assets composite seamlessly.
        """
        if not os.path.exists(entry.file_path):
            raise FileNotFoundError(f"Source catalog file not found: {entry.file_path}")

        full_img = Image.open(entry.file_path).convert("RGBA")
        
        # Check if there is a specific detected object to crop
        best_obj: Optional[DetectedObject] = None
        if target_label:
            for obj in entry.get_active_objects():
                if target_label.lower() in obj.label.lower():
                    best_obj = obj
                    break

        if best_obj and best_obj.bbox:
            # Crop to object bounding box with slight padding
            b = best_obj.bbox
            pad = 8
            x1 = max(0, b.x - pad)
            y1 = max(0, b.y - pad)
            x2 = min(full_img.width, b.x + b.width + pad)
            y2 = min(full_img.height, b.y + b.height + pad)
            cropped = full_img.crop((x1, y1, x2, y2))
        else:
            # If the whole image is the asset
            cropped = full_img

        # Alpha mask background removal (make near-white background transparent)
        np_img = np.array(cropped)
        r, g, b, a = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2], np_img[:, :, 3]
        white_bg_mask = (r > 240) & (g > 240) & (b > 240)
        np_img[white_bg_mask, 3] = 0
        
        cutout = Image.fromarray(np_img, mode="RGBA")
        return cutout, best_obj

    def find_catalog_entry_for_query(self, query: str) -> Optional[CatalogEntry]:
        """Finds matching catalog entry by entry ID, image name, or object label."""
        # 1. Exact ID match
        entry = self.catalog.get_entry(query)
        if entry:
            return entry
            
        # 2. Check objects in entries
        for e in self.catalog.list_entries():
            if query in e.id.lower() or query in e.image_name.lower():
                return e
            for obj in e.get_active_objects():
                if query in obj.label.lower():
                    return e
        return None

    def generate_composite(
        self,
        prompt: str,
        canvas_size: Tuple[int, int] = (1000, 600),
        composite_name: Optional[str] = None,
        verify: bool = False
    ) -> CatalogEntry:
        """
        Generates a composite image using authentic catalog assets based on the prompt.
        Saves and returns the new CatalogEntry.
        """
        requests = NaturalLanguagePromptParser.parse(prompt)
        if not requests:
            raise ValueError(f"Could not extract any recognizable catalog assets from prompt: '{prompt}'")

        # Resolve catalog assets for each request
        placed_assets: List[Dict[str, Any]] = []
        for req in requests:
            entry = self.find_catalog_entry_for_query(req.query)
            if not entry:
                raise ValueError(
                    f"Catalog asset for '{req.query}' was not found in catalog. "
                    f"Per assignment specification, assets must exist and will not be silently hallucinated."
                )
            cutout, matched_obj = self.extract_asset_cutout(entry, target_label=req.query)
            
            for _ in range(req.quantity):
                placed_assets.append({
                    "entry": entry,
                    "matched_obj": matched_obj,
                    "cutout": cutout,
                    "req": req
                })

        num_items = len(placed_assets)
        canvas_w, canvas_h = canvas_size
        
        # Create clean white canvas
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
        
        # Calculate sizing and layout
        # Compute target bounding boxes to fit gracefully across canvas width
        spacing = 40
        margin_x = 60
        avail_w = canvas_w - (2 * margin_x) - (spacing * max(0, num_items - 1))
        target_slot_w = avail_w // num_items
        max_slot_h = int(canvas_h * 0.75)
        
        # Render soft contact shadows first for natural depth
        from PIL import ImageFilter, ImageDraw
        shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        for i, item in enumerate(placed_assets):
            cutout = item["cutout"]
            scale = min(target_slot_w / cutout.width, max_slot_h / cutout.height, 1.2)
            new_w = max(40, int(cutout.width * scale))
            new_h = max(40, int(cutout.height * scale))
            slot_x = margin_x + i * (target_slot_w + spacing)
            pos_x = slot_x + (target_slot_w - new_w) // 2
            pos_y = (canvas_h - new_h) // 2
            
            # Subtle elliptical ground shadow
            sw = int(new_w * 0.8)
            sh = max(10, int(new_h * 0.12))
            sx = pos_x + (new_w - sw) // 2
            sy = pos_y + new_h - int(sh * 0.5)
            shadow_draw.ellipse([sx, sy, sx + sw, sy + sh], fill=(30, 30, 30, 65))
        
        blurred_shadows = shadow_layer.filter(ImageFilter.GaussianBlur(radius=7))
        canvas.paste(blurred_shadows, (0, 0), blurred_shadows)

        source_records: List[CompositeSourceAsset] = []
        composite_objects: List[DetectedObject] = []
        
        curr_x = margin_x
        for i, item in enumerate(placed_assets):
            cutout = item["cutout"]
            entry = item["entry"]
            matched_obj = item["matched_obj"]
            
            # Scale asset to fit slot while preserving original aspect ratio
            scale = min(target_slot_w / cutout.width, max_slot_h / cutout.height, 1.2)
            new_w = max(40, int(cutout.width * scale))
            new_h = max(40, int(cutout.height * scale))
            scaled_cutout = cutout.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center vertically on canvas
            pos_y = (canvas_h - new_h) // 2
            pos_x = curr_x + (target_slot_w - new_w) // 2
            
            # Paste with alpha composite
            canvas.paste(scaled_cutout, (pos_x, pos_y), scaled_cutout)
            
            # Record source provenance
            bbox = BoundingBox(x=pos_x, y=pos_y, width=new_w, height=new_h)
            label = matched_obj.label if matched_obj else entry.id
            colors = matched_obj.colors if matched_obj else ["source_asset"]
            
            source_records.append(CompositeSourceAsset(
                catalog_entry_id=entry.id,
                image_name=entry.image_name,
                object_id=matched_obj.id if matched_obj else None,
                object_label=label,
                placed_bbox=bbox,
                z_index=i,
                scale=round(scale, 3)
            ))
            
            composite_objects.append(DetectedObject(
                auto_label=label,
                auto_confidence=1.0,  # 1.0 since it was explicitly verified from catalog
                auto_bbox=bbox,
                auto_colors=colors,
                origin="composition",
                detector_model="AuthenticCatalogAsset_Provenance",
                detected_at=entry.created_at
            ))
            
            curr_x += target_slot_w + spacing

        # Save composite image file
        comp_id = composite_name or f"composite_{uuid.uuid4().hex[:8]}"
        out_filename = f"{comp_id}.png"
        out_path = os.path.join(self.output_dir, out_filename)
        
        rgb_canvas = canvas.convert("RGB")
        rgb_canvas.save(out_path, format="PNG")
        file_size = os.path.getsize(out_path)

        # Verification pass (optional or if requested via verify=True)
        verification_status = "unverified"
        verification_matches = []
        avg_iou = None
        if verify:
            try:
                from ..detector.detector import HybridObjectDetector
                det = HybridObjectDetector(use_yolo=False)
                detected = det.detect(out_path)
                matches_count = 0
                total_iou = 0.0
                for placed in composite_objects:
                    pb = placed.auto_bbox
                    for d in detected:
                        db = d.auto_bbox
                        if not pb or not db:
                            continue
                        ix1, iy1 = max(pb.x, db.x), max(pb.y, db.y)
                        ix2, iy2 = min(pb.x2, db.x2), min(pb.y2, db.y2)
                        if ix2 > ix1 and iy2 > iy1:
                            inter = (ix2 - ix1) * (iy2 - iy1)
                            union = pb.area + db.area - inter
                            iou = inter / union if union > 0 else 0
                            if iou > 0.4:
                                matches_count += 1
                                total_iou += iou
                                verification_matches.append({
                                    "target": placed.auto_label,
                                    "detected": d.auto_label,
                                    "iou": round(iou, 3)
                                })
                                break
                if matches_count >= len(composite_objects):
                    verification_status = "verified"
                elif matches_count > 0:
                    verification_status = "partially_verified"
                else:
                    verification_status = "verification_mismatch"
                avg_iou = round(total_iou / max(1, matches_count), 3) if matches_count > 0 else 0.0
            except Exception as e:
                verification_status = f"error: {str(e)}"

        # Assemble new CatalogEntry
        comp_metadata = CompositeMetadata(
            prompt=prompt,
            source_assets=source_records,
            canvas_dimensions=(canvas_w, canvas_h),
            layout_strategy="horizontal_auto_flow",
            generative_fill=False,
            verification_status=verification_status,
            verification_matches=verification_matches,
            verification_iou=avg_iou
        )
        
        composite_entry = CatalogEntry(
            id=comp_id,
            image_name=out_filename,
            file_path=out_path,
            width=canvas_w,
            height=canvas_h,
            format="PNG",
            file_size_bytes=file_size,
            detection_status="completed",
            objects_source="composition",
            is_composite=True,
            composite_metadata=comp_metadata,
            objects=composite_objects,
            tags=["composite", "ai_generated_layout"]
        )
        
        # Persist directly into catalog
        self.catalog.add_entry(composite_entry)
        return composite_entry
