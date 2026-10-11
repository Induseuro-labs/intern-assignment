"""
Data models for Image Catalog and Object Metadata.
Strictly distinguishes between automatically generated detection fields
and user-editable fields, supporting full auditability and manual source-of-truth persistence.
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    """Bounding box coordinates (x, y, width, height) in pixel values."""
    x: int = Field(..., description="Top-left X coordinate")
    y: int = Field(..., description="Top-left Y coordinate")
    width: int = Field(..., description="Box width in pixels")
    height: int = Field(..., description="Box height in pixels")

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


class ColorInfo(BaseModel):
    name: str = Field(..., description="Human-readable color name (e.g. 'red', 'navy')")
    hex_code: Optional[str] = None
    rgb: Optional[tuple[int, int, int]] = None
    dominance_pct: Optional[float] = None


class ProtectedObjectsError(Exception):
    """
    Raised when detection is requested on an image whose objects have manual edits,
    without explicitly passing force=True.
    Corresponds to HTTP 409 Conflict in REST API.
    """
    def __init__(self, message: str, entry_id: str, manual_count: int):
        super().__init__(message)
        self.entry_id = entry_id
        self.manual_count = manual_count


class DetectedObject(BaseModel):
    """
    Represents an object detected in an image.
    Maintains a strict boundary between automated detector outputs
    and user edits/overrides.
    """
    id: str = Field(default_factory=lambda: f"obj_{uuid.uuid4().hex[:8]}")

    # ================= AUTOMATIC DETECTION FIELDS (READ-ONLY) =================
    auto_label: Optional[str] = Field(None, description="Raw label emitted by detector")
    auto_confidence: Optional[float] = Field(None, description="Detector confidence score [0.0 - 1.0]")
    auto_bbox: Optional[BoundingBox] = Field(None, description="Detector bounding box")
    auto_colors: List[str] = Field(default_factory=list, description="Detector extracted dominant colors")
    detector_model: Optional[str] = Field(None, description="Name/version of detector used")
    detected_at: Optional[str] = Field(None, description="ISO timestamp when detection ran")
    
    # Occlusion and Geometry Fit
    shape: Optional[str] = Field(None, description="Geometric primitive (circle, square, triangle, rectangle, swatch)")
    occluded_by: List[str] = Field(default_factory=list, description="IDs or labels of objects occluding this object")
    visible_fraction: float = Field(1.0, description="Fraction of amodal area currently visible [0.0 - 1.0]")
    origin: str = Field("detector", description="Origin: detector | user | composition")

    # ================= USER-EDITABLE FIELDS (OVERRIDE LAYER) ==================
    user_label: Optional[str] = Field(None, description="User-assigned label correction")
    user_confidence: Optional[float] = Field(None, description="User-adjusted confidence")
    user_bbox: Optional[BoundingBox] = Field(None, description="User-modified bounding box")
    user_colors: Optional[List[str]] = Field(None, description="User-assigned colors")
    user_shape: Optional[str] = Field(None, description="User-corrected shape label")
    display_name: Optional[str] = Field(None, description="Custom display title")
    notes: Optional[str] = Field(None, description="User notes or annotation comments")

    # ================= AUDIT & SOURCE OF TRUTH FLAGS ==========================
    is_manually_added: bool = Field(False, description="True if object was added by user, not detector")
    is_manually_edited: bool = Field(False, description="True if user modified auto-detected fields")
    is_deleted: bool = Field(False, description="Soft-delete tombstone if removed by user")
    last_modified_at: Optional[str] = None

    # ================= RESOLVED EFFECTIVE PROPERTIES =========================
    @property
    def label(self) -> str:
        """Effective label: user override takes precedence, fallback to auto."""
        if self.user_label is not None:
            return self.user_label
        return self.auto_label or "unlabeled"

    @property
    def confidence(self) -> float:
        """Effective confidence: user override takes precedence, fallback to auto."""
        if self.user_confidence is not None:
            return self.user_confidence
        return self.auto_confidence or 1.0

    @property
    def bbox(self) -> Optional[BoundingBox]:
        """Effective bounding box."""
        if self.user_bbox is not None:
            return self.user_bbox
        return self.auto_bbox

    @property
    def colors(self) -> List[str]:
        """Effective colors list."""
        if self.user_colors is not None:
            return self.user_colors
        return self.auto_colors

    @property
    def primary_color(self) -> str:
        c = self.colors
        return c[0] if c else "unknown"

    @property
    def is_user_modified(self) -> bool:
        return self.is_manually_added or self.is_manually_edited

    def apply_edit(
        self,
        label: Optional[str] = None,
        confidence: Optional[float] = None,
        bbox: Optional[BoundingBox | tuple[int, int, int, int]] = None,
        colors: Optional[List[str]] = None,
        display_name: Optional[str] = None,
        notes: Optional[str] = None
    ) -> None:
        """Applies manual modifications while preserving original auto detection fields."""
        if label is not None:
            self.user_label = label
        if confidence is not None:
            self.user_confidence = confidence
        if bbox is not None:
            if isinstance(bbox, tuple):
                self.user_bbox = BoundingBox(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
            else:
                self.user_bbox = bbox
        if colors is not None:
            self.user_colors = colors
        if display_name is not None:
            self.display_name = display_name
        if notes is not None:
            self.notes = notes

        if not self.is_manually_added:
            self.is_manually_edited = True
        self.last_modified_at = datetime.now(timezone.utc).isoformat()


class CompositeSourceAsset(BaseModel):
    catalog_entry_id: str
    image_name: str
    object_id: Optional[str] = None
    object_label: Optional[str] = None
    placed_bbox: Optional[BoundingBox] = None
    z_index: int = 0
    scale: float = 1.0


class CompositeMetadata(BaseModel):
    prompt: str = Field(..., description="Natural language prompt used to generate composite")
    source_assets: List[CompositeSourceAsset] = Field(default_factory=list)
    canvas_dimensions: tuple[int, int] = (800, 600)
    layout_strategy: str = "auto_flow"
    generative_fill: bool = Field(False, description="Strict provenance proof: False means authentic catalog assets were used, never synthesized approximations")
    verification_status: Optional[str] = Field(None, description="verified | unverified | failed")
    verification_matches: List[Dict[str, Any]] = Field(default_factory=list)
    verification_iou: Optional[float] = None
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CatalogEntry(BaseModel):
    """
    Represents an image entry in the catalog with full metadata tracking.
    """
    id: str = Field(..., description="Unique entry ID / slug (e.g. 'apple')")
    image_name: str = Field(..., description="Display or original filename")
    file_path: str = Field(..., description="Local path to stored image file")
    width: int
    height: int
    format: str = "PNG"
    file_size_bytes: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Detection Lifecycle
    detection_status: str = Field(
        "pending",
        description="Status: pending | completed | failed | no_objects_found | manually_corrected"
    )
    objects_source: str = Field(
        "auto",
        description="Source of truth for objects: auto | manual | composition"
    )
    has_manual_corrections: bool = False
    detection_runs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Append-only history of detection executions and replaced object snapshots"
    )

    # Detected / Annotated Objects
    objects: List[DetectedObject] = Field(default_factory=list)

    # Composite Metadata (if this image was generated as a composite)
    is_composite: bool = False
    composite_metadata: Optional[CompositeMetadata] = None

    # User Notes / Tags
    tags: List[str] = Field(default_factory=list)
    user_notes: Optional[str] = None

    def get_active_objects(self) -> List[DetectedObject]:
        """Returns objects that are not marked as deleted (tombstone filter)."""
        return [obj for obj in self.objects if not obj.is_deleted]

    def add_manual_object(
        self,
        label: str,
        bbox: BoundingBox | tuple[int, int, int, int],
        colors: Optional[List[str]] = None,
        confidence: float = 1.0,
        notes: Optional[str] = None
    ) -> DetectedObject:
        """Adds a completely new object manually."""
        if isinstance(bbox, tuple):
            bbox_obj = BoundingBox(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
        else:
            bbox_obj = bbox

        new_obj = DetectedObject(
            user_label=label,
            user_bbox=bbox_obj,
            user_confidence=confidence,
            user_colors=colors or [],
            notes=notes,
            origin="user",
            is_manually_added=True,
            last_modified_at=datetime.now(timezone.utc).isoformat()
        )
        self.objects.append(new_obj)
        self.has_manual_corrections = True
        self.objects_source = "manual"
        self.detection_status = "manually_corrected"
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return new_obj

    def remove_object(self, object_id: str) -> bool:
        """Soft-deletes an object to preserve audit log (tombstone), marking as manual correction."""
        for obj in self.objects:
            if obj.id == object_id:
                obj.is_deleted = True
                obj.last_modified_at = datetime.now(timezone.utc).isoformat()
                self.has_manual_corrections = True
                self.objects_source = "manual"
                self.detection_status = "manually_corrected"
                self.updated_at = datetime.now(timezone.utc).isoformat()
                return True
        return False
