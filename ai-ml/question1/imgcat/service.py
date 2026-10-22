"""Catalog service: the single entry point used by the CLI, API, scenarios and tests."""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .detection import Detector, category_of
from .store import USER_OBJECT_FIELDS, CatalogImage, CatalogObject, CatalogStore


class ProtectedObjectsError(RuntimeError):
    """Raised when detection would overwrite user corrections without an explicit request."""


class CatalogService:
    def __init__(self, root: str | Path, detector: Detector | None = None):
        self.store = CatalogStore(root)
        self.masks = self.store.root / "masks"
        self.masks.mkdir(exist_ok=True)
        self._detector = detector

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            self._detector = Detector()
        return self._detector

    # ------------------------------------------------------------ images
    def add_image(self, path: str | Path, display_name: str | None = None, detect: bool = True,
                  tags: list[str] | None = None) -> CatalogImage:
        image_id = self.store.add_image_file(path, display_name, tags=tags)
        if detect:
            self.run_detection(image_id)
        return self.store.get_image(image_id)

    def get(self, image_id_or_name: str) -> CatalogImage:
        try:
            return self.store.get_image(image_id_or_name)
        except KeyError:
            img = self.store.find_image(image_id_or_name)
            if img is None:
                raise
            return img

    def load_rgb(self, image_id: str) -> np.ndarray:
        with Image.open(self.store.image_path(image_id)) as im:
            return np.array(im.convert("RGB"))

    # --------------------------------------------------------- detection
    def run_detection(self, image_id: str, force: bool = False) -> CatalogImage:
        """Detect objects. Refuses to touch user-corrected (or composed) objects unless ``force``."""
        img = self.store.get_image(image_id)
        if img.objects_source in ("manual", "composition") and not force:
            raise ProtectedObjectsError(
                f"image {image_id} has {img.objects_source} objects; they are the source of truth. "
                "Pass force=True to explicitly re-run detection and replace them.")
        replaced = [o.to_dict() for o in img.objects] if img.objects else None
        try:
            rgb = self.load_rgb(image_id)
            result = self.detector.detect(rgb)
        except Exception as exc:  # detection failure must not destroy existing objects
            self.store.set_image_auto(image_id, detection_status="failed",
                                      detection_error=f"{type(exc).__name__}: {exc}",
                                      detector_version=self.detector.version)
            self.store.log_run(image_id, "failed", self.detector.version, force, None, None,
                               traceback.format_exc(limit=3))
            return self.store.get_image(image_id)

        for o in img.objects:
            (self.masks / f"{o.id}.png").unlink(missing_ok=True)
        self.store.replace_auto_objects(image_id, [])
        autos = []
        for det in result.objects:
            oid = self.store.insert_object(image_id, "detector", auto=det.auto)
            if det.mask is not None:
                Image.fromarray(det.mask.astype(np.uint8) * 255).save(self.masks / f"{oid}.png")
                det.auto["mask"] = f"{oid}.png"
                self.store.db.execute("UPDATE objects SET auto=json_set(auto, '$.mask', ?) WHERE id=?",
                                      (f"{oid}.png", oid))
            autos.append(det.auto)
        self.store.db.commit()
        status = "detected" if result.objects else "no_objects"
        self.store.set_image_auto(image_id, detection_status=status,
                                  detection_error="; ".join(result.notes) or None,
                                  detector_version=self.detector.version,
                                  detected_at=self._now(), style=result.style, objects_source="auto")
        self.store.log_run(image_id, status, self.detector.version, force, autos, replaced, None)
        return self.store.get_image(image_id)

    # ------------------------------------------------------ corrections
    def _validate(self, image_id: str, fields: dict) -> dict:
        bad = set(fields) - set(USER_OBJECT_FIELDS)
        if bad:
            raise ValueError(f"not user-editable: {sorted(bad)}; editable: {USER_OBJECT_FIELDS}")
        img = self.store.get_image(image_id)
        if "bbox" in fields:
            b = fields["bbox"]
            b = {k: int(b[k]) for k in ("x", "y", "width", "height")}
            if b["width"] <= 0 or b["height"] <= 0 or b["x"] < 0 or b["y"] < 0 \
                    or b["x"] + b["width"] > img.width or b["y"] + b["height"] > img.height:
                raise ValueError(f"bbox {b} outside image {img.width}x{img.height}")
            fields["bbox"] = b
        if "confidence" in fields and fields["confidence"] is not None:
            c = float(fields["confidence"])
            if not 0 <= c <= 1:
                raise ValueError("confidence must be within [0, 1]")
            fields["confidence"] = c
        if "name" in fields and not str(fields["name"]).strip():
            raise ValueError("name cannot be empty")
        return fields

    def _mark_manual(self, image_id: str) -> None:
        img = self.store.get_image(image_id)
        if img.objects_source != "composition":
            self.store.set_image_auto(image_id, objects_source="manual")

    def add_object(self, image_id: str, name: str, bbox: dict, color: str | None = None,
                   confidence: float = 1.0, **extra: Any) -> CatalogObject:
        fields = self._validate(image_id, {"name": name, "bbox": bbox, "color": color,
                                           "confidence": confidence, **extra})
        fields.setdefault("category", category_of(name))
        oid = self.store.insert_object(image_id, "user", auto=None,
                                       user={k: v for k, v in fields.items() if v is not None})
        self._mark_manual(image_id)
        return self.store.get_object(oid)

    def update_object(self, object_id: str, **fields: Any) -> CatalogObject:
        obj = self.store.get_object(object_id)
        fields = self._validate(obj.image_id, fields)
        user = {**obj.user, **fields}
        if "name" in fields and "category" not in fields:
            derived = category_of(fields["name"])
            if derived != "object":  # a known label implies its category; unknown labels keep the old one
                user["category"] = derived
            elif user.get("category") == category_of(obj.user.get("name", "")):
                user.pop("category", None)
        self.store.set_object_user(object_id, user)
        self._mark_manual(obj.image_id)
        return self.store.get_object(object_id)

    def revert_field(self, object_id: str, field: str) -> CatalogObject:
        """Drop a user override so the detector's value shows through again."""
        obj = self.store.get_object(object_id)
        user = {k: v for k, v in obj.user.items() if k != field}
        if field == "name" and "name" in obj.user and user.get("category") == category_of(obj.user["name"]):
            user.pop("category", None)  # the category was derived from the reverted name
        self.store.set_object_user(object_id, user)
        return self.store.get_object(object_id)

    def delete_object(self, object_id: str) -> None:
        """User-created objects are removed; detector objects are tombstoned so they stay removed."""
        obj = self.store.get_object(object_id)
        if obj.origin == "detector":
            self.store.set_object_user(object_id, {**obj.user, "deleted": True})
        else:
            self.store.db.execute("DELETE FROM objects WHERE id=?", (object_id,))
            self.store.db.commit()
        self._mark_manual(obj.image_id)

    def restore_object(self, object_id: str) -> CatalogObject:
        obj = self.store.get_object(object_id)
        self.store.set_object_user(object_id, {k: v for k, v in obj.user.items() if k != "deleted"})
        return self.store.get_object(object_id)

    def object_mask(self, obj: CatalogObject) -> np.ndarray | None:
        """Stored pixel mask, valid only while the user has not moved the box."""
        if obj.auto and obj.auto.get("mask") and "bbox" not in obj.user:
            p = self.masks / obj.auto["mask"]
            if p.exists():
                return np.array(Image.open(p)) > 127
        return None

    # --------------------------------------------------------- composite
    def compose(self, instruction: str, display_name: str | None = None, use_llm: bool | None = None,
                verify: bool = False) -> CatalogImage:
        from .composite import compose

        return compose(self, instruction, display_name=display_name, use_llm=use_llm, verify=verify)

    @staticmethod
    def _now() -> str:
        from .store import now

        return now()
