"""
Catalog Storage and Persistence Engine.
Handles JSON persistence, retrieval, updates, and manual correction tracking.
"""

from __future__ import annotations
import os
import json
import shutil
from typing import Dict, List, Optional
from datetime import datetime, timezone

from .models import CatalogEntry, DetectedObject, BoundingBox

DEFAULT_STORAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON_PATH = os.path.join(DEFAULT_STORAGE_DIR, "catalog_data.json")

def resolve_catalog_file_path(path: str, image_name: Optional[str] = None) -> str:
    """
    Cross-platform path resolver for catalog image assets.
    Handles differences between Windows development paths and Linux/Docker production paths.
    """
    if path and os.path.exists(path):
        return os.path.abspath(path)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    comp_dir = os.path.join(base_dir, "catalog", "composites")
    fname = image_name or (os.path.basename(path.replace("\\", "/")) if path else "")

    if fname:
        candidates = [
            os.path.join(data_dir, fname),
            os.path.join(data_dir, "uploaded", fname),
            os.path.join(comp_dir, fname),
            os.path.join(base_dir, fname),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return os.path.abspath(cand)

    return path


class ImageCatalog:
    """
    Catalog repository for storing and managing reusable image assets and metadata.
    """
    def __init__(self, storage_path: str = DEFAULT_JSON_PATH):
        self.storage_path = storage_path
        self._entries: Dict[str, CatalogEntry] = {}
        self.load()

    def load(self) -> None:
        """Loads catalog entries from JSON file if it exists."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    self._entries = {}
                    for item in raw_data:
                        entry = CatalogEntry.model_validate(item)
                        entry.file_path = resolve_catalog_file_path(entry.file_path, entry.image_name)
                        self._entries[entry.id] = entry
            except Exception as e:
                print(f"[Warning] Failed to load catalog from {self.storage_path}: {e}")
                self._entries = {}
        else:
            self._entries = {}

    def save(self) -> None:
        """Saves all catalog entries to JSON file."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = [entry.model_dump() for entry in self._entries.values()]
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_entry(self, entry: CatalogEntry) -> CatalogEntry:
        """Adds or updates an entry and saves to disk."""
        entry.file_path = resolve_catalog_file_path(entry.file_path, entry.image_name)
        self._entries[entry.id] = entry
        self.save()
        return entry

    def get_entry(self, entry_id_or_name: str) -> Optional[CatalogEntry]:
        """Looks up entry by ID or image name."""
        entry = self._entries.get(entry_id_or_name)
        if not entry:
            for e in self._entries.values():
                if e.image_name == entry_id_or_name:
                    entry = e
                    break
        if entry and not os.path.exists(entry.file_path):
            entry.file_path = resolve_catalog_file_path(entry.file_path, entry.image_name)
        return entry

    def list_entries(self) -> List[CatalogEntry]:
        """Returns all entries."""
        for entry in self._entries.values():
            if not os.path.exists(entry.file_path):
                entry.file_path = resolve_catalog_file_path(entry.file_path, entry.image_name)
        return list(self._entries.values())

    def delete_entry(self, entry_id: str) -> bool:
        """Deletes an entry from catalog."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            self.save()
            return True
        return False

    def add_manual_object(
        self,
        entry_id: str,
        label: str,
        bbox: tuple[int, int, int, int] | BoundingBox,
        colors: Optional[List[str]] = None,
        confidence: float = 1.0,
        notes: Optional[str] = None
    ) -> Optional[DetectedObject]:
        """Manually adds an object to an entry and persists."""
        entry = self.get_entry(entry_id)
        if not entry:
            return None
        obj = entry.add_manual_object(label, bbox, colors, confidence, notes)
        self.save()
        return obj

    def edit_object(
        self,
        entry_id: str,
        object_id: str,
        label: Optional[str] = None,
        confidence: Optional[float] = None,
        bbox: Optional[tuple[int, int, int, int] | BoundingBox] = None,
        colors: Optional[List[str]] = None,
        display_name: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Optional[DetectedObject]:
        """Edits an existing detected or manual object and persists."""
        entry = self.get_entry(entry_id)
        if not entry:
            return None
        for obj in entry.objects:
            if obj.id == object_id:
                obj.apply_edit(label, confidence, bbox, colors, display_name, notes)
                entry.has_manual_corrections = True
                entry.objects_source = "manual"
                entry.detection_status = "manually_corrected"
                entry.updated_at = datetime.now(timezone.utc).isoformat()
                self.save()
                return obj
        return None

    def remove_object(self, entry_id: str, object_id: str) -> bool:
        """Removes an object from an entry (tombstone) and persists."""
        entry = self.get_entry(entry_id)
        if not entry:
            return False
        result = entry.remove_object(object_id)
        if result:
            self.save()
        return result

    def update_detection(
        self,
        entry_id: str,
        detected_objects: List[DetectedObject],
        detector_model: str = "HybridObjectDetector",
        force: bool = False
    ) -> CatalogEntry:
        """
        Updates detector output on an image entry.
        Strictly enforces manual source of truth:
        If manual corrections exist and force=False, raises ProtectedObjectsError.
        If force=True, archives previous user corrections into detection_runs snapshot.
        """
        from .models import ProtectedObjectsError
        entry = self.get_entry(entry_id)
        if not entry:
            raise KeyError(f"Catalog entry '{entry_id}' not found.")

        if (entry.objects_source == "manual" or entry.has_manual_corrections) and not force:
            raise ProtectedObjectsError(
                f"Conflict: Entry '{entry_id}' contains manual corrections which are the source of truth. "
                "Detection cannot overwrite user corrections without force=True.",
                entry_id=entry.id,
                manual_count=len(entry.get_active_objects())
            )

        # Forced run or initial run: archive replaced objects if any
        if entry.objects:
            entry.detection_runs.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "detector_model": detector_model,
                "forced": force,
                "replaced_snapshot": [obj.model_dump() for obj in entry.objects],
                "new_object_count": len(detected_objects)
            })

        entry.objects = detected_objects
        entry.objects_source = "auto"
        entry.has_manual_corrections = False
        entry.detection_status = "completed" if detected_objects else "no_objects_found"
        entry.updated_at = datetime.now(timezone.utc).isoformat()
        self.save()
        return entry

    def clear(self) -> None:
        """Clears all entries."""
        self._entries.clear()
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
