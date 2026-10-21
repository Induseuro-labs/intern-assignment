"""Persistent image catalog: SQLite metadata + image files on disk.

Field ownership (the auto / user split required by section 3):

  image   auto  : width, height, format, sha256, kind, created_at, provenance,
                  detection {status, backend, error, ran_at, runs}
          user  : display_name, description, tags
  object  auto  : origin, detector, auto_original, created_at
          user  : label, display_name, color, confidence, bbox, attributes
                  (initially filled by the detector; any user edit flips
                  edited_by_user and snapshots the detector's values into
                  auto_original so nothing is lost)

Source-of-truth rule: once an image has any user correction (object added,
edited or deleted) it is marked corrected. run_detection() then refuses to
touch it unless the caller passes force=True - i.e. the user explicitly asks
to rerun. keep_manual=True reruns detection but keeps the user's objects.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .detect import DetectionResult, default_detector

USER_OBJECT_FIELDS = {"label", "display_name", "color", "confidence", "bbox", "attributes"}
USER_IMAGE_FIELDS = {"display_name", "description", "tags"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    file TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    format TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    kind TEXT NOT NULL,                 -- 'upload' | 'composite'
    provenance TEXT,                    -- JSON, composites only
    created_at TEXT NOT NULL,
    detection_status TEXT NOT NULL,     -- pending | detected | no_objects | failed | skipped
    detection_backend TEXT,
    detection_error TEXT,
    detection_ran_at TEXT,
    detection_runs INTEGER NOT NULL DEFAULT 0,
    corrected_at TEXT,                  -- set on first user correction
    display_name TEXT,
    description TEXT,
    tags TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS objects (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    origin TEXT NOT NULL,               -- 'detector' | 'user' | 'composer'
    detector TEXT,
    label TEXT NOT NULL,
    display_name TEXT,
    color TEXT,
    confidence REAL,
    x INTEGER NOT NULL, y INTEGER NOT NULL, w INTEGER NOT NULL, h INTEGER NOT NULL,
    attributes TEXT NOT NULL DEFAULT '{}',
    edited_by_user INTEGER NOT NULL DEFAULT 0,
    auto_original TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class CatalogError(Exception):
    pass


class ManualCorrectionsExist(CatalogError):
    """Raised when detection would overwrite user corrections without explicit consent."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Catalog:
    def __init__(self, root: str | Path, detector=None):
        self.root = Path(root)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "catalog.db")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(_SCHEMA)
        self._detector = detector

    @property
    def detector(self):
        if self._detector is None:
            self._detector = default_detector()
        return self._detector

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------ images
    def add_image(self, source, name: str, description: str = "", tags: dict | None = None,
                  detect: bool = True, kind: str = "upload", provenance: dict | None = None) -> dict:
        img = source if isinstance(source, Image.Image) else Image.open(source)
        img.load()
        image_id = uuid.uuid4().hex[:12]
        file = self.root / "images" / f"{image_id}.png"
        if isinstance(source, (str, Path)) and str(source).lower().endswith(".png"):
            shutil.copyfile(source, file)
        else:
            img.save(file, format="PNG")
        sha = hashlib.sha256(file.read_bytes()).hexdigest()
        with self.db:
            try:
                self.db.execute(
                    "INSERT INTO images (id, name, file, width, height, format, sha256, kind, provenance,"
                    " created_at, detection_status, description, tags) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (image_id, name, file.name, img.width, img.height, img.format or "PNG", sha, kind,
                     json.dumps(provenance) if provenance else None, _now(),
                     "pending" if detect else "skipped", description, json.dumps(tags or {})))
            except sqlite3.IntegrityError:
                file.unlink(missing_ok=True)
                raise CatalogError(f"an image named {name!r} already exists")
        if detect:
            self.run_detection(image_id)
        return self.get(image_id)

    def _row(self, ref: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM images WHERE id = ? OR name = ?", (ref, ref)).fetchone()
        if row is None:
            raise CatalogError(f"no image {ref!r}")
        return row

    def get(self, ref: str) -> dict:
        row = self._row(ref)
        objects = self.db.execute("SELECT * FROM objects WHERE image_id = ? ORDER BY x, y", (row["id"],))
        return {
            "id": row["id"],
            "name": row["name"],
            "auto": {
                "width": row["width"], "height": row["height"], "format": row["format"],
                "sha256": row["sha256"], "kind": row["kind"], "created_at": row["created_at"],
                "file": str(self.root / "images" / row["file"]),
                "provenance": json.loads(row["provenance"]) if row["provenance"] else None,
                "detection": {"status": row["detection_status"], "backend": row["detection_backend"],
                              "error": row["detection_error"], "ran_at": row["detection_ran_at"],
                              "runs": row["detection_runs"]},
            },
            "user": {"display_name": row["display_name"] or row["name"], "description": row["description"],
                     "tags": json.loads(row["tags"])},
            "corrected": row["corrected_at"] is not None,
            "objects": [self._object_dict(o) for o in objects],
        }

    def list(self) -> list[dict]:
        return [self.get(r["id"]) for r in self.db.execute("SELECT id FROM images ORDER BY created_at, name")]

    def update_image(self, ref: str, **fields) -> dict:
        bad = set(fields) - USER_IMAGE_FIELDS
        if bad:
            raise CatalogError(f"not user-editable: {sorted(bad)}")
        row = self._row(ref)
        if "tags" in fields:
            fields["tags"] = json.dumps(fields["tags"])
        with self.db:
            for k, v in fields.items():
                self.db.execute(f"UPDATE images SET {k} = ? WHERE id = ?", (v, row["id"]))
        return self.get(row["id"])

    def load_pil(self, ref: str) -> Image.Image:
        return Image.open(self.root / "images" / self._row(ref)["file"]).convert("RGB")

    # --------------------------------------------------------------- detection
    def run_detection(self, ref: str, force: bool = False, keep_manual: bool = False) -> dict:
        """Run (or rerun) detection.

        If the user has corrected this image, refuse unless force=True. With
        keep_manual=True, user-added/edited objects survive and only untouched
        detector output is replaced.
        """
        row = self._row(ref)
        if row["corrected_at"] and not force:
            raise ManualCorrectionsExist(
                f"{row['name']!r} has manual corrections; pass force=True (optionally keep_manual=True) "
                "to explicitly rerun detection")
        try:
            result: DetectionResult = self.detector.detect(self.load_pil(row["id"]))
        except Exception as exc:  # a broken backend must not break ingestion
            result = DetectionResult("failed", getattr(self.detector, "name", "unknown"), error=str(exc))
        now = _now()
        with self.db:
            if keep_manual:
                self.db.execute("DELETE FROM objects WHERE image_id = ? AND origin = 'detector' "
                                "AND edited_by_user = 0", (row["id"],))
            else:
                self.db.execute("DELETE FROM objects WHERE image_id = ?", (row["id"],))
                self.db.execute("UPDATE images SET corrected_at = NULL WHERE id = ?", (row["id"],))
            for det in result.objects:
                self._insert_object(row["id"], "detector", result.backend, det.label, det.color,
                                    det.confidence, det.bbox, det.attributes, now)
            self.db.execute(
                "UPDATE images SET detection_status = ?, detection_backend = ?, detection_error = ?,"
                " detection_ran_at = ?, detection_runs = detection_runs + 1 WHERE id = ?",
                (result.status, result.backend, result.error, now, row["id"]))
        return self.get(row["id"])

    def redetect_all(self) -> dict[str, str]:
        """Batch reprocessing (e.g. after a detector upgrade). Corrected images are skipped."""
        outcome = {}
        for r in self.db.execute("SELECT id, name, corrected_at FROM images WHERE kind = 'upload'").fetchall():
            if r["corrected_at"]:
                outcome[r["name"]] = "skipped (manual corrections)"
            else:
                outcome[r["name"]] = self.run_detection(r["id"])["auto"]["detection"]["status"]
        return outcome

    # ----------------------------------------------------------------- objects
    def _insert_object(self, image_id, origin, detector, label, color, confidence, bbox, attributes,
                       now=None, display_name=None) -> str:
        oid = uuid.uuid4().hex[:12]
        now = now or _now()
        x, y, w, h = (int(v) for v in bbox)
        self.db.execute(
            "INSERT INTO objects (id, image_id, origin, detector, label, display_name, color, confidence,"
            " x, y, w, h, attributes, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (oid, image_id, origin, detector, label, display_name, color, confidence, x, y, w, h,
             json.dumps(attributes or {}), now, now))
        return oid

    @staticmethod
    def _object_dict(o: sqlite3.Row) -> dict:
        return {
            "id": o["id"], "origin": o["origin"], "detector": o["detector"],
            "edited_by_user": bool(o["edited_by_user"]),
            "label": o["label"], "display_name": o["display_name"] or (
                f"{o['color']} {o['label']}" if o["color"] else o["label"]),
            "color": o["color"], "confidence": o["confidence"],
            "bbox": {"x": o["x"], "y": o["y"], "w": o["w"], "h": o["h"]},
            "attributes": json.loads(o["attributes"]),
            "auto_original": json.loads(o["auto_original"]) if o["auto_original"] else None,
        }

    def _check_bbox(self, image_id: str, bbox) -> tuple[int, int, int, int]:
        if isinstance(bbox, dict):
            bbox = (bbox["x"], bbox["y"], bbox["w"], bbox["h"])
        x, y, w, h = (int(v) for v in bbox)
        row = self._row(image_id)
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > row["width"] or y + h > row["height"]:
            raise CatalogError(f"bbox {bbox} outside image {row['width']}x{row['height']}")
        return x, y, w, h

    def _mark_corrected(self, image_id: str) -> None:
        self.db.execute("UPDATE images SET corrected_at = COALESCE(corrected_at, ?) WHERE id = ?",
                        (_now(), image_id))

    def add_object(self, ref: str, label: str, bbox, color: str | None = None, confidence: float = 1.0,
                   display_name: str | None = None, attributes: dict | None = None) -> dict:
        row = self._row(ref)
        bbox = self._check_bbox(row["id"], bbox)
        with self.db:
            oid = self._insert_object(row["id"], "user", None, label, color, confidence, bbox,
                                      attributes, display_name=display_name)
            self._mark_corrected(row["id"])
        return self.get_object(oid)

    def get_object(self, object_id: str) -> dict:
        o = self.db.execute("SELECT * FROM objects WHERE id = ?", (object_id,)).fetchone()
        if o is None:
            raise CatalogError(f"no object {object_id!r}")
        return self._object_dict(o)

    def update_object(self, object_id: str, **fields) -> dict:
        bad = set(fields) - USER_OBJECT_FIELDS
        if bad:
            raise CatalogError(f"not user-editable: {sorted(bad)}")
        o = self.db.execute("SELECT * FROM objects WHERE id = ?", (object_id,)).fetchone()
        if o is None:
            raise CatalogError(f"no object {object_id!r}")
        cols = {}
        for k, v in fields.items():
            if k == "bbox":
                cols.update(zip("xywh", self._check_bbox(o["image_id"], v)))
            elif k == "attributes":
                cols[k] = json.dumps(v)
            elif k == "confidence" and not 0 <= float(v) <= 1:
                raise CatalogError("confidence must be between 0 and 1")
            else:
                cols[k] = v
        with self.db:
            if o["origin"] == "detector" and not o["edited_by_user"]:
                snapshot = {k: o[k] for k in ("label", "color", "confidence", "x", "y", "w", "h")}
                cols["auto_original"] = json.dumps(snapshot)
            cols["edited_by_user"] = 1
            cols["updated_at"] = _now()
            self.db.execute(f"UPDATE objects SET {', '.join(f'{k} = ?' for k in cols)} WHERE id = ?",
                            (*cols.values(), object_id))
            self._mark_corrected(o["image_id"])
        return self.get_object(object_id)

    def delete_object(self, object_id: str) -> None:
        o = self.db.execute("SELECT image_id FROM objects WHERE id = ?", (object_id,)).fetchone()
        if o is None:
            raise CatalogError(f"no object {object_id!r}")
        with self.db:
            self.db.execute("DELETE FROM objects WHERE id = ?", (object_id,))
            self._mark_corrected(o["image_id"])
