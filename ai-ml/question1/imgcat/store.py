"""SQLite-backed image catalog.

Field ownership is explicit in the schema:

* ``images``  — auto columns (width, height, sha256, detection_*, kind,
  provenance) are written only by the system; ``display_name``,
  ``description`` and ``tags`` are user-editable.
* ``objects`` — every detected object keeps the detector's output in the
  read-only ``auto`` JSON column. User corrections live in the ``user`` JSON
  column as overrides. The value the rest of the system sees is
  ``effective = {**auto, **user}``. A user-created object has ``auto = NULL``.
* ``detection_runs`` — append-only history of every detection run, including
  the object set a forced re-run replaced, so nothing a user did is lost.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

AUTO_OBJECT_FIELDS = ("name", "display_name", "confidence", "bbox", "color", "colors", "category", "shape",
                      "attributes", "detector")
USER_OBJECT_FIELDS = ("name", "display_name", "color", "bbox", "confidence", "category",
                      "attributes", "notes", "deleted")
USER_IMAGE_FIELDS = ("display_name", "description", "tags")

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    -- auto (system-owned)
    file_name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    format TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'original',          -- original | composite
    style TEXT,                                     -- photo | illustration (auto-estimated)
    provenance TEXT,                                -- JSON, composites only
    detection_status TEXT NOT NULL DEFAULT 'pending',
    -- pending | detected | no_objects | failed | composed
    detection_error TEXT,
    detector_version TEXT,
    detected_at TEXT,
    objects_source TEXT NOT NULL DEFAULT 'auto',    -- auto | manual (user corrected)
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    -- user-editable
    display_name TEXT,
    description TEXT,
    tags TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS objects (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    origin TEXT NOT NULL,          -- detector | user | composition
    auto TEXT,                     -- JSON, detector output (never edited)
    user TEXT NOT NULL DEFAULT '{}',  -- JSON, user overrides
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS objects_image ON objects(image_id);
CREATE TABLE IF NOT EXISTS detection_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    started_at TEXT NOT NULL,
    status TEXT NOT NULL,
    detector_version TEXT,
    forced INTEGER NOT NULL DEFAULT 0,
    result TEXT,                   -- JSON list of auto objects produced
    replaced TEXT,                 -- JSON snapshot of objects replaced by a forced run
    error TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CatalogObject:
    id: str
    image_id: str
    origin: str
    auto: dict | None
    user: dict
    created_at: str
    updated_at: str

    @property
    def deleted(self) -> bool:
        return bool(self.user.get("deleted"))

    @property
    def effective(self) -> dict:
        eff = {k: None for k in AUTO_OBJECT_FIELDS}
        eff.update(self.auto or {})
        eff.update({k: v for k, v in self.user.items() if k != "deleted"})
        if "name" in self.user and "display_name" not in self.user:
            eff["display_name"] = None  # the auto display name described the old label
        eff["display_name"] = eff.get("display_name") or eff.get("name")
        return eff

    @property
    def edited_fields(self) -> list[str]:
        return sorted(k for k in self.user if k != "deleted")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "origin": self.origin,
            "deleted": self.deleted,
            "effective": self.effective,
            "auto": self.auto,
            "user": self.user,
            "edited_fields": self.edited_fields,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class CatalogImage:
    row: dict
    objects: list[CatalogObject] = field(default_factory=list)

    def __getattr__(self, item):
        try:
            return self.row[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    @property
    def live_objects(self) -> list[CatalogObject]:
        return [o for o in self.objects if not o.deleted]

    def to_dict(self, include_deleted: bool = False) -> dict:
        r = dict(self.row)
        r["tags"] = json.loads(r["tags"] or "[]")
        r["provenance"] = json.loads(r["provenance"]) if r["provenance"] else None
        objs = self.objects if include_deleted else self.live_objects
        return {
            "id": r["id"],
            "auto": {k: r[k] for k in ("file_name", "sha256", "width", "height", "format", "kind",
                                       "style", "provenance", "detection_status", "detection_error",
                                       "detector_version", "detected_at", "objects_source",
                                       "created_at", "updated_at")},
            "user": {k: r[k] for k in USER_IMAGE_FIELDS},
            "objects": [o.to_dict() for o in objs],
        }


class CatalogStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.files = self.root / "images"
        self.files.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "catalog.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------- images
    def add_image_file(self, path: str | Path, display_name: str | None = None,
                       kind: str = "original", provenance: dict | None = None,
                       tags: list[str] | None = None) -> str:
        path = Path(path)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        with Image.open(path) as im:
            width, height, fmt = im.width, im.height, (im.format or path.suffix[1:]).upper()
        image_id = uuid.uuid4().hex[:12]
        file_name = f"{image_id}{path.suffix.lower()}"
        shutil.copyfile(path, self.files / file_name)
        ts = now()
        self.db.execute(
            "INSERT INTO images (id, file_name, sha256, width, height, format, kind, provenance,"
            " detection_status, created_at, updated_at, display_name, tags)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (image_id, file_name, digest, width, height, fmt, kind,
             json.dumps(provenance) if provenance else None,
             "composed" if kind == "composite" else "pending", ts, ts,
             display_name or path.stem, json.dumps(tags or [])),
        )
        self.db.commit()
        return image_id

    def image_path(self, image_id: str) -> Path:
        return self.files / self._image_row(image_id)["file_name"]

    def _image_row(self, image_id: str) -> dict:
        row = self.db.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        if row is None:
            raise KeyError(f"no image {image_id}")
        return dict(row)

    def get_image(self, image_id: str) -> CatalogImage:
        return CatalogImage(self._image_row(image_id), self.list_objects(image_id))

    def find_image(self, name: str) -> CatalogImage | None:
        row = self.db.execute(
            "SELECT id FROM images WHERE display_name=? ORDER BY created_at DESC LIMIT 1", (name,)
        ).fetchone()
        return self.get_image(row["id"]) if row else None

    def list_images(self) -> list[CatalogImage]:
        rows = self.db.execute("SELECT id FROM images ORDER BY created_at, rowid").fetchall()
        return [self.get_image(r["id"]) for r in rows]

    def update_image_user_fields(self, image_id: str, **fields: Any) -> None:
        bad = set(fields) - set(USER_IMAGE_FIELDS)
        if bad:
            raise ValueError(f"not user-editable: {sorted(bad)}")
        for k, v in fields.items():
            if k == "tags":
                v = json.dumps(v)
            self.db.execute(f"UPDATE images SET {k}=?, updated_at=? WHERE id=?", (v, now(), image_id))
        self.db.commit()

    def set_image_auto(self, image_id: str, **fields: Any) -> None:
        """System-only setter for auto columns."""
        for k, v in fields.items():
            if k == "provenance" and v is not None:
                v = json.dumps(v)
            self.db.execute(f"UPDATE images SET {k}=?, updated_at=? WHERE id=?", (v, now(), image_id))
        self.db.commit()

    def delete_image(self, image_id: str) -> None:
        path = self.image_path(image_id)
        self.db.execute("DELETE FROM images WHERE id=?", (image_id,))
        self.db.commit()
        path.unlink(missing_ok=True)

    # ------------------------------------------------------------ objects
    def list_objects(self, image_id: str) -> list[CatalogObject]:
        rows = self.db.execute(
            "SELECT * FROM objects WHERE image_id=? ORDER BY created_at, rowid", (image_id,)
        ).fetchall()
        return [self._obj(r) for r in rows]

    def get_object(self, object_id: str) -> CatalogObject:
        row = self.db.execute("SELECT * FROM objects WHERE id=?", (object_id,)).fetchone()
        if row is None:
            raise KeyError(f"no object {object_id}")
        return self._obj(row)

    @staticmethod
    def _obj(r) -> CatalogObject:
        return CatalogObject(r["id"], r["image_id"], r["origin"],
                             json.loads(r["auto"]) if r["auto"] else None,
                             json.loads(r["user"]), r["created_at"], r["updated_at"])

    def insert_object(self, image_id: str, origin: str, auto: dict | None = None,
                      user: dict | None = None) -> str:
        oid = uuid.uuid4().hex[:12]
        ts = now()
        self.db.execute(
            "INSERT INTO objects (id, image_id, origin, auto, user, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (oid, image_id, origin, json.dumps(auto) if auto is not None else None,
             json.dumps(user or {}), ts, ts),
        )
        self.db.commit()
        return oid

    def set_object_user(self, object_id: str, user: dict) -> None:
        self.db.execute("UPDATE objects SET user=?, updated_at=? WHERE id=?",
                        (json.dumps(user), now(), object_id))
        self.db.commit()

    def replace_auto_objects(self, image_id: str, objects: list[dict], origin: str = "detector") -> None:
        self.db.execute("DELETE FROM objects WHERE image_id=?", (image_id,))
        self.db.commit()
        for auto in objects:
            self.insert_object(image_id, origin, auto=auto)

    # ------------------------------------------------------ detection log
    def log_run(self, image_id: str, status: str, detector_version: str | None, forced: bool,
                result: list[dict] | None, replaced: list[dict] | None, error: str | None) -> None:
        self.db.execute(
            "INSERT INTO detection_runs (image_id, started_at, status, detector_version, forced,"
            " result, replaced, error) VALUES (?,?,?,?,?,?,?,?)",
            (image_id, now(), status, detector_version, int(forced),
             json.dumps(result) if result is not None else None,
             json.dumps(replaced) if replaced is not None else None, error),
        )
        self.db.commit()

    def runs(self, image_id: str) -> list[dict]:
        rows = self.db.execute("SELECT * FROM detection_runs WHERE image_id=? ORDER BY id",
                               (image_id,)).fetchall()
        return [dict(r) for r in rows]
