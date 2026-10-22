"""Question collection in SQLite with free-form key–value tags.

Tags live in their own table (question_id, key, value), so any new key or
value can appear at any time without a schema change, and a question can
carry several values for one key. Tags are for organising and filtering only;
nothing in duplicate prevention or detection compares tags.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .text import exact_key, normalize

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    normalized TEXT NOT NULL,
    exact_key TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'human',   -- human | llm
    status TEXT NOT NULL DEFAULT 'active',  -- active | needs_review
    duplicate_of TEXT,                      -- set when stored despite a match (review queue)
    detection TEXT,                         -- JSON verdict from the independent detector
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS questions_exact ON questions(exact_key);
CREATE TABLE IF NOT EXISTS question_tags (
    question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (question_id, key, value)
);
CREATE INDEX IF NOT EXISTS tags_kv ON question_tags(key, value);
CREATE TABLE IF NOT EXISTS embeddings (
    question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (question_id, model)
);
"""


@dataclass
class Question:
    id: str
    text: str
    tags: dict[str, list[str]] = field(default_factory=dict)
    source: str = "human"
    status: str = "active"
    duplicate_of: str | None = None
    detection: dict | None = None


class QuestionStore:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)

    def add(self, text: str, tags: dict | None = None, qid: str | None = None, source: str = "human",
            status: str = "active", duplicate_of: str | None = None, detection: dict | None = None) -> Question:
        if not text.strip():
            raise ValueError("empty question")
        qid = qid or f"q{self.count() + 1:05d}"
        self.db.execute(
            "INSERT INTO questions (id, text, normalized, exact_key, source, status, duplicate_of, detection, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (qid, text.strip(), normalize(text), exact_key(text), source, status, duplicate_of,
             json.dumps(detection) if detection else None, datetime.now(timezone.utc).isoformat(timespec="seconds")))
        self.set_tags(qid, tags or {}, commit=False)
        self.db.commit()
        return self.get(qid)

    def set_tags(self, qid: str, tags: dict, commit: bool = True) -> None:
        for key, values in tags.items():
            for v in values if isinstance(values, (list, tuple, set)) else [values]:
                self.db.execute("INSERT OR IGNORE INTO question_tags VALUES (?,?,?)", (qid, str(key), str(v)))
        if commit:
            self.db.commit()

    def get(self, qid: str) -> Question:
        r = self.db.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
        if r is None:
            raise KeyError(qid)
        return self._q(r)

    def _q(self, r) -> Question:
        tags: dict[str, list[str]] = {}
        for t in self.db.execute("SELECT key, value FROM question_tags WHERE question_id=? ORDER BY key, value",
                                 (r["id"],)):
            tags.setdefault(t["key"], []).append(t["value"])
        return Question(r["id"], r["text"], tags, r["source"], r["status"], r["duplicate_of"],
                        json.loads(r["detection"]) if r["detection"] else None)

    def all(self, tags: dict | None = None, include_review: bool = True) -> list[Question]:
        """All questions, optionally filtered by tags (every key must match one of its values)."""
        sql, args = "SELECT * FROM questions q WHERE 1=1", []
        if not include_review:
            sql += " AND status='active'"
        for key, values in (tags or {}).items():
            values = values if isinstance(values, (list, tuple, set)) else [values]
            sql += (f" AND EXISTS (SELECT 1 FROM question_tags t WHERE t.question_id=q.id AND t.key=? "
                    f"AND t.value IN ({','.join('?' * len(values))}))")
            args += [key, *map(str, values)]
        return [self._q(r) for r in self.db.execute(sql + " ORDER BY rowid", args)]

    def by_exact_key(self, key: str) -> list[str]:
        return [r["id"] for r in self.db.execute("SELECT id FROM questions WHERE exact_key=?", (key,))]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM questions").fetchone()[0]

    def tag_keys(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in self.db.execute("SELECT DISTINCT key, value FROM question_tags ORDER BY key, value"):
            out.setdefault(r["key"], []).append(r["value"])
        return out

    # embedding cache
    def get_vectors(self, model: str) -> dict[str, bytes]:
        return {r["question_id"]: r["vector"] for r in
                self.db.execute("SELECT question_id, vector FROM embeddings WHERE model=?", (model,))}

    def put_vectors(self, model: str, items: dict[str, bytes]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO embeddings VALUES (?,?,?)",
                            [(qid, model, v) for qid, v in items.items()])
        self.db.commit()
