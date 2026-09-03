"""Persistence and search over enrolled body profiles.

Storage is SQLite with the embeddings as raw ``float32`` blobs, and search is a
vectorised brute-force cosine over an in-memory matrix. At 100k people that is
100 MB of floats and a fraction of a millisecond per query, measured by
``scripts/benchmark_gallery.py`` — an approximate index would add a dependency,
a build step and a recall cliff in exchange for nothing.

The matrix is maintained *incrementally*. Rebuilding it on every write looked
harmless and was not: the benchmark showed a query following an enrolment
costing 190 ms against 0.22 ms in steady state, because each write threw the
index away and the next read paid to rebuild it. Enrolment-heavy traffic —
exactly what a batch job produces — would have run three orders of magnitude
slower than the steady-state number anyone would have quoted.

**What is stored, and what is not.** Body embeddings and measurements are
biometric data. This store keeps the derived vector, the ratios and the
measurements; it never keeps the photograph. That is a deliberate default, not
an oversight: a profile is enough to recognise someone, but not enough to
reconstruct the image they were recognised from, and :meth:`Gallery.forget` is a
real, complete delete rather than a soft flag.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from arcbody.config import GallerySettings
from arcbody.embed.encoder import fuse
from arcbody.errors import PersonNotFoundError

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    person_id        TEXT PRIMARY KEY,
    external_face_id TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    metadata         TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS profiles (
    profile_id   TEXT PRIMARY KEY,
    person_id    TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    created_at   TEXT NOT NULL,
    embedding    BLOB NOT NULL,
    dim          INTEGER NOT NULL,
    ratios       TEXT NOT NULL DEFAULT '{}',
    measurements TEXT NOT NULL DEFAULT '{}',
    quality      REAL NOT NULL DEFAULT 0.0,
    view         TEXT NOT NULL DEFAULT 'unknown',
    note         TEXT
);

CREATE INDEX IF NOT EXISTS profiles_person ON profiles(person_id);
CREATE INDEX IF NOT EXISTS persons_face ON persons(external_face_id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class PersonRecord:
    """A person and the fused signature of everything enrolled for them."""

    person_id: str
    embedding: np.ndarray
    profile_count: int
    external_face_id: str | None = None
    ratios: dict[str, float] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Match:
    """One identification candidate."""

    person_id: str
    similarity: float
    profile_count: int
    external_face_id: str | None = None


class Gallery:
    """Thread-safe store of enrolled body profiles."""

    def __init__(self, settings: GallerySettings) -> None:
        self.settings = settings
        path = Path(settings.database_path)
        if path.parent and str(path.parent) not in {"", "."}:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(SCHEMA)
        self._connection.commit()

        # Search index. Rows are maintained in place; ``_used`` marks how many
        # of the matrix's rows are live, so appends and deletes cost O(1)
        # amortised instead of a full rebuild.
        self._matrix: np.ndarray | None = None
        self._used = 0
        self._index_ids: list[str] = []
        self._index_counts: list[int] = []
        self._index_faces: list[str | None] = []
        self._row_of: dict[str, int] = {}
        self._capacity_warned = False

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # -- writes -----------------------------------------------------------

    def enrol(
        self,
        person_id: str,
        embedding: np.ndarray,
        *,
        ratios: dict[str, float] | None = None,
        measurements: dict[str, Any] | None = None,
        quality: float = 0.0,
        view: str = "unknown",
        external_face_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> str:
        """Add one profile to a person, creating the person if new.

        Enrolment is additive by design. A person photographed again in a
        different outfit or at a different weight gains a profile rather than
        overwriting one, and identification compares against the fusion of all
        of them — which is how the signature stays current without losing the
        history that made it trustworthy.
        """
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        if vector.size == 0:
            raise ValueError("cannot enrol an empty embedding")
        profile_id = uuid.uuid4().hex
        timestamp = _now()

        with self._lock:
            self._connection.execute(
                """
                INSERT INTO persons (person_id, external_face_id, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    external_face_id = COALESCE(
                        excluded.external_face_id, persons.external_face_id),
                    metadata = CASE
                        WHEN excluded.metadata = '{}' THEN persons.metadata
                        ELSE excluded.metadata END
                """,
                (
                    person_id,
                    external_face_id,
                    timestamp,
                    timestamp,
                    json.dumps(metadata or {}),
                ),
            )
            self._connection.execute(
                """
                INSERT INTO profiles
                    (profile_id, person_id, created_at, embedding, dim,
                     ratios, measurements, quality, view, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_id,
                    person_id,
                    timestamp,
                    vector.tobytes(),
                    int(vector.size),
                    json.dumps(ratios or {}),
                    json.dumps(measurements or {}, default=str),
                    float(quality),
                    view,
                    note,
                ),
            )
            self._connection.commit()
            self._touch_person(person_id)
        return profile_id

    def forget(self, person_id: str) -> int:
        """Delete a person and every profile of them. Returns rows removed."""
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM persons WHERE person_id = ?", (person_id,)
            )
            self._connection.commit()
            self._drop_row(person_id)
            return int(cursor.rowcount)

    # -- reads ------------------------------------------------------------

    def get(self, person_id: str) -> PersonRecord:
        with self._lock:
            person = self._connection.execute(
                "SELECT * FROM persons WHERE person_id = ?", (person_id,)
            ).fetchone()
            if person is None:
                raise PersonNotFoundError(f"no enrolled body profile for {person_id!r}")
            rows = self._connection.execute(
                "SELECT * FROM profiles WHERE person_id = ? ORDER BY created_at", (person_id,)
            ).fetchall()
        return self._record(person, rows)

    def list_persons(self, limit: int = 100, offset: int = 0) -> list[PersonRecord]:
        with self._lock:
            people = self._connection.execute(
                "SELECT * FROM persons ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (int(limit), int(offset)),
            ).fetchall()
            records = []
            for person in people:
                rows = self._connection.execute(
                    "SELECT * FROM profiles WHERE person_id = ?", (person["person_id"],)
                ).fetchall()
                records.append(self._record(person, rows))
        return records

    def count(self) -> tuple[int, int]:
        """``(people, profiles)``."""
        with self._lock:
            people = self._connection.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            profiles = self._connection.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
        return int(people), int(profiles)

    def _record(self, person: sqlite3.Row, rows: list[sqlite3.Row]) -> PersonRecord:
        embeddings = [
            np.frombuffer(row["embedding"], dtype=np.float32) for row in rows
        ]
        fused = (
            fuse(embeddings) if embeddings else np.zeros(0, dtype=np.float32)
        )
        # The most recent profile's measurements describe the person now; older
        # ones are history, not an average worth quoting.
        latest = rows[-1] if rows else None
        return PersonRecord(
            person_id=person["person_id"],
            embedding=fused,
            profile_count=len(rows),
            external_face_id=person["external_face_id"],
            ratios=json.loads(latest["ratios"]) if latest else {},
            measurements=json.loads(latest["measurements"]) if latest else {},
            metadata=json.loads(person["metadata"]),
            created_at=person["created_at"],
            updated_at=person["updated_at"],
        )

    # -- search -----------------------------------------------------------

    def _invalidate(self) -> None:
        """Drop the index entirely. Only for changes that cannot be localised."""
        self._matrix = None
        self._used = 0
        self._index_ids = []
        self._index_counts = []
        self._index_faces = []
        self._row_of = {}

    def _person_vector(self, person_id: str) -> tuple[np.ndarray, int, str | None] | None:
        """The fused signature of one person, straight from the database."""
        rows = self._connection.execute(
            """
            SELECT p.external_face_id, f.embedding
            FROM persons p JOIN profiles f ON f.person_id = p.person_id
            WHERE p.person_id = ?
            ORDER BY f.created_at
            """,
            (person_id,),
        ).fetchall()
        if not rows:
            return None
        vectors = [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
        width = len(vectors[0])
        if any(len(vector) != width for vector in vectors):
            return None
        return fuse(vectors), len(vectors), rows[0]["external_face_id"]

    def _touch_person(self, person_id: str) -> None:
        """Refresh one person's row after a write.

        Called with the lock held. When the index has not been built yet there
        is nothing to refresh — the lazy full build will pick the person up.
        """
        if self._matrix is None:
            return
        entry = self._person_vector(person_id)
        if entry is None:
            self._drop_row(person_id)
            return
        vector, count, face = entry

        row = self._row_of.get(person_id)
        if row is not None:
            if vector.size != self._matrix.shape[1]:
                # The embedding dimension changed under a live gallery, which
                # means the encoder checkpoint changed. Nothing local can fix
                # that; rebuild and let the width filter sort it out.
                self._invalidate()
                return
            self._matrix[row] = vector
            self._index_counts[row] = count
            self._index_faces[row] = face
            return

        if self._used and vector.size != self._matrix.shape[1]:
            self._invalidate()
            return
        self._append_row(person_id, vector, count, face)

    def _append_row(
        self, person_id: str, vector: np.ndarray, count: int, face: str | None
    ) -> None:
        """Add a new person to the index, growing the backing array if needed."""
        assert self._matrix is not None
        if self._used >= self._matrix.shape[0] or self._matrix.shape[1] != vector.size:
            capacity = max(64, int(self._used * 1.5) + 1)
            grown = np.zeros((capacity, vector.size), dtype=np.float32)
            if self._used:
                grown[: self._used] = self._matrix[: self._used]
            self._matrix = grown

        self._matrix[self._used] = vector
        self._index_ids.append(person_id)
        self._index_counts.append(count)
        self._index_faces.append(face)
        self._row_of[person_id] = self._used
        self._used += 1

        limit = int(self.settings.max_identify_candidates)
        if self._used > limit and not self._capacity_warned:
            # Everyone stays searchable: quietly dropping people from an
            # identity search is a correctness bug wearing a memory-limit
            # costume. The warning is the signal to move to a real index.
            self._capacity_warned = True
            logger.warning(
                "gallery holds %d people, above max_identify_candidates=%d; brute-force "
                "search still returns every person but memory and latency now grow "
                "unbounded. Consider an approximate index.",
                self._used,
                limit,
            )

    def _drop_row(self, person_id: str) -> None:
        """Remove a person from the index by swapping the last row into place."""
        if self._matrix is None:
            return
        row = self._row_of.pop(person_id, None)
        if row is None:
            return
        last = self._used - 1
        if row != last:
            self._matrix[row] = self._matrix[last]
            moved = self._index_ids[last]
            self._index_ids[row] = moved
            self._index_counts[row] = self._index_counts[last]
            self._index_faces[row] = self._index_faces[last]
            self._row_of[moved] = row
        self._index_ids.pop()
        self._index_counts.pop()
        self._index_faces.pop()
        self._used = last

    def _build_index(self) -> None:
        """Materialise one fused unit vector per person, from scratch."""
        rows = self._connection.execute(
            """
            SELECT p.person_id, p.external_face_id, f.embedding
            FROM persons p JOIN profiles f ON f.person_id = p.person_id
            ORDER BY p.person_id, f.created_at
            """
        ).fetchall()

        grouped: dict[str, list[np.ndarray]] = {}
        faces: dict[str, str | None] = {}
        for row in rows:
            grouped.setdefault(row["person_id"], []).append(
                np.frombuffer(row["embedding"], dtype=np.float32)
            )
            faces[row["person_id"]] = row["external_face_id"]

        if not grouped:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            self._used = 0
            self._index_ids = []
            self._index_counts = []
            self._index_faces = []
            self._row_of = {}
            return

        ids = sorted(grouped)
        # Ragged dimensions mean a checkpoint changed under a live gallery.
        # Re-enrolment is the fix; silently truncating would corrupt the metric.
        width = len(grouped[ids[0]][0])
        usable = [pid for pid in ids if all(len(v) == width for v in grouped[pid])]
        self._matrix = np.stack([fuse(grouped[pid]) for pid in usable]).astype(np.float32)
        self._used = len(usable)
        self._index_ids = list(usable)
        self._index_counts = [len(grouped[pid]) for pid in usable]
        self._index_faces = [faces[pid] for pid in usable]
        self._row_of = {pid: row for row, pid in enumerate(usable)}

    def identify(
        self, embedding: np.ndarray, *, top_k: int | None = None, minimum: float = -1.0
    ) -> list[Match]:
        """Nearest enrolled people by cosine similarity."""
        query = np.asarray(embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query))
        if norm < 1e-8:
            return []
        query = query / norm

        with self._lock:
            if self._matrix is None:
                self._build_index()
            matrix = self._matrix
            used = self._used
            ids = list(self._index_ids)
            counts = list(self._index_counts)
            faces = list(self._index_faces)

        if matrix is None or used == 0 or matrix.shape[1] != query.size:
            return []

        scores = matrix[:used] @ query
        k = min(len(ids), top_k or self.settings.default_top_k)
        if k <= 0:
            return []
        # argpartition finds the top k without sorting the other 99,995.
        candidates = np.argpartition(-scores, k - 1)[:k]
        candidates = candidates[np.argsort(-scores[candidates])]
        return [
            Match(
                person_id=ids[index],
                similarity=float(scores[index]),
                profile_count=counts[index],
                external_face_id=faces[index],
            )
            for index in candidates
            if scores[index] >= minimum
        ]

    def verify(self, person_id: str, embedding: np.ndarray) -> float:
        """Cosine similarity between an embedding and one enrolled person."""
        record = self.get(person_id)
        if record.embedding.size == 0:
            raise PersonNotFoundError(f"{person_id!r} has no enrolled profiles")
        query = np.asarray(embedding, dtype=np.float32).ravel()
        if query.size != record.embedding.size:
            raise ValueError(
                f"embedding dimension {query.size} does not match the enrolled "
                f"{record.embedding.size}; the encoder checkpoint has changed"
            )
        denominator = float(np.linalg.norm(query) * np.linalg.norm(record.embedding))
        if denominator < 1e-8:
            return 0.0
        return float(np.clip(np.dot(query, record.embedding) / denominator, -1.0, 1.0))
