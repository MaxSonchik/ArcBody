"""Asynchronous batch jobs, with results that survive a restart.

Deliberately not a queue broker. The service's stated ceiling is a few hundred
photos per job on a single node, and a thread pool plus SQLite covers that
without the operational surface of Redis or Celery.

**What persistence does and does not buy.** Job records and finished results
live in the database, so the common case — a client submits, goes away, and
polls an hour later across a deployment — now works. A job interrupted
*mid-flight* cannot resume, because resuming would mean having kept the
submitted photographs, and not keeping them is a deliberate privacy property of
this service. Such a job is marked failed at startup with that reason spelled
out, which is a better answer than a 404 that looks like the job never existed.

Payloads crossing this module are plain JSON-able dicts, not pydantic models:
they are written to a text column, and a model here would put the wire schema
inside the job runner.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arcbody.config import Settings
from arcbody.errors import ArcBodyError, JobNotFoundError

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS batch_jobs (
    job_id      TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    submitted   INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS batch_items (
    job_id    TEXT NOT NULL REFERENCES batch_jobs(job_id) ON DELETE CASCADE,
    position  INTEGER NOT NULL,
    reference TEXT NOT NULL,
    status    TEXT NOT NULL,
    payload   TEXT,
    error     TEXT,
    PRIMARY KEY (job_id, position)
);

CREATE INDEX IF NOT EXISTS batch_jobs_finished ON batch_jobs(finished_at);
"""

INTERRUPTED_ERROR = {
    "code": "interrupted",
    "message": (
        "the service restarted while this job was running. ArcBody does not store "
        "submitted photographs, so the job cannot resume — resubmit it."
    ),
    "details": {},
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class JobItemResult:
    reference: str
    status: str
    payload: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


@dataclass
class Job:
    job_id: str
    submitted: int
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None
    results: list[JobItemResult] = field(default_factory=list)

    @property
    def completed(self) -> int:
        return sum(1 for item in self.results if item.status == "ok")

    @property
    def failed(self) -> int:
        return sum(1 for item in self.results if item.status == "failed")


class JobStore:
    """Runs batch work on a bounded pool and keeps results in SQLite."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings.batch
        path = Path(settings.gallery.database_path)
        if path.parent and str(path.parent) not in {"", "."}:
            path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(SCHEMA)
        self._connection.commit()

        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(
            max_workers=max(1, self.settings.max_concurrent_jobs),
            thread_name_prefix="arcbody-batch",
        )
        self._recover_interrupted()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            self._connection.close()

    # -- lifecycle --------------------------------------------------------

    def _recover_interrupted(self) -> int:
        """Fail jobs left mid-flight by a restart, and say why."""
        with self._lock:
            cursor = self._connection.execute(
                "SELECT job_id FROM batch_jobs WHERE status IN ('queued', 'running')"
            )
            stranded = [row["job_id"] for row in cursor.fetchall()]
            for job_id in stranded:
                self._connection.execute(
                    """
                    UPDATE batch_items SET status = 'failed', error = ?
                    WHERE job_id = ? AND status = 'pending'
                    """,
                    (json.dumps(INTERRUPTED_ERROR), job_id),
                )
                self._connection.execute(
                    "UPDATE batch_jobs SET status = 'failed', finished_at = ? WHERE job_id = ?",
                    (_now(), job_id),
                )
            self._connection.commit()
        if stranded:
            logger.warning(
                "marked %d batch job(s) as failed: they were interrupted by a restart "
                "and cannot resume because submitted photographs are not stored",
                len(stranded),
            )
        return len(stranded)

    # -- submission -------------------------------------------------------

    def submit(
        self, references: list[str], work: Callable[[int], dict[str, Any]]
    ) -> Job:
        """Queue a job whose ``work(index)`` produces one item's JSON payload."""
        if len(references) > self.settings.max_items_per_job:
            raise ArcBodyError(
                f"a batch job may hold at most {self.settings.max_items_per_job} items",
            )
        job_id = uuid.uuid4().hex
        created = _now()
        with self._lock:
            self._evict_expired()
            self._connection.execute(
                "INSERT INTO batch_jobs (job_id, status, submitted, created_at) VALUES (?,?,?,?)",
                (job_id, "queued", len(references), created),
            )
            self._connection.executemany(
                "INSERT INTO batch_items (job_id, position, reference, status) VALUES (?,?,?,?)",
                [
                    (job_id, position, reference, "pending")
                    for position, reference in enumerate(references)
                ],
            )
            self._connection.commit()

        self._executor.submit(self._run, job_id, references, work)
        return Job(job_id=job_id, submitted=len(references), created_at=created)

    def _run(
        self, job_id: str, references: list[str], work: Callable[[int], dict[str, Any]]
    ) -> None:
        self._set_status(job_id, "running")
        # Items run sequentially inside a job: the models already saturate the
        # available cores on a single image, so fanning out within one job would
        # trade throughput for latency variance and nothing else.
        for position, reference in enumerate(references):
            try:
                self._record(job_id, position, "ok", payload=work(position))
            except ArcBodyError as error:
                self._record(job_id, position, "failed", error=error.to_dict())
            except Exception as error:  # noqa: BLE001 - one bad item must not kill the job
                logger.exception("batch item %s failed", reference)
                self._record(
                    job_id,
                    position,
                    "failed",
                    error={"code": "internal_error", "message": str(error), "details": {}},
                )
        self._set_status(job_id, "completed", finished=True)

    def _set_status(self, job_id: str, status: str, *, finished: bool = False) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE batch_jobs SET status = ?, finished_at = ? WHERE job_id = ?",
                (status, _now() if finished else None, job_id),
            )
            self._connection.commit()

    def _record(
        self,
        job_id: str,
        position: int,
        status: str,
        *,
        payload: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE batch_items SET status = ?, payload = ?, error = ? "
                "WHERE job_id = ? AND position = ?",
                (
                    status,
                    json.dumps(payload) if payload is not None else None,
                    json.dumps(error) if error is not None else None,
                    job_id,
                    position,
                ),
            )
            self._connection.commit()

    # -- reads ------------------------------------------------------------

    def get(self, job_id: str) -> Job:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM batch_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(
                    f"no job {job_id!r}; finished jobs are retained for "
                    f"{self.settings.retain_seconds} seconds"
                )
            items = self._connection.execute(
                "SELECT * FROM batch_items WHERE job_id = ? ORDER BY position", (job_id,)
            ).fetchall()

        return Job(
            job_id=row["job_id"],
            submitted=int(row["submitted"]),
            status=row["status"],
            created_at=row["created_at"],
            finished_at=row["finished_at"],
            results=[
                JobItemResult(
                    reference=item["reference"],
                    status=item["status"],
                    payload=json.loads(item["payload"]) if item["payload"] else None,
                    error=json.loads(item["error"]) if item["error"] else None,
                )
                for item in items
                # A pending item is work not yet done, not a result.
                if item["status"] != "pending"
            ],
        )

    def _evict_expired(self) -> int:
        """Drop finished jobs past their retention window. Lock must be held."""
        cutoff = datetime.now(UTC).timestamp() - self.settings.retain_seconds
        rows = self._connection.execute(
            "SELECT job_id, finished_at FROM batch_jobs WHERE finished_at IS NOT NULL"
        ).fetchall()
        stale = [
            row["job_id"]
            for row in rows
            if datetime.fromisoformat(row["finished_at"]).timestamp() < cutoff
        ]
        for job_id in stale:
            self._connection.execute("DELETE FROM batch_jobs WHERE job_id = ?", (job_id,))
        if stale:
            self._connection.commit()
        return len(stale)


_store: JobStore | None = None
_store_lock = threading.Lock()


def get_job_store(settings: Settings) -> JobStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = JobStore(settings)
    return _store


def reset_job_store() -> None:
    global _store
    with _store_lock:
        if _store is not None:
            _store.shutdown()
        _store = None
