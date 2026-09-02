"""In-process asynchronous batch jobs.

Deliberately not a queue broker. The service's stated ceiling is a few hundred
photos per job on a single node, and a thread pool plus a dict covers that
without operational surface. The seam is the ``JobStore`` interface: swapping in
Redis or a real worker pool later means replacing this file, not the routes.

Jobs live in memory and do not survive a restart. That is a real limitation and
it is documented rather than hidden — callers get a job id and are told to poll,
and a 404 after a restart means resubmit.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from arcbody.config import BatchSettings
from arcbody.errors import ArcBodyError, JobNotFoundError

logger = logging.getLogger(__name__)


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
    """Runs batch work on a bounded pool and keeps results for a while."""

    def __init__(self, settings: BatchSettings) -> None:
        self.settings = settings
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.max_concurrent_jobs),
            thread_name_prefix="arcbody-batch",
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def submit(
        self,
        references: list[str],
        work: Callable[[int], dict[str, Any]],
    ) -> Job:
        """Queue a job whose ``work(index)`` produces one item's payload."""
        if len(references) > self.settings.max_items_per_job:
            raise ArcBodyError(
                f"a batch job may hold at most {self.settings.max_items_per_job} items",
            )
        job = Job(job_id=uuid.uuid4().hex, submitted=len(references))
        with self._lock:
            self._evict_expired()
            self._jobs[job.job_id] = job
        self._executor.submit(self._run, job, references, work)
        return job

    def _run(
        self,
        job: Job,
        references: list[str],
        work: Callable[[int], dict[str, Any]],
    ) -> None:
        job.status = "running"
        # Items are processed sequentially inside the job. The models already
        # saturate the available cores on a single image, so fanning out within
        # a job would only trade throughput for latency variance.
        for index, reference in enumerate(references):
            try:
                job.results.append(
                    JobItemResult(reference=reference, status="ok", payload=work(index))
                )
            except ArcBodyError as error:
                job.results.append(
                    JobItemResult(reference=reference, status="failed", error=error.to_dict())
                )
            except Exception as error:  # noqa: BLE001 - one bad item must not kill the job
                logger.exception("batch item %s failed", reference)
                job.results.append(
                    JobItemResult(
                        reference=reference,
                        status="failed",
                        error={"code": "internal_error", "message": str(error), "details": {}},
                    )
                )
        job.status = "completed"
        job.finished_at = _now()

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(
                f"no job {job_id!r}; jobs are held in memory and do not survive a restart"
            )
        return job

    def _evict_expired(self) -> None:
        """Drop finished jobs past their retention window."""
        cutoff = datetime.now(UTC).timestamp() - self.settings.retain_seconds
        for job_id, job in list(self._jobs.items()):
            if job.finished_at is None:
                continue
            if datetime.fromisoformat(job.finished_at).timestamp() < cutoff:
                del self._jobs[job_id]


_store: JobStore | None = None
_store_lock = threading.Lock()


def get_job_store(settings: BatchSettings) -> JobStore:
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
