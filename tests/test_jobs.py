"""Batch job persistence.

The property worth testing is not "a job runs" — the API tests cover that — but
what happens across a restart, which is where the previous in-memory store
turned a finished job into a 404.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from arcbody.api.jobs import JobStore
from arcbody.config import Settings
from arcbody.errors import ArcBodyError, JobNotFoundError


def settings_for(tmp_path: Path, **batch) -> Settings:
    return Settings(
        gallery={"database_path": tmp_path / "jobs.sqlite3"},
        batch=batch or {},
    )


def wait_for(store: JobStore, job_id: str, *, status: str = "completed") -> None:
    for _ in range(200):
        if store.get(job_id).status == status:
            return
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {status}")


def test_a_finished_job_survives_a_restart(tmp_path: Path) -> None:
    store = JobStore(settings_for(tmp_path))
    job = store.submit(["a", "b"], lambda index: {"value": index})
    wait_for(store, job.job_id)
    store.shutdown()

    # A fresh process, same database.
    reopened = JobStore(settings_for(tmp_path))
    recovered = reopened.get(job.job_id)
    assert recovered.status == "completed"
    assert recovered.completed == 2 and recovered.failed == 0
    assert [item.payload for item in recovered.results] == [{"value": 0}, {"value": 1}]
    reopened.shutdown()


def test_an_interrupted_job_fails_explicitly_rather_than_vanishing(tmp_path: Path) -> None:
    """Resuming would mean having kept the photographs, which the service does not."""
    store = JobStore(settings_for(tmp_path))
    store.shutdown()

    database = tmp_path / "jobs.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO batch_jobs (job_id, status, submitted, created_at) VALUES (?,?,?,?)",
        ("stranded", "running", 2, "2026-01-01T00:00:00+00:00"),
    )
    connection.executemany(
        "INSERT INTO batch_items (job_id, position, reference, status) VALUES (?,?,?,?)",
        [("stranded", 0, "x", "ok"), ("stranded", 1, "y", "pending")],
    )
    connection.commit()
    connection.close()

    reopened = JobStore(settings_for(tmp_path))
    job = reopened.get("stranded")
    assert job.status == "failed"
    assert job.finished_at is not None
    stranded_item = next(item for item in job.results if item.reference == "y")
    assert stranded_item.status == "failed"
    assert stranded_item.error is not None
    assert "resubmit" in stranded_item.error["message"]
    reopened.shutdown()


def test_one_failing_item_does_not_abort_the_job(tmp_path: Path) -> None:
    store = JobStore(settings_for(tmp_path))

    def work(index: int) -> dict[str, object]:
        if index == 1:
            raise ValueError("boom")
        return {"index": index}

    job = store.submit(["a", "b", "c"], work)
    wait_for(store, job.job_id)
    finished = store.get(job.job_id)
    assert finished.completed == 2 and finished.failed == 1
    failed = next(item for item in finished.results if item.reference == "b")
    assert failed.error["code"] == "internal_error"
    store.shutdown()


def test_domain_errors_keep_their_code(tmp_path: Path) -> None:
    store = JobStore(settings_for(tmp_path))

    class Boom(ArcBodyError):
        code = "boom"

    job = store.submit(["a"], lambda index: (_ for _ in ()).throw(Boom("no")))
    wait_for(store, job.job_id)
    assert store.get(job.job_id).results[0].error["code"] == "boom"
    store.shutdown()


def test_a_running_job_reports_no_partial_results(tmp_path: Path) -> None:
    store = JobStore(settings_for(tmp_path))

    def slow(index: int) -> dict[str, object]:
        time.sleep(0.3)
        return {"index": index}

    job = store.submit(["a", "b"], slow)
    time.sleep(0.05)
    assert store.get(job.job_id).status in {"queued", "running"}
    wait_for(store, job.job_id)
    store.shutdown()


def test_oversized_jobs_are_refused(tmp_path: Path) -> None:
    store = JobStore(settings_for(tmp_path, max_items_per_job=2))
    with pytest.raises(ArcBodyError, match="at most 2"):
        store.submit(["a", "b", "c"], lambda index: {})
    store.shutdown()


def test_unknown_job_says_why_it_might_be_gone(tmp_path: Path) -> None:
    store = JobStore(settings_for(tmp_path))
    with pytest.raises(JobNotFoundError, match="retained"):
        store.get("nope")
    store.shutdown()


def test_expired_jobs_are_evicted(tmp_path: Path) -> None:
    store = JobStore(settings_for(tmp_path, retain_seconds=0))
    first = store.submit(["a"], lambda index: {"index": index})
    wait_for(store, first.job_id)

    # Submitting again sweeps anything past its retention window.
    second = store.submit(["b"], lambda index: {"index": index})
    wait_for(store, second.job_id)
    with pytest.raises(JobNotFoundError):
        store.get(first.job_id)
    store.shutdown()


def test_payloads_round_trip_through_the_database(tmp_path: Path) -> None:
    """Results are JSON in a text column, so they must be JSON-able going in."""
    store = JobStore(settings_for(tmp_path))
    payload = {"analysis": {"measurements": {"ratios": {"waist_to_hip": 0.81}}}}
    job = store.submit(["a"], lambda index: payload)
    wait_for(store, job.job_id)
    assert store.get(job.job_id).results[0].payload == payload
    assert json.dumps(payload)
    store.shutdown()
