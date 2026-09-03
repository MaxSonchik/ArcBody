"""Asynchronous batch processing."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from arcbody.api.deps import get_pipeline
from arcbody.api.inputs import decode_images
from arcbody.api.jobs import Job, get_job_store
from arcbody.api.mapping import analysis_response
from arcbody.config import Settings, get_settings
from arcbody.pipeline import ArcBodyPipeline
from arcbody.schemas import AnalyzeResponse, BatchItemResult, BatchJobModel, BatchRequest

router = APIRouter(prefix="/v1/batch", tags=["batch"])


def _to_model(job: Job) -> BatchJobModel:
    # Results appear once the job has stopped moving; a running job's partial
    # results would invite a client to act on half an answer.
    results = None
    if job.status in {"completed", "failed"}:
        results = [
            BatchItemResult(
                reference=item.reference,
                status=item.status,  # type: ignore[arg-type]
                person_id=(item.payload or {}).get("person_id"),
                profile_id=(item.payload or {}).get("profile_id"),
                analysis=(item.payload or {}).get("analysis"),
                error=item.error,
            )
            for item in job.results
        ]
    return BatchJobModel(
        job_id=job.job_id,
        status=job.status,  # type: ignore[arg-type]
        submitted=job.submitted,
        completed=job.completed,
        failed=job.failed,
        created_at=job.created_at,
        finished_at=job.finished_at,
        results=results,
    )


@router.post("", response_model=BatchJobModel, status_code=status.HTTP_202_ACCEPTED)
def submit_batch(
    request: BatchRequest,
    pipeline: ArcBodyPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> BatchJobModel:
    """Queue many subjects and return a job id to poll.

    Images are decoded up front so a malformed payload fails the submission
    rather than one item deep into a job the caller has already been told was
    accepted. Everything after that is per-item: one subject failing to segment
    does not abort the rest.
    """
    decoded = [decode_images(item.images, settings) for item in request.items]

    def work(index: int) -> dict[str, object]:
        item = request.items[index]
        result = pipeline.analyse(
            decoded[index],
            stature_cm=item.stature_cm,
            weight_kg=item.weight_kg,
            want_control_maps=request.include_control_maps,
            strict_quality=request.strict_quality,
        )
        # A plain JSON-able dict, not a pydantic model: the payload is written
        # to a text column, and a model here would put the wire schema inside
        # the job runner.
        payload: dict[str, object] = {
            "analysis": analysis_response(result, settings).model_dump(mode="json"),
        }
        if item.person_id:
            payload["profile_id"] = pipeline.enrol(item.person_id, result)
            payload["person_id"] = item.person_id
        return payload

    job = get_job_store(settings).submit(
        [item.reference for item in request.items], work
    )
    return _to_model(job)


@router.get("/{job_id}", response_model=BatchJobModel)
def get_batch(job_id: str, settings: Settings = Depends(get_settings)) -> BatchJobModel:
    """Poll a job. Results appear once the status reaches ``completed``."""
    return _to_model(get_job_store(settings).get(job_id))


# Re-exported so FastAPI resolves the forward reference in BatchItemResult.
__all__ = ["router", "AnalyzeResponse"]
