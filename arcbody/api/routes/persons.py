"""Enrolling, listing and deleting body profiles."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response, status

from arcbody.api.deps import get_pipeline
from arcbody.api.inputs import decode_images
from arcbody.api.mapping import analysis_response, person_model
from arcbody.config import Settings, get_settings
from arcbody.errors import PersonNotFoundError
from arcbody.pipeline import ArcBodyPipeline
from arcbody.schemas import (
    EnrolRequest,
    EnrolResponse,
    PersonListResponse,
    PersonModel,
)

router = APIRouter(prefix="/v1/persons", tags=["persons"])


@router.post(
    "/{person_id}/enroll",
    response_model=EnrolResponse,
    status_code=status.HTTP_201_CREATED,
)
def enroll(
    person_id: str,
    request: EnrolRequest,
    pipeline: ArcBodyPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> EnrolResponse:
    """Register a body under a caller-owned id.

    Enrolment is additive: photograph the same person again and they gain a
    profile rather than replacing one, and their signature becomes the fusion of
    all of them. ``person_id`` and ``external_face_id`` are opaque to ArcBody —
    identity is your face service's business, and this service only ever links
    to it.
    """
    result = pipeline.analyse(
        decode_images(request.images, settings),
        stature_cm=request.stature_cm,
        weight_kg=request.weight_kg,
        want_control_maps=request.include_control_maps,
        strict_quality=request.strict_quality,
    )
    profile_id = pipeline.enrol(
        person_id,
        result,
        external_face_id=request.external_face_id,
        metadata=dict(request.metadata),
    )
    record = pipeline.gallery.get(person_id)
    return EnrolResponse(
        person_id=person_id,
        profile_id=profile_id,
        profile_count=record.profile_count,
        external_face_id=record.external_face_id,
        analysis=analysis_response(
            result, settings, include_embedding=request.include_embedding
        ),
    )


@router.get("", response_model=PersonListResponse)
def list_persons(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    pipeline: ArcBodyPipeline = Depends(get_pipeline),
) -> PersonListResponse:
    persons, profiles = pipeline.gallery.count()
    return PersonListResponse(
        persons=[person_model(record) for record in pipeline.gallery.list_persons(limit, offset)],
        total_persons=persons,
        total_profiles=profiles,
    )


@router.get("/{person_id}", response_model=PersonModel)
def get_person(
    person_id: str, pipeline: ArcBodyPipeline = Depends(get_pipeline)
) -> PersonModel:
    return person_model(pipeline.gallery.get(person_id))


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_person(
    person_id: str, pipeline: ArcBodyPipeline = Depends(get_pipeline)
) -> Response:
    """Erase a person and every profile of them.

    A hard delete, not a flag. Body embeddings are biometric data and a deletion
    request has to actually remove them.
    """
    if pipeline.gallery.forget(person_id) == 0:
        raise PersonNotFoundError(f"no enrolled body profile for {person_id!r}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
