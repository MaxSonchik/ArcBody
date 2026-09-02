"""Measuring and encoding a body, and comparing it with an enrolled one."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from arcbody.api.deps import get_pipeline
from arcbody.api.inputs import decode_images
from arcbody.api.mapping import analysis_response, generation_check_response, match_model
from arcbody.config import Settings, get_settings
from arcbody.imaging import decode
from arcbody.pipeline import ArcBodyPipeline, ImageInput
from arcbody.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    GenerationCheckRequest,
    GenerationCheckResponse,
    IdentifyRequest,
    IdentifyResponse,
)
from arcbody.types import ViewLabel

router = APIRouter(prefix="/v1", tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    pipeline: ArcBodyPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    """Measure a body and produce everything a generator needs to reproduce it.

    Supply one photo for proportions and single-view girths; add a side view to
    replace the depth prior with a measurement. Supply ``stature_cm`` to get
    centimetres — without it the response carries proportions only, because a
    photograph determines shape and never size.
    """
    result = pipeline.analyse(
        decode_images(request.images, settings),
        stature_cm=request.stature_cm,
        weight_kg=request.weight_kg,
        want_control_maps=request.include_control_maps,
        strict_quality=request.strict_quality,
    )
    return analysis_response(result, settings, include_embedding=request.include_embedding)


@router.post("/analyze/upload", response_model=AnalyzeResponse)
async def analyze_upload(
    files: list[UploadFile] = File(..., description="One or more photos of the same subject."),
    stature_cm: float | None = Form(None),
    weight_kg: float | None = Form(None),
    view: str = Form("auto", description="Applied to every file; use JSON for per-image views."),
    include_control_maps: bool = Form(True),
    pipeline: ArcBodyPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    """The same analysis, as a plain multipart upload.

    Convenience for clients that have files rather than base64 — a browser form,
    or curl. Per-image view labels need the JSON endpoint.
    """
    label = None if view == "auto" else ViewLabel(view)
    items = [
        ImageInput(
            image=decode(await file.read(), max_pixels=settings.max_image_pixels),
            view=label,
            label=file.filename,
        )
        for file in files[: settings.max_images_per_request]
    ]
    result = pipeline.analyse(
        items,
        stature_cm=stature_cm,
        weight_kg=weight_kg,
        want_control_maps=include_control_maps,
    )
    return analysis_response(result, settings)


@router.post("/identify", response_model=IdentifyResponse)
def identify(
    request: IdentifyRequest,
    pipeline: ArcBodyPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> IdentifyResponse:
    """Find the enrolled bodies most similar to the one in these photos."""
    result = pipeline.analyse(
        decode_images(request.images, settings),
        stature_cm=request.stature_cm,
        weight_kg=request.weight_kg,
        want_control_maps=request.include_control_maps,
        strict_quality=request.strict_quality,
    )
    matches = pipeline.gallery.identify(
        result.profile.embedding,
        top_k=request.top_k,
        minimum=request.minimum_similarity,
    )
    return IdentifyResponse(
        matches=[match_model(match) for match in matches],
        threshold=settings.embedding.match_threshold,
        encoder_trained=pipeline.encoder.trained,
        analysis=analysis_response(
            result, settings, include_embedding=request.include_embedding
        ),
    )


@router.post(
    "/persons/{person_id}/check-generation", response_model=GenerationCheckResponse
)
def check_generation(
    person_id: str,
    request: GenerationCheckRequest,
    pipeline: ArcBodyPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> GenerationCheckResponse:
    """Score a generated image against the body enrolled under ``person_id``.

    This is the loop-closer: send the photo you enrolled from, generate, then
    send the generation back here. The similarity says whether it is still the
    same body and the ratio deltas say which proportions the generator moved, so
    a failure points at the prompt term to fix instead of at a re-roll.
    """
    result = pipeline.analyse(
        decode_images(request.images, settings),
        stature_cm=request.stature_cm,
        weight_kg=request.weight_kg,
        want_control_maps=request.include_control_maps,
        # A generated image is not a controlled capture and will often fail the
        # framing gates. Rejecting it would defeat the purpose of the check, so
        # the gates report rather than block here.
        strict_quality=False,
    )
    check = pipeline.check_generation(person_id, result, threshold=request.threshold)
    return generation_check_response(
        check,
        analysis_response(result, settings, include_embedding=request.include_embedding),
    )
