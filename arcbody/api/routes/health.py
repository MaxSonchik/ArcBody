"""Liveness and capability reporting."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from arcbody.api.deps import get_pipeline
from arcbody.api.security import Caller, authenticate_optional
from arcbody.config import Settings, get_settings
from arcbody.pipeline import ArcBodyPipeline
from arcbody.schemas import HealthResponse
from arcbody.version import __version__

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
def health(
    pipeline: ArcBodyPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
    caller: Caller | None = Depends(authenticate_optional),
) -> HealthResponse:
    """Report readiness *and* the caveats that affect result quality.

    Deliberately more than a 200: a service running an untrained encoder or a
    fallback perception backend is up but not fully useful, and saying so here
    means an operator finds out from a health check rather than from a user
    complaining that similarity scores look random.
    """
    # Reachable without a key so an orchestrator can probe liveness, but the
    # enrolment counts are only for someone who could have read them anyway.
    if caller is not None:
        persons, profiles = pipeline.gallery.count()
    else:
        persons, profiles = -1, -1
    warnings: list[str] = []
    if not settings.security.api_keys:
        warnings.append(
            "no API keys configured: the /v1 endpoints are open to anyone who can "
            "reach this process"
        )
    if not pipeline.encoder.trained:
        warnings.append(
            "encoder is untrained: embeddings and similarity scores are not meaningful"
        )
    if pipeline.perception.name == "classic":
        warnings.append(
            "using the classic silhouette backend: it requires a plain, contrasting "
            "background. Install the 'yolo' extra for photographs in the wild."
        )
    return HealthResponse(
        status="degraded" if warnings else "ok",
        version=__version__,
        perception_backend=pipeline.perception.name,
        encoder_trained=pipeline.encoder.trained,
        embedding_dim=settings.embedding.dim,
        persons=persons,
        profiles=profiles,
        warnings=warnings,
    )
