"""The ArcBody FastAPI application."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from arcbody.api.deps import get_pipeline, reset_pipeline
from arcbody.api.jobs import get_job_store, reset_job_store
from arcbody.api.routes import analyze, batch, health, persons
from arcbody.api.security import (
    RateLimitedError,
    authenticate,
    reset_rate_limiter,
)
from arcbody.config import get_settings
from arcbody.errors import ArcBodyError
from arcbody.measure.anthropometry import ScaleError
from arcbody.version import __version__

logger = logging.getLogger(__name__)

DESCRIPTION = """
ArcBody turns a photograph of a person into three things a generative pipeline
can use: a **body embedding** that identifies the shape, **anthropometry** in
centimetres, and **control maps** plus a **prompt** that steer a generator
towards that body.

It is the body-side counterpart to a face-recognition service, not a replacement
for one. Face identity stays where it already lives; ArcBody stores an opaque
`external_face_id` so the two can be joined.

**Authentication.** Every `/v1` endpoint expects an `X-API-Key` header. The
service refuses to start in production without keys configured, and reports
`degraded` on `/healthz` when it is running without them elsewhere.

**Two things worth knowing before you integrate.**

A photograph determines a body's *proportions*, never its *size*. Send
`stature_cm` and you get centimetres; omit it and you get ratios. Nothing here
guesses how tall someone is.

A girth is never measured from one photo, only reconstructed: a camera sees
breadth, not depth. Single-view girths use a population depth prior and say so
in their `source` field, with an interval to match. Add a side view and the
depth becomes a measurement.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Load models at startup rather than on the first request, so a container
    # that reports ready actually is.
    pipeline = get_pipeline()
    get_job_store(settings)
    logger.info(
        "arcbody %s ready (perception=%s, encoder_trained=%s)",
        __version__,
        pipeline.perception.name,
        pipeline.encoder.trained,
    )
    try:
        yield
    finally:
        reset_job_store()
        reset_pipeline()
        reset_rate_limiter()


def create_app() -> FastAPI:
    app = FastAPI(
        title="ArcBody",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    @app.exception_handler(ArcBodyError)
    async def _arcbody_error(request: Request, error: ArcBodyError) -> JSONResponse:
        """Map domain errors to their own status codes, keeping the code stable."""
        headers: dict[str, str] = {}
        if isinstance(error, RateLimitedError):
            # Without Retry-After a throttled client retries immediately and
            # makes the situation it is being throttled for worse.
            wait = error.details.get("retry_after_seconds", 1)
            seconds = float(wait) if isinstance(wait, int | float) else 1.0
            headers["Retry-After"] = str(max(1, int(seconds) + 1))
        return JSONResponse(
            status_code=error.http_status, content=error.to_dict(), headers=headers
        )

    @app.exception_handler(ScaleError)
    async def _scale_error(request: Request, error: ScaleError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"code": "invalid_stature", "message": str(error), "details": {}},
        )

    # Authentication is attached to the routers, not to individual routes, so a
    # new endpoint cannot be added unprotected by forgetting a decorator.
    protected = [Depends(authenticate)]
    app.include_router(health.router)
    app.include_router(analyze.router, dependencies=protected)
    app.include_router(persons.router, dependencies=protected)
    app.include_router(batch.router, dependencies=protected)
    return app


app = create_app()
