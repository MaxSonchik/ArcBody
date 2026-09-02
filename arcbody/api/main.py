"""The ArcBody FastAPI application."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from arcbody.api.deps import get_pipeline, reset_pipeline
from arcbody.api.jobs import get_job_store, reset_job_store
from arcbody.api.routes import analyze, batch, health, persons
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
    get_job_store(settings.batch)
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
        return JSONResponse(status_code=error.http_status, content=error.to_dict())

    @app.exception_handler(ScaleError)
    async def _scale_error(request: Request, error: ScaleError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"code": "invalid_stature", "message": str(error), "details": {}},
        )

    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(persons.router)
    app.include_router(batch.router)
    return app


app = create_app()
