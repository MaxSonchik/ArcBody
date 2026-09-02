"""Process-wide singletons for the HTTP layer.

The pipeline holds two loaded neural models and an open database handle, so it
is built once at startup rather than per request. Construction is guarded by a
lock because uvicorn's worker threads can race the first request.

``get_pipeline`` deliberately takes no arguments. FastAPI introspects the
signature of anything passed to ``Depends`` and treats a Pydantic-model
parameter with a default as a *request body field* — a ``settings`` argument
here silently turned every endpoint's body into ``{"request": ..., "settings":
...}`` and made every call fail validation. Configuration is injected through
:func:`set_pipeline` instead.
"""

from __future__ import annotations

import threading

from arcbody.config import Settings, get_settings
from arcbody.pipeline import ArcBodyPipeline

_lock = threading.Lock()
_pipeline: ArcBodyPipeline | None = None


def build_pipeline(settings: Settings | None = None) -> ArcBodyPipeline:
    """Construct a pipeline without touching the singleton."""
    return ArcBodyPipeline(settings or get_settings())


def get_pipeline() -> ArcBodyPipeline:
    """The shared pipeline, built on first use. Used as a FastAPI dependency."""
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                _pipeline = build_pipeline()
    return _pipeline


def set_pipeline(pipeline: ArcBodyPipeline | None) -> None:
    """Install a pipeline explicitly, replacing any existing one."""
    global _pipeline
    with _lock:
        if _pipeline is not None and _pipeline is not pipeline:
            _pipeline.close()
        _pipeline = pipeline


def reset_pipeline() -> None:
    """Drop the singleton. Used by tests and by the app's shutdown hook."""
    set_pipeline(None)
