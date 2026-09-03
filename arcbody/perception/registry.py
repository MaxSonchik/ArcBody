"""Choosing a perception backend."""

from __future__ import annotations

import logging

from arcbody.config import PerceptionSettings
from arcbody.errors import BackendUnavailableError
from arcbody.perception.base import PerceptionBackend
from arcbody.perception.classic import ClassicPerception
from arcbody.perception.yolo import YoloPerception

logger = logging.getLogger(__name__)


def build_backend(settings: PerceptionSettings) -> PerceptionBackend:
    """Instantiate the configured backend.

    ``auto`` prefers YOLO and falls back to the classic silhouette path when its
    extras are absent — a deployment that installed them gets the good detector
    without configuration, and one that did not still starts. An explicit
    ``yolo`` never falls back: if you asked for it and it cannot run, that is a
    misconfiguration worth failing on rather than silently degrading the
    accuracy of every measurement the service returns.
    """
    if settings.backend == "classic":
        return ClassicPerception(settings)

    yolo = YoloPerception(settings)
    if settings.backend == "yolo":
        if not yolo.available():
            raise BackendUnavailableError(
                "perception backend 'yolo' was requested but ultralytics/opencv are missing",
                backend="yolo",
            )
        return yolo

    if yolo.available():
        return yolo
    logger.warning(
        "ultralytics/opencv not installed; falling back to the classic silhouette backend, "
        "which requires a plain background"
    )
    return ClassicPerception(settings)
