"""Shared domain vocabulary.

These are the types that cross module boundaries: perception produces a
``PersonObservation``, measurement turns observations into ``BodyMeasurements``,
the encoder turns them into an embedding, and the two meet in a ``BodyProfile``.

They are plain dataclasses rather than pydantic models on purpose — they carry
numpy arrays and live entirely inside the process. ``arcbody.schemas`` holds the
pydantic mirror that crosses the HTTP boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from arcbody import keypoints as kp


class ViewLabel(StrEnum):
    """Which way the subject faces the camera."""

    FRONT = "front"
    SIDE = "side"
    BACK = "back"
    UNKNOWN = "unknown"


class MeasurementSource(StrEnum):
    """How a measurement was obtained, so callers can weight it.

    ``SILHOUETTE`` is measured off the segmentation mask; ``SKELETON`` off
    keypoint distances; ``ELLIPSE_PRIOR`` is a girth reconstructed from one or
    two widths plus a population depth/breadth ratio; ``NEURAL`` comes from the
    encoder's auxiliary shape head; ``CLIENT`` was supplied in the request.
    """

    SILHOUETTE = "silhouette"
    SKELETON = "skeleton"
    ELLIPSE_PRIOR = "ellipse_prior"
    NEURAL = "neural"
    CLIENT = "client"


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates, ``x2``/``y2`` exclusive."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def clipped(self, width: int, height: int) -> BoundingBox:
        return BoundingBox(
            x1=float(np.clip(self.x1, 0, width)),
            y1=float(np.clip(self.y1, 0, height)),
            x2=float(np.clip(self.x2, 0, width)),
            y2=float(np.clip(self.y2, 0, height)),
        )

    def expanded(self, ratio: float, width: int, height: int) -> BoundingBox:
        """Grow the box by ``ratio`` on every side, then clip to the image."""
        dx, dy = self.width * ratio, self.height * ratio
        return BoundingBox(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy).clipped(
            width, height
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class GateResult:
    """One named quality gate: what was measured, what was required, and why."""

    name: str
    value: float
    threshold: float
    passed: bool
    explanation: str


@dataclass
class QualityReport:
    """The verdict on whether a photo can carry trustworthy measurements."""

    score: float
    gates: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    @property
    def failures(self) -> list[GateResult]:
        return [gate for gate in self.gates if not gate.passed]

    def hints(self) -> list[str]:
        """Actionable advice for the person who took the photo."""
        return [gate.explanation for gate in self.failures]


@dataclass
class PersonObservation:
    """One person, seen in one image.

    ``mask`` is optional: the skeleton path alone yields lengths and breadths at
    joint level, but girths need a silhouette.
    """

    bbox: BoundingBox
    keypoints: np.ndarray
    image_size: tuple[int, int]  # (width, height)
    detection_score: float = 1.0
    mask: np.ndarray | None = None
    view: ViewLabel = ViewLabel.UNKNOWN
    backend: str = "unknown"

    def __post_init__(self) -> None:
        self.keypoints = kp.validate(self.keypoints)
        if self.mask is not None:
            mask = np.asarray(self.mask)
            if mask.ndim != 2:
                raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
            self.mask = mask.astype(bool, copy=False)

    @property
    def has_mask(self) -> bool:
        return self.mask is not None and bool(self.mask.any())

    def observed(self, min_score: float) -> np.ndarray:
        return kp.observed(self.keypoints, min_score)


@dataclass(frozen=True)
class MeasurementValue:
    """A single anthropometric value with an honest uncertainty interval."""

    name: str
    value_cm: float
    ci_low_cm: float
    ci_high_cm: float
    source: MeasurementSource
    confidence: float

    @property
    def relative_ci(self) -> float:
        """Interval half-width relative to the value; 0.10 means +/-10%."""
        if self.value_cm <= 0:
            return float("inf")
        return (self.ci_high_cm - self.ci_low_cm) / (2.0 * self.value_cm)


@dataclass
class BodyMeasurements:
    """Anthropometry for one subject, plus the scale-free ratios behind it.

    ``ratios`` are the primary product: they are what the model actually sees
    and what survives an unknown stature. ``values`` are those ratios multiplied
    by the client-supplied stature, and only exist when a stature was given.
    """

    ratios: dict[str, float] = field(default_factory=dict)
    values: dict[str, MeasurementValue] = field(default_factory=dict)
    stature_cm: float | None = None
    weight_kg: float | None = None
    somatotype: str | None = None
    notes: list[str] = field(default_factory=list)

    def get(self, name: str) -> MeasurementValue | None:
        return self.values.get(name)

    def confident_values(self, max_relative_ci: float) -> dict[str, MeasurementValue]:
        return {
            name: value
            for name, value in self.values.items()
            if value.relative_ci <= max_relative_ci
        }


@dataclass
class BodyProfile:
    """Everything ArcBody knows about one subject from one request."""

    embedding: np.ndarray | None
    measurements: BodyMeasurements
    quality: QualityReport
    observations: list[PersonObservation] = field(default_factory=list)
    prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_embedding(self) -> bool:
        return self.embedding is not None and self.embedding.size > 0
