"""Pydantic models for the HTTP boundary.

These mirror the dataclasses in :mod:`arcbody.types` rather than replacing them.
The internal types carry numpy arrays and exist to be computed with; these exist
to be validated, documented and serialised, and keeping them apart means a
change to the wire format cannot quietly alter the maths.

Every response that carries a number also carries the means to judge it: an
interval and a source on each measurement, a scored quality report with
actionable hints, and an explicit flag when the encoder behind an embedding is
untrained.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from arcbody.types import (
    BodyMeasurements,
    GateResult,
    MeasurementValue,
    QualityReport,
    ViewLabel,
)

ViewName = Literal["front", "side", "back", "auto"]


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------


class ImagePayload(BaseModel):
    """One photo, base64-encoded, with what the client knows about it."""

    content_base64: str = Field(
        ...,
        description="Image bytes, base64-encoded. A 'data:image/...;base64,' prefix is accepted.",
    )
    view: ViewName = Field(
        "auto",
        description=(
            "Which way the subject faces. 'auto' infers it from the pose, but a declared "
            "view is trusted over inference: getting this wrong swaps a breadth for a "
            "depth in the girth model."
        ),
    )
    label: str | None = Field(None, max_length=128, description="Caller's own label, echoed back.")

    @field_validator("content_base64")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content_base64 must not be empty")
        return value

    def view_label(self) -> ViewLabel | None:
        return None if self.view == "auto" else ViewLabel(self.view)


class SubjectFields(BaseModel):
    """What the client knows that a photograph cannot reveal."""

    stature_cm: Annotated[float, Field(ge=120, le=230)] | None = Field(
        None,
        description=(
            "The subject's height. Without it the response carries proportions only: "
            "a single photo determines shape, never size."
        ),
    )
    weight_kg: Annotated[float, Field(ge=25, le=350)] | None = Field(
        None, description="Optional; used only for the BMI term in the prompt."
    )


class AnalyzeRequest(SubjectFields):
    """Measure and encode one subject from one or more photos."""

    images: list[ImagePayload] = Field(..., min_length=1, max_length=8)
    include_control_maps: bool = Field(
        True, description="Return pose, silhouette and normalised-crop PNGs."
    )
    include_embedding: bool = Field(
        True, description="Return the raw embedding vector."
    )
    strict_quality: bool = Field(
        True,
        description=(
            "Reject the request when no photo passes the quality gates. Turning this off "
            "returns measurements anyway, with the failed gates listed — useful for "
            "debugging a capture flow, not for production numbers."
        ),
    )


class EnrolRequest(AnalyzeRequest):
    """Analyse and store the result under a person id."""

    external_face_id: str | None = Field(
        None,
        max_length=128,
        description=(
            "Opaque identifier owned by your face-recognition service. ArcBody stores it "
            "alongside the body profile so the two can be joined, and never interprets it."
        ),
    )
    metadata: dict[str, str] = Field(default_factory=dict, description="Caller-owned key/values.")


class IdentifyRequest(AnalyzeRequest):
    """Find the closest enrolled bodies."""

    top_k: Annotated[int, Field(ge=1, le=50)] = 5
    minimum_similarity: Annotated[float, Field(ge=-1.0, le=1.0)] = -1.0


class GenerationCheckRequest(AnalyzeRequest):
    """Score a generated image against an enrolled body."""

    threshold: Annotated[float, Field(ge=-1.0, le=1.0)] | None = Field(
        None, description="Overrides the configured match threshold for this request."
    )


class BatchItem(SubjectFields):
    """One subject inside a batch job."""

    reference: str = Field(..., max_length=128, description="Caller's id for this item.")
    images: list[ImagePayload] = Field(..., min_length=1, max_length=8)
    person_id: str | None = Field(
        None, max_length=128, description="If set, the result is enrolled under this id."
    )


class BatchRequest(BaseModel):
    """Submit many subjects for asynchronous processing."""

    items: list[BatchItem] = Field(..., min_length=1)
    include_control_maps: bool = Field(
        False,
        description=(
            "Off by default in batch: three PNGs per subject makes a large job's result "
            "document enormous. Re-analyse the ones you want maps for."
        ),
    )
    strict_quality: bool = True


# ---------------------------------------------------------------------------
# responses
# ---------------------------------------------------------------------------


class GateModel(BaseModel):
    name: str
    value: float
    threshold: float
    passed: bool
    explanation: str

    @classmethod
    def of(cls, gate: GateResult) -> GateModel:
        return cls(
            name=gate.name,
            value=round(gate.value, 4),
            threshold=gate.threshold,
            passed=gate.passed,
            explanation=gate.explanation,
        )


class QualityModel(BaseModel):
    score: float
    passed: bool
    gates: list[GateModel]
    hints: list[str]

    @classmethod
    def of(cls, report: QualityReport) -> QualityModel:
        return cls(
            score=report.score,
            passed=report.passed,
            gates=[GateModel.of(gate) for gate in report.gates],
            hints=report.hints(),
        )


class MeasurementModel(BaseModel):
    value_cm: float
    ci_low_cm: float
    ci_high_cm: float
    relative_ci: float = Field(
        ..., description="Interval half-width over the value; 0.10 means plus or minus 10%."
    )
    source: str = Field(
        ...,
        description=(
            "How it was obtained. 'ellipse_prior' means the depth was assumed from a "
            "population table rather than measured — supply a side photo to replace it."
        ),
    )
    confidence: float

    @classmethod
    def of(cls, value: MeasurementValue) -> MeasurementModel:
        return cls(
            value_cm=value.value_cm,
            ci_low_cm=value.ci_low_cm,
            ci_high_cm=value.ci_high_cm,
            relative_ci=round(value.relative_ci, 4),
            source=value.source.value,
            confidence=value.confidence,
        )


class MeasurementsModel(BaseModel):
    stature_cm: float | None
    weight_kg: float | None
    somatotype: str | None
    ratios: dict[str, float]
    values: dict[str, MeasurementModel]
    notes: list[str]

    @classmethod
    def of(cls, measurements: BodyMeasurements) -> MeasurementsModel:
        return cls(
            stature_cm=measurements.stature_cm,
            weight_kg=measurements.weight_kg,
            somatotype=measurements.somatotype,
            ratios={name: round(value, 5) for name, value in measurements.ratios.items()},
            values={
                name: MeasurementModel.of(value) for name, value in measurements.values.items()
            },
            notes=measurements.notes,
        )


class PromptModel(BaseModel):
    prompt: str
    negative_prompt: str
    build_phrase: str
    proportion_phrases: list[str]
    measurement_phrases: list[str]
    omitted_for_low_confidence: list[str]


class ControlMapsModel(BaseModel):
    """Base64 PNGs, all in one shared frame so they overlay exactly."""

    pose: str
    silhouette: str
    normalised_crop: str
    width: int
    height: int


class AnalyzeResponse(BaseModel):
    embedding: list[float] | None
    embedding_dim: int
    encoder_trained: bool = Field(
        ...,
        description=(
            "False when no checkpoint was loaded. The embedding is then reproducible but "
            "close to meaningless, and similarity scores must not be relied on."
        ),
    )
    view_agreement: float
    measurements: MeasurementsModel
    prompt: PromptModel
    control_maps: ControlMapsModel | None
    quality: QualityModel
    per_image_quality: list[QualityModel]
    warnings: list[str]
    metadata: dict[str, object]


class EnrolResponse(BaseModel):
    person_id: str
    profile_id: str
    profile_count: int
    external_face_id: str | None
    analysis: AnalyzeResponse


class MatchModel(BaseModel):
    person_id: str
    similarity: float
    profile_count: int
    external_face_id: str | None


class IdentifyResponse(BaseModel):
    matches: list[MatchModel]
    threshold: float = Field(
        ..., description="Similarity above which the service considers a match the same body."
    )
    encoder_trained: bool
    analysis: AnalyzeResponse


class GenerationCheckResponse(BaseModel):
    person_id: str
    similarity: float
    threshold: float
    matched: bool
    verdict: Literal["consistent", "drifted", "different_body"] = Field(
        ...,
        description=(
            "'consistent' is the same body within tolerance; 'drifted' is recognisably the "
            "same body with proportions that moved, which is a prompt problem; "
            "'different_body' is a generation to discard."
        ),
    )
    ratio_deltas: dict[str, float]
    worst_ratios: list[tuple[str, float]]
    notes: list[str]
    analysis: AnalyzeResponse


class PersonModel(BaseModel):
    person_id: str
    profile_count: int
    external_face_id: str | None
    ratios: dict[str, float]
    measurements: dict[str, float]
    metadata: dict[str, object]
    created_at: str
    updated_at: str


class PersonListResponse(BaseModel):
    persons: list[PersonModel]
    total_persons: int
    total_profiles: int


class BatchItemResult(BaseModel):
    reference: str
    status: Literal["ok", "failed"]
    person_id: str | None = None
    profile_id: str | None = None
    analysis: AnalyzeResponse | None = None
    error: dict[str, object] | None = None


class BatchJobModel(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    submitted: int
    completed: int
    failed: int
    created_at: str
    finished_at: str | None
    results: list[BatchItemResult] | None = Field(
        None, description="Present once the job has finished."
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    perception_backend: str
    encoder_trained: bool
    embedding_dim: int
    persons: int
    profiles: int
    warnings: list[str]


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)
