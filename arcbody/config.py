"""Runtime configuration.

Everything is overridable through ``ARCBODY_*`` environment variables so the
container needs no config file.  Nested settings use ``__`` as the delimiter,
e.g. ``ARCBODY_PERCEPTION__BACKEND=yolo``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PerceptionBackend = Literal["auto", "yolo", "classic"]
EncoderBackend = Literal["auto", "torch", "none"]


class PerceptionSettings(BaseModel):
    """Person detection, pose and silhouette extraction."""

    backend: PerceptionBackend = "auto"
    # Ultralytics checkpoints. Fetched once from the GitHub release assets and
    # then cached; bake them into the image for air-gapped deployments.
    pose_weights: str = "yolov8n-pose.pt"
    seg_weights: str = "yolov8n-seg.pt"
    device: str = "cpu"
    # A detection below this score is not considered a person at all.
    min_detection_score: float = 0.35
    # A keypoint below this score is treated as unobserved.
    min_keypoint_score: float = 0.30
    # When several people are found, the subject must own at least this share of
    # the largest mask area, otherwise the request is ambiguous.
    subject_dominance: float = 1.6


class QualitySettings(BaseModel):
    """Gates a photo must pass before its measurements are trusted.

    These are not cosmetic: pixel-to-centimetre scaling assumes an upright,
    fully-visible, roughly fronto-parallel subject.  Violating that does not
    make the numbers noisy, it makes them wrong, so we reject instead.
    """

    # Subject height as a fraction of image height. Too small -> not enough
    # pixels for silhouette widths to mean anything.
    min_subject_height_ratio: float = 0.35
    # Fraction of COCO keypoints that must be observed.
    min_keypoint_coverage: float = 0.70
    # Max tilt of the torso axis away from vertical, in degrees.
    max_torso_tilt_deg: float = 20.0
    # Max |x| offset of the subject centre from the image centre, in subject
    # widths. Large offsets mean strong perspective at the frame edge.
    max_off_centre: float = 0.75
    # Below this overall score the request is rejected outright.
    reject_below: float = 0.45


class MeasurementSettings(BaseModel):
    """Anthropometric estimation."""

    # Plausible stature range accepted from the client, in centimetres.
    min_stature_cm: float = 120.0
    max_stature_cm: float = 230.0
    # Relative 1-sigma uncertainty of a girth measured from a single frontal
    # view; widths measured directly are roughly half as uncertain.
    girth_sigma_single_view: float = 0.075
    girth_sigma_two_view: float = 0.040
    width_sigma: float = 0.035
    # Reported interval half-width, in sigmas (1.96 -> ~95%).
    ci_sigmas: float = 1.96


class EmbeddingSettings(BaseModel):
    """The ArcBody encoder."""

    backend: EncoderBackend = "auto"
    dim: int = 256
    # Body crops keep the 1:2 aspect ratio conventional for person re-ID.
    input_width: int = 128
    input_height: int = 256
    weights_path: Path | None = None
    device: str = "cpu"
    # ArcFace margin hyper-parameters, used at training time only.
    arc_scale: float = 30.0
    arc_margin: float = 0.30
    # Cosine similarity above which two crops are called the same body.
    match_threshold: float = 0.55


class GallerySettings(BaseModel):
    """Enrolled body profiles."""

    database_path: Path = Path("var/arcbody.sqlite3")
    # Brute-force cosine search is linear but vectorised; at 256 floats per
    # record, 100k records is ~100 MB and a few milliseconds per query.
    max_identify_candidates: int = 200_000
    default_top_k: int = 5


class GenAISettings(BaseModel):
    """Prompt and control-map generation."""

    control_map_width: int = 512
    control_map_height: int = 768
    # Emit a measurement in the prompt only when its relative CI is tighter
    # than this. The number is a judgement about *steering*, not about tailoring:
    # a single-view girth carries roughly a +/-17% interval, so a stricter gate
    # drops every chest, waist and hip measurement and leaves the prompt with
    # nothing but height. A wide-but-present girth still pulls a generator
    # towards the right body; silence lets it default to its training mean.
    # Tighten this once the sigmas in MeasurementSettings have been refitted
    # against measured subjects.
    prompt_max_relative_ci: float = 0.20
    units: Literal["metric", "imperial", "both"] = "metric"


class SecuritySettings(BaseModel):
    """Who may call the service, and how often.

    Body embeddings are biometric data, so the default posture is that an
    unauthenticated ``/v1`` surface is a misconfiguration rather than a
    convenience. In ``prod`` the service refuses to start without keys; in
    ``dev`` and ``staging`` it starts, disables auth, and says so loudly on
    ``/healthz`` — silence there would be the worst of both.
    """

    #: Accepted API keys. Set ``ARCBODY_SECURITY__API_KEYS`` to a JSON array or
    #: a comma-separated list. Empty means "no authentication", which is only
    #: permitted outside production.
    #:
    #: ``NoDecode`` turns off the settings source's own JSON parsing, which
    #: otherwise rejects ``key1,key2`` before any validator sees it — the
    #: comma-separated form has to be handled here or not at all.
    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    header_name: str = "X-API-Key"

    #: Sustained rate per caller, and the burst it may spend at once.
    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 30
    #: Rate limiting is per process. Behind several replicas each gets its own
    #: budget, so a shared store is needed before the limit means anything
    #: globally. Set false to disable rather than to pretend.
    rate_limit_enabled: bool = True

    @field_validator("api_keys", mode="before")
    @classmethod
    def _parse_keys(cls, value: object) -> object:
        """Accept a JSON array or a comma-separated list."""
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text.startswith("["):
            import json

            return json.loads(text)
        return [item.strip() for item in text.split(",") if item.strip()]


class BatchSettings(BaseModel):
    """Asynchronous batch jobs."""

    max_items_per_job: int = 256
    max_concurrent_jobs: int = 4
    worker_threads: int = 2
    # Finished jobs are evicted this long after completion.
    retain_seconds: int = 3600
    # Job state lives in the gallery database so a completed job can still be
    # polled after a restart. Photographs are never persisted, so a job
    # interrupted mid-flight cannot resume and is failed explicitly instead.
    persist: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARCBODY_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    # Hard cap on decoded pixels, to bound memory per request.
    max_image_pixels: int = 40_000_000
    max_images_per_request: int = 8

    perception: PerceptionSettings = Field(default_factory=PerceptionSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    measurement: MeasurementSettings = Field(default_factory=MeasurementSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    gallery: GallerySettings = Field(default_factory=GallerySettings)
    genai: GenAISettings = Field(default_factory=GenAISettings)
    batch: BatchSettings = Field(default_factory=BatchSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read from the environment once."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings. Tests use this after patching the environment."""
    get_settings.cache_clear()
