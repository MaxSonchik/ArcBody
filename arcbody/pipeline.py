"""The orchestration layer: images in, body profiles out.

Everything above this module is transport and everything below is a single
concern, so this is the one place that knows the whole story — that a photo
becomes an observation, an observation becomes a crop and a silhouette, and
those become an embedding and a set of measurements that are then either
returned, enrolled, or compared against something already enrolled.

The comparison is the point of the service, not an afterthought. A generated
image is checked against an enrolled body the same way a second photograph would
be: encode it, take the cosine, and diff the proportions. That gives a number for
"did the generator keep the body" that does not depend on anyone eyeballing the
result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from arcbody.config import Settings, get_settings
from arcbody.embed.encoder import BodyEncoder, EncodedBody, cosine_similarity, fuse, view_agreement
from arcbody.errors import (
    AmbiguousSubjectError,
    InsufficientQualityError,
    NoPersonFoundError,
)
from arcbody.gallery.store import Gallery, Match
from arcbody.genai import controlmaps, prompt
from arcbody.measure import anthropometry
from arcbody.measure.quality import assess
from arcbody.measure.schema import RATIO_NAMES
from arcbody.perception.base import PerceptionBackend
from arcbody.perception.registry import build_backend
from arcbody.types import BodyProfile, PersonObservation, QualityReport, ViewLabel

logger = logging.getLogger(__name__)


@dataclass
class ImageInput:
    """One photo and what the client says about it."""

    image: np.ndarray
    view: ViewLabel | None = None
    label: str | None = None


@dataclass
class AnalysisResult:
    """A profile plus everything the API layer needs to render a response."""

    profile: BodyProfile
    prompt: prompt.PromptBundle
    negative_prompt: str
    control_maps: controlmaps.ControlMaps | None
    per_image_quality: list[QualityReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    view_agreement: float = 1.0


@dataclass
class GenerationCheck:
    """Whether a generated image still shows the enrolled body.

    ``similarity`` is the headline: cosine between the reference signature and
    the generated image's embedding. The ratio deltas say *how* it drifted when
    it did, which is what turns a failing score into an actionable prompt fix.
    """

    person_id: str
    similarity: float
    threshold: float
    matched: bool
    verdict: str
    ratio_deltas: dict[str, float] = field(default_factory=dict)
    worst_ratios: list[tuple[str, float]] = field(default_factory=list)
    quality: QualityReport | None = None
    notes: list[str] = field(default_factory=list)


class ArcBodyPipeline:
    """Holds the loaded models and the gallery for the lifetime of the process."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.perception: PerceptionBackend = build_backend(self.settings.perception)
        self.encoder = BodyEncoder(self.settings.embedding)
        self.gallery = Gallery(self.settings.gallery)
        logger.info(
            "pipeline ready: perception=%s encoder_trained=%s dim=%d",
            self.perception.name,
            self.encoder.trained,
            self.settings.embedding.dim,
        )

    def close(self) -> None:
        self.gallery.close()

    # -- core -------------------------------------------------------------

    def observe(self, item: ImageInput) -> tuple[PersonObservation, QualityReport]:
        """Find the subject in one photo and judge whether it is measurable."""
        result = self.perception.analyse(item.image)
        if not result.observations:
            raise NoPersonFoundError(
                "no person found in the image",
                backend=self.perception.name,
            )

        subject = result.observations[0]
        if len(result.observations) > 1:
            runner_up = result.observations[1]
            if runner_up.bbox.area * self.settings.perception.subject_dominance > subject.bbox.area:
                raise AmbiguousSubjectError(
                    "several people are equally prominent; crop to the subject",
                    detected=len(result.observations),
                )

        # A client-declared view beats inference. They know whether they asked
        # the person to turn, and a misread view silently swaps a breadth for a
        # depth in the girth model.
        if item.view is not None and item.view is not ViewLabel.UNKNOWN:
            subject.view = item.view

        report = assess(subject, self.settings.quality, self.settings.perception)
        return subject, report

    def analyse(
        self,
        items: list[ImageInput],
        *,
        stature_cm: float | None = None,
        weight_kg: float | None = None,
        want_control_maps: bool = True,
        strict_quality: bool = True,
    ) -> AnalysisResult:
        """Run the full pipeline over one subject's photos."""
        if not items:
            raise NoPersonFoundError("no images supplied")
        if len(items) > self.settings.max_images_per_request:
            raise NoPersonFoundError(
                "too many images in one request",
                supplied=len(items),
                maximum=self.settings.max_images_per_request,
            )

        observations: list[PersonObservation] = []
        reports: list[QualityReport] = []
        usable: list[tuple[np.ndarray, PersonObservation]] = []
        warnings: list[str] = []

        for index, item in enumerate(items):
            observation, report = self.observe(item)
            observations.append(observation)
            reports.append(report)
            if not report.passed:
                warnings.extend(f"image {index}: {hint}" for hint in report.hints())
            usable.append((item.image, observation))

        best_index = int(np.argmax([report.score for report in reports]))
        best_report = reports[best_index]
        if strict_quality and best_report.score < self.settings.quality.reject_below:
            raise InsufficientQualityError(
                "no supplied photo is good enough to measure",
                score=round(best_report.score, 3),
                required=self.settings.quality.reject_below,
                hints=best_report.hints(),
            )

        encoded = self.encoder.encode_batch(usable)
        embeddings = [item.embedding for item in encoded]
        agreement = view_agreement(embeddings)
        if len(embeddings) > 1 and agreement < 0.75:
            warnings.append(
                "the supplied photos do not look like the same body "
                f"(view agreement {agreement:.2f}); check they are one person"
            )

        stature = anthropometry.validate_stature(stature_cm, self.settings.measurement)
        measurements = anthropometry.estimate(
            observations,
            best_report,
            stature_cm=stature,
            weight_kg=weight_kg,
            settings=self.settings.measurement,
            perception=self.settings.perception,
        )
        self._corroborate(measurements, encoded, warnings)

        bundle = prompt.build(measurements, self.settings.genai)
        maps = (
            controlmaps.build(
                usable[best_index][0], observations[best_index], self.settings.genai
            )
            if want_control_maps
            else None
        )

        profile = BodyProfile(
            embedding=fuse(embeddings),
            measurements=measurements,
            quality=best_report,
            observations=observations,
            prompt=bundle.prompt,
            metadata={
                "perception_backend": self.perception.name,
                "views": [observation.view.value for observation in observations],
                "image_count": len(items),
                **encoded[best_index].as_metadata(),
            },
        )
        return AnalysisResult(
            profile=profile,
            prompt=bundle,
            negative_prompt=prompt.negative_prompt(measurements),
            control_maps=maps,
            per_image_quality=reports,
            warnings=warnings,
            view_agreement=agreement,
        )

    def _corroborate(
        self, measurements, encoded: list[EncodedBody], warnings: list[str]
    ) -> None:
        """Compare the geometric ratios against the encoder's own reading.

        Two independent paths produce the same quantity: one from silhouette
        geometry, one learned from pixels. Where they disagree sharply, at least
        one is wrong, and the caller deserves to know that before feeding the
        number to a generator. Nothing is silently overridden — a disagreement is
        reported, not resolved.
        """
        if not encoded or not encoded[0].trained or not measurements.ratios:
            return
        predicted = {
            name: float(np.mean([item.ratios.get(name, 0.0) for item in encoded]))
            for name in RATIO_NAMES
        }
        disagreements = []
        for name, geometric in measurements.ratios.items():
            neural = predicted.get(name)
            if not neural or geometric <= 0:
                continue
            relative = abs(neural - geometric) / geometric
            if relative > 0.25:
                disagreements.append((name, round(relative, 3)))
        if disagreements:
            worst = sorted(disagreements, key=lambda item: -item[1])[:3]
            warnings.append(
                "geometric and learned estimates disagree on "
                + ", ".join(f"{name} ({value:.0%})" for name, value in worst)
            )
            measurements.notes.append("estimates corroborated: disagreement flagged")
        else:
            measurements.notes.append("estimates corroborated by the encoder")

    # -- gallery operations ----------------------------------------------

    def enrol(
        self,
        person_id: str,
        result: AnalysisResult,
        *,
        external_face_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Store a body profile under a caller-owned person id."""
        profile = result.profile
        if not profile.has_embedding:
            raise InsufficientQualityError("cannot enrol a profile without an embedding")
        measurements = {
            name: value.value_cm for name, value in profile.measurements.values.items()
        }
        return self.gallery.enrol(
            person_id,
            profile.embedding,
            ratios=profile.measurements.ratios,
            measurements=measurements,
            quality=profile.quality.score,
            view=profile.observations[0].view.value if profile.observations else "unknown",
            external_face_id=external_face_id,
            metadata=metadata,
        )

    def identify(self, result: AnalysisResult, *, top_k: int | None = None) -> list[Match]:
        if not result.profile.has_embedding:
            return []
        return self.gallery.identify(result.profile.embedding, top_k=top_k)

    def check_generation(
        self, person_id: str, result: AnalysisResult, *, threshold: float | None = None
    ) -> GenerationCheck:
        """Score a generated image against an enrolled body.

        The verdict has three levels rather than a pass/fail, because the
        failures differ in kind. A body that is recognisably the same but has
        drifted in proportion is a prompt problem the caller can fix; a body that
        is simply someone else is a generation to discard.
        """
        record = self.gallery.get(person_id)
        threshold = (
            threshold if threshold is not None else self.settings.embedding.match_threshold
        )
        similarity = cosine_similarity(result.profile.embedding, record.embedding)

        deltas: dict[str, float] = {}
        for name, reference in record.ratios.items():
            observed = result.profile.measurements.ratios.get(name)
            if observed is None or reference in (None, 0):
                continue
            deltas[name] = round((observed - reference) / reference, 4)

        worst = sorted(deltas.items(), key=lambda item: -abs(item[1]))[:5]
        drifted = [name for name, value in deltas.items() if abs(value) > 0.15]

        notes: list[str] = []
        if not self.encoder.trained:
            notes.append(
                "the encoder is untrained, so the similarity score is not meaningful; "
                "train a checkpoint before relying on this verdict"
            )
        if not record.ratios:
            notes.append("the enrolled profile carries no ratios to compare against")

        matched = similarity >= threshold
        if matched and not drifted:
            verdict = "consistent"
        elif matched:
            verdict = "drifted"
        else:
            verdict = "different_body"

        return GenerationCheck(
            person_id=person_id,
            similarity=round(similarity, 4),
            threshold=threshold,
            matched=matched,
            verdict=verdict,
            ratio_deltas=deltas,
            worst_ratios=worst,
            quality=result.profile.quality,
            notes=notes,
        )
