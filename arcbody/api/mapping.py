"""Translating pipeline results into wire models."""

from __future__ import annotations

from arcbody.config import Settings
from arcbody.gallery.store import Match, PersonRecord
from arcbody.pipeline import AnalysisResult, GenerationCheck
from arcbody.schemas import (
    AnalyzeResponse,
    ControlMapsModel,
    GenerationCheckResponse,
    MatchModel,
    MeasurementsModel,
    PersonModel,
    PromptModel,
    QualityModel,
)


def analysis_response(
    result: AnalysisResult, settings: Settings, *, include_embedding: bool = True
) -> AnalyzeResponse:
    profile = result.profile
    maps = None
    if result.control_maps is not None:
        encoded = result.control_maps.as_png_base64()
        maps = ControlMapsModel(
            **encoded,
            width=settings.genai.control_map_width,
            height=settings.genai.control_map_height,
        )

    embedding = None
    if include_embedding and profile.has_embedding:
        embedding = [round(float(value), 6) for value in profile.embedding]

    return AnalyzeResponse(
        embedding=embedding,
        embedding_dim=int(profile.embedding.size) if profile.has_embedding else 0,
        encoder_trained=bool(profile.metadata.get("encoder_trained", False)),
        view_agreement=round(result.view_agreement, 4),
        measurements=MeasurementsModel.of(profile.measurements),
        prompt=PromptModel(
            prompt=result.prompt.prompt,
            negative_prompt=result.negative_prompt,
            build_phrase=result.prompt.build_phrase,
            proportion_phrases=result.prompt.proportion_phrases,
            measurement_phrases=result.prompt.measurement_phrases,
            omitted_for_low_confidence=result.prompt.omitted,
        ),
        control_maps=maps,
        quality=QualityModel.of(profile.quality),
        per_image_quality=[QualityModel.of(report) for report in result.per_image_quality],
        warnings=result.warnings,
        metadata=profile.metadata,
    )


def match_model(match: Match) -> MatchModel:
    return MatchModel(
        person_id=match.person_id,
        similarity=round(match.similarity, 4),
        profile_count=match.profile_count,
        external_face_id=match.external_face_id,
    )


def person_model(record: PersonRecord) -> PersonModel:
    return PersonModel(
        person_id=record.person_id,
        profile_count=record.profile_count,
        external_face_id=record.external_face_id,
        ratios=record.ratios,
        measurements={
            name: float(value)
            for name, value in record.measurements.items()
            if isinstance(value, int | float)
        },
        metadata=record.metadata,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def generation_check_response(
    check: GenerationCheck, analysis: AnalyzeResponse
) -> GenerationCheckResponse:
    return GenerationCheckResponse(
        person_id=check.person_id,
        similarity=check.similarity,
        threshold=check.threshold,
        matched=check.matched,
        verdict=check.verdict,  # type: ignore[arg-type]
        ratio_deltas=check.ratio_deltas,
        worst_ratios=check.worst_ratios,
        notes=check.notes,
        analysis=analysis,
    )
