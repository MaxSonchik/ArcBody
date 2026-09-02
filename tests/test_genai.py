"""Prompt text and control maps."""

from __future__ import annotations

import numpy as np
import pytest

from arcbody.config import GenAISettings, PerceptionSettings, QualitySettings
from arcbody.genai import controlmaps, prompt
from arcbody.measure import anthropometry as anth
from arcbody.measure.calibration import REFERENCE_PERCENTILES, band_of, percentile_of
from arcbody.measure.quality import assess
from arcbody.measure.schema import RATIO_NAMES
from arcbody.types import BodyMeasurements


def measure(capture, perception, params):
    observation = perception.analyse(capture.image).subject
    report = assess(observation, QualitySettings(), PerceptionSettings())
    return observation, anth.estimate([observation], report, stature_cm=params.stature_cm)


def test_the_prompt_names_the_build_and_the_numbers(capture, perception, subject) -> None:
    _, measurements = measure(capture, perception, subject)
    bundle = prompt.build(measurements)
    assert bundle.build_phrase
    assert bundle.build_phrase in bundle.prompt
    assert "height" in bundle.prompt
    assert bundle.measurement_phrases, "girths must survive the confidence gate"


def test_low_confidence_measurements_are_dropped_not_rounded(
    capture, perception, subject
) -> None:
    _, measurements = measure(capture, perception, subject)
    strict = prompt.build(measurements, GenAISettings(prompt_max_relative_ci=0.001))
    # Height survives any gate and should: the client supplied it, so it is not
    # an estimate and carries no interval. Everything the service *estimated*
    # must go.
    assert strict.measurement_phrases == [f"height {measurements.stature_cm:.0f} cm"]
    assert strict.omitted, "dropping a measurement must be reported"
    assert "waist_girth" in strict.omitted


def test_the_prompt_never_describes_the_person(capture, perception, subject) -> None:
    """Age, sex and ethnicity are neither measured nor guessed."""
    _, measurements = measure(capture, perception, subject)
    text = prompt.build(measurements).prompt.lower()
    forbidden = ("male", "female", "man", "woman", "young", "old", "caucasian", "asian")
    assert not any(word in text.split() for word in forbidden)


def test_average_proportions_are_left_unsaid() -> None:
    """A phrase for every ratio would bury the ones that carry signal."""
    median = REFERENCE_PERCENTILES["shoulder_to_stature"][1]
    assert prompt.describe_ratio("shoulder_to_stature", median) is None


def test_descriptors_track_the_reference_distribution() -> None:
    low, median, high = REFERENCE_PERCENTILES["shoulder_to_stature"]
    assert percentile_of("shoulder_to_stature", median) == pytest.approx(0.5, abs=1e-6)
    assert percentile_of("shoulder_to_stature", low) < 0.2
    assert percentile_of("shoulder_to_stature", high) > 0.8
    assert band_of("shoulder_to_stature", low) < band_of("shoulder_to_stature", high)
    assert percentile_of("not_a_ratio", 1.0) is None


def test_every_reported_ratio_has_a_reference() -> None:
    """A ratio with no calibration entry silently loses its descriptor."""
    missing = set(RATIO_NAMES) - set(REFERENCE_PERCENTILES)
    assert not missing, f"uncalibrated ratios: {sorted(missing)}"


def test_the_negative_prompt_opposes_the_body_s_own_extremes() -> None:
    broad = BodyMeasurements(ratios={"shoulder_to_stature": 0.30, "hip_to_stature": 0.15})
    text = prompt.negative_prompt(broad)
    assert "narrow shoulder" in text
    assert "wide hip" in text
    assert "distorted proportions" in text


def test_units_can_be_switched() -> None:
    measurements = BodyMeasurements(stature_cm=180.0)
    from arcbody.types import MeasurementSource, MeasurementValue

    measurements.values["stature"] = MeasurementValue(
        "stature", 180.0, 180.0, 180.0, MeasurementSource.CLIENT, 1.0
    )
    assert "cm" in prompt.build(measurements, GenAISettings(units="metric")).prompt
    assert "in" in prompt.build(measurements, GenAISettings(units="imperial")).prompt


def test_control_maps_share_one_frame(capture, perception) -> None:
    observation = perception.analyse(capture.image).subject
    settings = GenAISettings()
    maps = controlmaps.build(capture.image, observation, settings)
    shape = (settings.control_map_height, settings.control_map_width)
    assert maps.pose.shape[:2] == shape
    assert maps.silhouette.shape == shape
    assert maps.normalised_crop.shape[:2] == shape
    assert maps.frame.height / maps.frame.width == pytest.approx(
        settings.control_map_height / settings.control_map_width, rel=1e-6
    )


def test_the_skeleton_is_drawn_and_the_silhouette_is_populated(capture, perception) -> None:
    observation = perception.analyse(capture.image).subject
    maps = controlmaps.build(capture.image, observation, GenAISettings())
    assert (maps.pose.sum(axis=2) > 0).mean() > 0.005
    assert 0.05 < (maps.silhouette > 127).mean() < 0.6


def test_the_skeleton_lands_on_the_silhouette(capture, perception) -> None:
    """Alignment is the point: the maps are stacked as ControlNet inputs."""
    observation = perception.analyse(capture.image).subject
    maps = controlmaps.build(capture.image, observation, GenAISettings())
    drawn = maps.pose.sum(axis=2) > 0
    body = maps.silhouette > 127
    from scipy import ndimage

    # Allow a few pixels of slack: joint discs and limb strokes have width, and
    # a wrist marker legitimately sits on the silhouette edge.
    near_body = ndimage.binary_dilation(body, iterations=8)
    assert (drawn & near_body).sum() / max(drawn.sum(), 1) > 0.85


def test_a_missing_mask_yields_an_empty_silhouette(capture, perception) -> None:
    observation = perception.analyse(capture.image).subject
    observation.mask = None
    maps = controlmaps.build(capture.image, observation, GenAISettings())
    assert maps.silhouette.max() == 0
    assert np.asarray(maps.pose).any(), "pose must still render without a mask"


def test_maps_encode_to_png(capture, perception) -> None:
    observation = perception.analyse(capture.image).subject
    encoded = controlmaps.build(capture.image, observation, GenAISettings()).as_png_base64()
    assert set(encoded) == {"pose", "silhouette", "normalised_crop"}
    assert all(len(value) > 100 for value in encoded.values())
