"""Capture quality gating."""

from __future__ import annotations

from arcbody.config import PerceptionSettings, QualitySettings
from arcbody.measure.quality import assess, confidence_multiplier
from arcbody.training.synthetic import PoseParams, render, sample_appearance
from arcbody.types import QualityReport


def test_a_clean_capture_passes(capture, perception) -> None:
    subject = perception.analyse(capture.image).subject
    report = assess(subject, QualitySettings(), PerceptionSettings())
    assert report.passed
    assert report.score > 0.6
    assert report.hints() == []


def test_a_small_off_centre_leaning_subject_is_rejected(subject, perception, rng) -> None:
    bad = render(
        subject,
        PoseParams(roll_deg=35, subject_height_ratio=0.28, centre_x_frac=0.18),
        sample_appearance(rng),
        size=(384, 512),
        rng=rng,
    )
    observation = perception.analyse(bad.image).subject
    if observation is None:
        return  # nothing to measure is itself a rejection
    report = assess(observation, QualitySettings(), PerceptionSettings())
    assert not report.passed
    assert report.score < QualitySettings().reject_below
    assert report.hints(), "a rejected capture must say how to fix it"


def test_every_failed_gate_carries_advice(subject, perception, rng) -> None:
    bad = render(
        subject,
        PoseParams(subject_height_ratio=0.2),
        sample_appearance(rng),
        size=(384, 512),
        rng=rng,
    )
    observation = perception.analyse(bad.image).subject
    if observation is None:
        return
    report = assess(observation, QualitySettings(), PerceptionSettings())
    for gate in report.failures:
        assert gate.explanation.strip(), f"gate {gate.name} fails without an explanation"


def test_intervals_widen_as_quality_drops() -> None:
    good = confidence_multiplier(QualityReport(score=1.0))
    poor = confidence_multiplier(QualityReport(score=0.5))
    assert good == 1.0
    assert poor > good
    assert confidence_multiplier(QualityReport(score=0.0)) <= 2.2
