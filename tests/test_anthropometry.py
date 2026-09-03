"""End-to-end measurement accuracy against exact ground truth.

The synthetic generator defines a body by its ratios, so these are not
comparisons against another estimate — they are comparisons against the number
the body was built from. The thresholds are regression guards set a little above
the pipeline's current error, not accuracy claims about photographs of people.
"""

from __future__ import annotations

import numpy as np
import pytest

from arcbody.config import MeasurementSettings, PerceptionSettings, QualitySettings
from arcbody.measure import anthropometry as anth
from arcbody.measure.quality import assess
from arcbody.measure.schema import (
    DEPTH_OVER_BREADTH_PRIOR,
    girth_from_breadth_depth,
)
from arcbody.training.synthetic import sample_params
from arcbody.types import MeasurementSource, QualityReport, ViewLabel

#: Median relative error each measurement must stay under, across the cohort.
TOLERANCE = {
    "stature": 0.001,
    "waist_breadth": 0.06,
    "hip_breadth": 0.08,
    "chest_breadth": 0.12,
    "neck_breadth": 0.06,
    "shoulder_breadth": 0.10,
    "waist_girth": 0.08,
    "hip_girth": 0.09,
    "neck_girth": 0.06,
    "leg_length": 0.06,
    "inseam": 0.09,
    "knee_height": 0.06,
}

COHORT = 12


def _measure(params, shoot, perception, views=(ViewLabel.FRONT,)):
    observations = []
    for view in views:
        capture = shoot(params, view=view)
        observation = perception.analyse(capture.image).subject
        if observation is None:
            return None
        observation.view = view
        observations.append(observation)
    report = assess(observations[0], QualitySettings(), PerceptionSettings())
    return anth.estimate(observations, report, stature_cm=params.stature_cm)


@pytest.fixture(scope="module")
def cohort_errors(request):
    """Relative errors per measurement across a cohort, computed once."""
    from arcbody.perception.classic import ClassicPerception

    perception = ClassicPerception(PerceptionSettings())
    rng = np.random.default_rng(7)

    def shoot(params, view=ViewLabel.FRONT):
        from arcbody.training.synthetic import render, sample_appearance, sample_pose

        return render(
            params,
            sample_pose(rng, easy=True),
            sample_appearance(rng),
            size=(384, 512),
            view=view,
            rng=rng,
        )

    errors: dict[str, list[float]] = {}
    for _ in range(COHORT):
        params = sample_params(rng)
        measurements = _measure(params, shoot, perception)
        if measurements is None:
            continue
        truth = params.measurements_cm()
        for name, value in measurements.values.items():
            if name in truth and truth[name] > 0:
                errors.setdefault(name, []).append(
                    abs(value.value_cm - truth[name]) / truth[name]
                )
    return errors


@pytest.mark.parametrize("name", sorted(TOLERANCE))
def test_measurement_accuracy(cohort_errors, name) -> None:
    values = cohort_errors.get(name)
    assert values, f"{name} was never produced across the cohort"
    median = float(np.median(values))
    assert median <= TOLERANCE[name], f"{name} median error {median:.1%}"


def test_stature_scales_the_output_without_touching_the_shape(
    subject, shoot, perception
) -> None:
    """The load-bearing property of the whole design.

    A photograph determines proportions; stature only rescales them afterwards.
    So the *same* photo analysed with two different statures must yield
    bit-identical ratios and centimetre values in exact proportion. Anything
    else would mean the supplied height had leaked into the shape estimate.
    """
    capture = shoot(subject)
    observation = perception.analyse(capture.image).subject
    report = assess(observation, QualitySettings(), PerceptionSettings())

    short = anth.estimate([observation], report, stature_cm=155.0)
    tall = anth.estimate([observation], report, stature_cm=195.0)

    assert short.ratios == tall.ratios
    factor = 195.0 / 155.0
    for name, value in short.values.items():
        assert tall.values[name].value_cm == pytest.approx(
            value.value_cm * factor, rel=1e-3
        ), name


def test_proportions_are_stable_across_captures(subject, shoot, perception) -> None:
    """Two different photos of one body must agree on its proportions.

    Looser than the scaling test above, and deliberately so: this one includes
    pose, framing and segmentation noise. Compound ratios such as waist-to-hip
    divide two noisy estimates and are checked separately at a wider tolerance,
    because their errors multiply rather than average out.
    """
    first = _measure(subject, shoot, perception)
    second = _measure(subject, shoot, perception)
    assert first is not None and second is not None

    direct = [name for name in first.ratios if name.endswith("_to_stature")]
    assert direct, "no stature-relative ratios were produced"
    for name in direct:
        other = second.ratios.get(name)
        if other is None:
            continue
        assert abs(first.ratios[name] - other) / first.ratios[name] < 0.15, name

    for name in ("waist_to_hip", "shoulder_to_waist"):
        if name in first.ratios and name in second.ratios:
            drift = abs(first.ratios[name] - second.ratios[name]) / first.ratios[name]
            assert drift < 0.30, f"{name} drifted {drift:.0%} between captures"


def test_no_stature_means_no_centimetres(subject, shoot, perception) -> None:
    capture = shoot(subject)
    observation = perception.analyse(capture.image).subject
    report = assess(observation, QualitySettings(), PerceptionSettings())
    measurements = anth.estimate([observation], report, stature_cm=None)
    assert measurements.ratios, "proportions must still be reported"
    assert measurements.values == {}
    assert any("no stature" in note for note in measurements.notes)


def test_a_side_view_replaces_the_depth_prior(subject, shoot, perception) -> None:
    single = _measure(subject, shoot, perception)
    both = _measure(subject, shoot, perception, views=(ViewLabel.FRONT, ViewLabel.SIDE))
    assert single is not None and both is not None
    assert single.values["waist_girth"].source is MeasurementSource.ELLIPSE_PRIOR
    assert both.values["waist_girth"].source is MeasurementSource.SILHOUETTE


def test_absurd_statures_are_refused() -> None:
    settings = MeasurementSettings()
    assert anth.validate_stature(None, settings) is None
    assert anth.validate_stature(175.0, settings) == 175.0
    for value in (10.0, 400.0):
        with pytest.raises(anth.ScaleError):
            anth.validate_stature(value, settings)


def test_implausible_ratios_are_flagged_not_returned_silently() -> None:
    assert anth.implausible_ratios({"waist_to_hip": 0.8}) == []
    assert "waist_to_hip" in anth.implausible_ratios({"waist_to_hip": 4.0})


def test_a_circular_section_has_a_circular_perimeter() -> None:
    assert girth_from_breadth_depth(10.0, 10.0) == pytest.approx(np.pi * 10.0, rel=1e-6)


def test_every_girth_has_a_documented_depth_prior() -> None:
    from arcbody.measure.schema import GIRTHS

    assert set(GIRTHS) == set(DEPTH_OVER_BREADTH_PRIOR)
    assert all(0.5 < value <= 1.0 for value in DEPTH_OVER_BREADTH_PRIOR.values())


def test_intervals_bracket_the_value_and_widen_with_poor_quality(
    subject, shoot, perception
) -> None:
    capture = shoot(subject)
    observation = perception.analyse(capture.image).subject
    good = anth.estimate(
        [observation], QualityReport(score=1.0), stature_cm=subject.stature_cm
    )
    poor = anth.estimate(
        [observation], QualityReport(score=0.5), stature_cm=subject.stature_cm
    )
    for name, value in good.values.items():
        assert value.ci_low_cm <= value.value_cm <= value.ci_high_cm
        if value.source is not MeasurementSource.CLIENT:
            assert poor.values[name].relative_ci > value.relative_ci, name
