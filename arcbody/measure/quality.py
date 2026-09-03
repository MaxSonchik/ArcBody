"""Deciding whether a photo can carry a trustworthy measurement.

Pixel-to-centimetre scaling rests on assumptions — the subject is upright, whole,
large in frame, and not badly foreshortened — and when they break the numbers do
not get noisier, they get *wrong* in a way no confidence interval discloses. So
each assumption is a named gate with a threshold and a human-readable remedy,
and the report travels with the profile all the way to the API response.
"""

from __future__ import annotations

import math

import numpy as np

from arcbody import keypoints as kp
from arcbody.config import PerceptionSettings, QualitySettings
from arcbody.types import GateResult, PersonObservation, QualityReport, ViewLabel


def _torso_tilt_degrees(points: np.ndarray, min_score: float) -> float | None:
    """Angle of the shoulder-to-hip axis away from vertical."""
    shoulder = kp.midpoint(points, kp.LEFT_SHOULDER, kp.RIGHT_SHOULDER, min_score)
    hip = kp.midpoint(points, kp.LEFT_HIP, kp.RIGHT_HIP, min_score)
    if shoulder is None or hip is None:
        return None
    dx = float(hip[0] - shoulder[0])
    dy = float(hip[1] - shoulder[1])
    if abs(dy) < 1e-6:
        return 90.0
    return abs(math.degrees(math.atan2(dx, dy)))


def assess(
    observation: PersonObservation,
    quality: QualitySettings,
    perception: PerceptionSettings,
) -> QualityReport:
    """Run every gate and combine them into one score.

    The score is the mean of per-gate margins rather than a pass/fail count, so
    a photo that clears everything comfortably scores above one that scrapes
    through — and the estimator widens its intervals accordingly instead of
    reporting the same precision for both.
    """
    width, height = observation.image_size
    points = observation.keypoints
    min_score = perception.min_keypoint_score
    gates: list[GateResult] = []

    def gate(name: str, value: float, threshold: float, passed: bool, explanation: str) -> None:
        gates.append(GateResult(name, float(value), float(threshold), passed, explanation))

    subject_ratio = observation.bbox.height / max(1.0, height)
    gate(
        "subject_size",
        subject_ratio,
        quality.min_subject_height_ratio,
        subject_ratio >= quality.min_subject_height_ratio,
        "The subject is too small in frame. Fill most of the height of the photo.",
    )

    observed = observation.observed(min_score)
    coverage = float(observed.mean())
    gate(
        "keypoint_coverage",
        coverage,
        quality.min_keypoint_coverage,
        coverage >= quality.min_keypoint_coverage,
        "Parts of the body are hidden or out of frame. Show the whole body, head to feet.",
    )

    required = all(bool(observed[index]) for index in kp.REQUIRED_FOR_MEASUREMENT)
    gate(
        "landmarks_present",
        1.0 if required else 0.0,
        1.0,
        required,
        "Shoulders, hips and ankles must all be visible for measurements to be possible.",
    )

    tilt = _torso_tilt_degrees(points, min_score)
    if tilt is None:
        gate("torso_upright", 90.0, quality.max_torso_tilt_deg, False,
             "The torso could not be located. Stand facing the camera.")
    else:
        gate(
            "torso_upright",
            tilt,
            quality.max_torso_tilt_deg,
            tilt <= quality.max_torso_tilt_deg,
            "The body is leaning. Stand upright, square to the camera.",
        )

    centre_offset = abs(observation.bbox.centre[0] - width / 2.0) / max(
        1.0, observation.bbox.width
    )
    gate(
        "framing",
        centre_offset,
        quality.max_off_centre,
        centre_offset <= quality.max_off_centre,
        "The subject is far from the centre of the frame, which distorts widths. Re-centre.",
    )

    gate(
        "silhouette",
        1.0 if observation.has_mask else 0.0,
        1.0,
        observation.has_mask,
        "No silhouette could be segmented, so girths cannot be estimated. "
        "Use a plain background that contrasts with the clothing.",
    )

    gate(
        "view_known",
        1.0 if observation.view is not ViewLabel.UNKNOWN else 0.0,
        1.0,
        observation.view is not ViewLabel.UNKNOWN,
        "The facing direction is unclear. Stand square to the camera, front or side on.",
    )

    return QualityReport(score=_score(gates, quality), gates=gates)


def _score(gates: list[GateResult], quality: QualitySettings) -> float:
    """Mean per-gate margin, in ``[0, 1]``.

    Gates whose value should be *large* and gates whose value should be *small*
    are normalised into the same "headroom" scale so neither kind dominates.
    """
    lower_is_better = {"torso_upright", "framing"}
    margins: list[float] = []
    for item in gates:
        if item.threshold <= 0:
            margins.append(1.0 if item.passed else 0.0)
            continue
        if item.name in lower_is_better:
            margin = 1.0 - item.value / item.threshold
        else:
            margin = item.value / item.threshold - 1.0
        # A gate can only bank so much credit; comfortably passing four gates
        # must not buy a pass on a fifth that failed.
        margins.append(float(np.clip(0.6 + margin, 0.0, 1.0)))
    score = float(np.mean(margins)) if margins else 0.0
    if any(not item.passed for item in gates):
        score = min(score, quality.reject_below * 0.98)
    return round(score, 4)


def confidence_multiplier(report: QualityReport) -> float:
    """How much to widen measurement intervals given the capture quality.

    A perfect capture leaves the interval alone; a marginal one roughly doubles
    it. The floor stops a bad-but-passing photo from being reported as if the
    estimator had no idea, which would be its own kind of dishonesty.
    """
    return float(np.clip(2.0 - report.score, 1.0, 2.2))
