"""The classic silhouette backend."""

from __future__ import annotations

import numpy as np
import pytest

from arcbody import keypoints as kp
from arcbody.config import PerceptionSettings
from arcbody.perception.base import largest_component, rank_observations
from arcbody.perception.classic import otsu_threshold, segment_subject
from arcbody.perception.registry import build_backend
from arcbody.perception.view import classify_view
from arcbody.types import BoundingBox, PersonObservation, ViewLabel


def test_segmentation_recovers_the_body(capture, perception) -> None:
    mask = segment_subject(capture.image)
    intersection = (mask & capture.mask).sum()
    union = (mask | capture.mask).sum()
    assert intersection / union > 0.90


def test_segmentation_keeps_skin_as_well_as_clothing(capture) -> None:
    """Regression: an Otsu-only threshold split skin onto the background side.

    That amputated the head and both arms while still looking like a plausible
    silhouette, so it was invisible until keypoints were checked.
    """
    mask = segment_subject(capture.image)
    head_band = slice(0, int(0.2 * mask.shape[0]))
    assert mask[head_band].any(), "the head was segmented away"


def test_blank_image_yields_no_person(perception) -> None:
    blank = np.full((400, 300, 3), 210, np.uint8)
    assert perception.analyse(blank).observations == []


def test_landmarks_land_close_to_the_truth(capture, perception) -> None:
    subject = perception.analyse(capture.image).subject
    assert subject is not None
    stature = capture.stature_px
    for name in ("left_shoulder", "left_hip", "left_knee", "left_ankle"):
        index = kp.INDEX[name]
        error = np.linalg.norm(capture.keypoints[index, :2] - subject.keypoints[index, :2])
        assert error / stature < 0.06, f"{name} off by {error / stature:.1%} of stature"


def test_arm_tracing_stops_at_the_wrist(capture, perception) -> None:
    """Regression: the trace used to run past the hand and pick up the legs."""
    subject = perception.analyse(capture.image).subject
    wrist = subject.keypoints[kp.LEFT_WRIST]
    if wrist[2] == 0:
        pytest.skip("no arm was separable in this capture")
    error = np.linalg.norm(capture.keypoints[kp.LEFT_WRIST, :2] - wrist[:2])
    assert error / capture.stature_px < 0.08


def test_derived_keypoints_are_scored_below_a_real_detection(capture, perception) -> None:
    subject = perception.analyse(capture.image).subject
    observed = subject.keypoints[subject.keypoints[:, 2] > 0]
    assert observed[:, 2].max() < 1.0


def test_otsu_splits_a_bimodal_distribution() -> None:
    values = np.concatenate([np.zeros(500), np.full(500, 100.0)])
    assert 10 < otsu_threshold(values) < 90


def test_largest_component_fills_holes() -> None:
    mask = np.zeros((50, 50), bool)
    mask[10:40, 10:40] = True
    mask[20:25, 20:25] = False
    mask[0:3, 0:3] = True
    assert largest_component(mask).sum() == 30 * 30


def test_ranking_prefers_the_large_central_subject() -> None:
    small = PersonObservation(BoundingBox(0, 0, 10, 20), kp.empty(), (200, 200), 0.9)
    large = PersonObservation(BoundingBox(70, 10, 130, 190), kp.empty(), (200, 200), 0.9)
    assert rank_observations([small, large], (200, 200))[0] is large


def test_view_classification() -> None:
    points = kp.empty()
    for index, (x, y) in {
        kp.LEFT_SHOULDER: (40, 100),
        kp.RIGHT_SHOULDER: (90, 100),
        kp.LEFT_HIP: (50, 180),
        kp.RIGHT_HIP: (80, 180),
        kp.NOSE: (65, 70),
    }.items():
        points[index] = (x, y, 0.9)
    assert classify_view(points, 0.3)[0] is ViewLabel.FRONT

    profile = points.copy()
    profile[kp.RIGHT_SHOULDER] = (48, 100, 0.9)
    assert classify_view(profile, 0.3)[0] is ViewLabel.SIDE

    behind = points.copy()
    behind[kp.NOSE, 2] = 0.0
    behind[kp.LEFT_EAR] = (55, 70, 0.8)
    behind[kp.RIGHT_EAR] = (75, 70, 0.8)
    assert classify_view(behind, 0.3)[0] is ViewLabel.BACK

    assert classify_view(kp.empty(), 0.3)[0] is ViewLabel.UNKNOWN


def test_registry_honours_an_explicit_choice() -> None:
    assert build_backend(PerceptionSettings(backend="classic")).name == "classic"
