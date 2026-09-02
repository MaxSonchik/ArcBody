"""Deciding which way the subject is facing.

The view is not cosmetic: a girth reconstructed from a frontal breadth and one
reconstructed from a profile depth are different measurements, and averaging
them without knowing which is which produces a number that describes nobody.
"""

from __future__ import annotations

import numpy as np

from arcbody import keypoints as kp
from arcbody.types import ViewLabel


def classify_view(keypoints: np.ndarray, min_score: float) -> tuple[ViewLabel, float]:
    """Infer the view from face and shoulder geometry.

    Two signals carry it. Shoulder separation relative to torso length collapses
    in profile, because one shoulder hides behind the other. And the face tells
    front from back: a visible nose means the subject is looking at the camera,
    while two ears and no nose means the back of a head.

    Returns the label and a confidence in ``[0, 1]``.
    """
    scores = keypoints[:, 2]
    nose_visible = scores[kp.NOSE] >= min_score
    eyes_visible = int(scores[kp.LEFT_EYE] >= min_score) + int(scores[kp.RIGHT_EYE] >= min_score)
    ears_visible = int(scores[kp.LEFT_EAR] >= min_score) + int(scores[kp.RIGHT_EAR] >= min_score)

    shoulders = (
        scores[kp.LEFT_SHOULDER] >= min_score and scores[kp.RIGHT_SHOULDER] >= min_score
    )
    hips = scores[kp.LEFT_HIP] >= min_score and scores[kp.RIGHT_HIP] >= min_score
    if not (shoulders and hips):
        return ViewLabel.UNKNOWN, 0.0

    shoulder_span = float(
        abs(keypoints[kp.LEFT_SHOULDER, 0] - keypoints[kp.RIGHT_SHOULDER, 0])
    )
    shoulder_mid = (keypoints[kp.LEFT_SHOULDER, :2] + keypoints[kp.RIGHT_SHOULDER, :2]) / 2.0
    hip_mid = (keypoints[kp.LEFT_HIP, :2] + keypoints[kp.RIGHT_HIP, :2]) / 2.0
    torso_length = float(np.linalg.norm(shoulder_mid - hip_mid))
    if torso_length <= 1e-6:
        return ViewLabel.UNKNOWN, 0.0

    # A fronto-parallel adult torso is roughly as wide across the shoulders as
    # it is long; in profile that ratio falls by more than half.
    openness = shoulder_span / torso_length

    if openness < 0.42:
        confidence = float(np.clip((0.42 - openness) / 0.30, 0.3, 1.0))
        return ViewLabel.SIDE, confidence

    confidence = float(np.clip((openness - 0.42) / 0.35, 0.3, 1.0))
    if nose_visible or eyes_visible >= 1:
        return ViewLabel.FRONT, confidence
    if ears_visible >= 1:
        return ViewLabel.BACK, confidence * 0.8
    # No face at all: a cropped head or a distant subject. Front is the safer
    # default, but the low confidence tells the estimator to widen its interval.
    return ViewLabel.FRONT, confidence * 0.4
