"""The COCO-17 keypoint schema, shared by every perception backend.

Backends differ wildly in what they emit; they all normalise to this array so
that measurement, control-map rendering and quality gating have one contract:

    keypoints: float32 array of shape (17, 3) -> (x_px, y_px, score)

A score of 0 means "not observed". Coordinates of unobserved points are
meaningless and must never be read without checking the score.
"""

from __future__ import annotations

import numpy as np

NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

NUM_KEYPOINTS = len(NAMES)
INDEX: dict[str, int] = {name: i for i, name in enumerate(NAMES)}

NOSE = INDEX["nose"]
LEFT_EYE = INDEX["left_eye"]
RIGHT_EYE = INDEX["right_eye"]
LEFT_EAR = INDEX["left_ear"]
RIGHT_EAR = INDEX["right_ear"]
LEFT_SHOULDER = INDEX["left_shoulder"]
RIGHT_SHOULDER = INDEX["right_shoulder"]
LEFT_ELBOW = INDEX["left_elbow"]
RIGHT_ELBOW = INDEX["right_elbow"]
LEFT_WRIST = INDEX["left_wrist"]
RIGHT_WRIST = INDEX["right_wrist"]
LEFT_HIP = INDEX["left_hip"]
RIGHT_HIP = INDEX["right_hip"]
LEFT_KNEE = INDEX["left_knee"]
RIGHT_KNEE = INDEX["right_knee"]
LEFT_ANKLE = INDEX["left_ankle"]
RIGHT_ANKLE = INDEX["right_ankle"]

#: Limb connectivity, used for control-map rendering and tilt estimation.
SKELETON: tuple[tuple[int, int], ...] = (
    (LEFT_ANKLE, LEFT_KNEE),
    (LEFT_KNEE, LEFT_HIP),
    (RIGHT_ANKLE, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_EYE, RIGHT_EYE),
    (NOSE, LEFT_EYE),
    (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR),
    (RIGHT_EYE, RIGHT_EAR),
)

#: Per-limb colours in the OpenPose convention, which ControlNet pose adapters
#: were trained against. Rendering with arbitrary colours degrades conditioning.
SKELETON_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (255, 0, 0),
    (85, 255, 0),
    (170, 255, 0),
    (255, 170, 0),
    (255, 255, 0),
    (255, 0, 170),
    (255, 0, 85),
    (255, 0, 255),
    (170, 0, 255),
    (85, 0, 255),
)

#: Keypoints that must be observed for anthropometry to be attempted at all.
#: Without shoulders, hips and ankles there is no torso frame and no stature.
REQUIRED_FOR_MEASUREMENT: tuple[int, ...] = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)


def empty() -> np.ndarray:
    """An all-unobserved keypoint array."""
    return np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)


def validate(keypoints: np.ndarray) -> np.ndarray:
    """Coerce to the canonical ``(17, 3) float32`` layout, or raise."""
    array = np.asarray(keypoints, dtype=np.float32)
    if array.shape != (NUM_KEYPOINTS, 3):
        raise ValueError(f"keypoints must have shape ({NUM_KEYPOINTS}, 3), got {array.shape}")
    return array


def observed(keypoints: np.ndarray, min_score: float) -> np.ndarray:
    """Boolean mask of keypoints whose score clears ``min_score``."""
    return keypoints[:, 2] >= min_score


def midpoint(keypoints: np.ndarray, left: int, right: int, min_score: float) -> np.ndarray | None:
    """Midpoint of a bilateral pair, or ``None`` unless both sides are observed.

    Falling back to a single side would silently bias every downstream width,
    so callers are made to handle the missing case explicitly.
    """
    if keypoints[left, 2] < min_score or keypoints[right, 2] < min_score:
        return None
    return (keypoints[left, :2] + keypoints[right, :2]) / 2.0
