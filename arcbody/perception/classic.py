"""A silhouette-first perception backend with no learned weights.

This is the backend that pre-deep-learning photogrammetry used, and it still
earns its place for three reasons: it runs anywhere, with no download and no
GPU; it is fully deterministic, so measurement regressions are attributable to
the estimator rather than to a detector's nondeterminism; and it degrades
loudly rather than quietly, refusing images it cannot segment instead of
inventing a person.

Its assumption is explicit and matches the capture instructions the service
gives: **one subject, standing, against a plain background that differs from
their clothing**. Against a cluttered scene it will fail the quality gates,
which is the correct outcome — use the YOLO backend there.

Joints are *derived*, not detected: anatomical proportions place a search band,
and the silhouette's own structure locates the landmark inside it. Every derived
point is scored below a real detection so that downstream code, and the quality
report, can tell the difference.
"""

from __future__ import annotations

import numpy as np

from arcbody import keypoints as kp
from arcbody.config import PerceptionSettings
from arcbody.measure.schema import LANDMARK_LEVELS
from arcbody.measure.silhouette import SilhouetteProfile
from arcbody.perception.base import PerceptionResult, largest_component, rank_observations
from arcbody.perception.view import classify_view
from arcbody.types import BoundingBox, PersonObservation

#: Confidence attached to each family of derived landmark. Torso joints sit on
#: strong silhouette structure; limb joints are interpolated along an arm and
#: deserve to be trusted less.
SCORE_TORSO = 0.75
SCORE_LEG = 0.70
SCORE_HEAD = 0.50
SCORE_ARM = 0.45

#: Hand length as a fraction of stature (Drillis & Contini), used to step back
#: from the traced fingertip to the anatomical wrist.
HAND_LENGTH = 0.108

#: A mask covering less of the frame than this is noise; more than this is a
#: failed background estimate rather than a very close subject.
MIN_MASK_FRACTION = 0.012
MAX_MASK_FRACTION = 0.85


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold: the split maximising between-class variance."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    histogram, edges = np.histogram(finite, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    total = histogram.sum()
    if total == 0:
        return float(centres[0])

    weight_low = np.cumsum(histogram)
    weight_high = total - weight_low
    valid = (weight_low > 0) & (weight_high > 0)
    if not valid.any():
        return float(centres[0])

    cumulative = np.cumsum(histogram * centres)
    mean_low = np.divide(
        cumulative, weight_low, out=np.zeros_like(cumulative), where=weight_low > 0
    )
    mean_high = np.divide(
        cumulative[-1] - cumulative,
        weight_high,
        out=np.zeros_like(cumulative),
        where=weight_high > 0,
    )
    variance = weight_low * weight_high * (mean_low - mean_high) ** 2
    variance[~valid] = -np.inf

    # With a sharply bimodal image every split between the two clusters scores
    # identically, and taking argmax lands on the *lowest* of them — a threshold
    # sitting right on the shoulder of the background cluster, where a little
    # sensor noise flips whole regions into the foreground. The centre of the
    # tied plateau is the stable choice, and is what a split "between the two
    # peaks" is meant to mean.
    best = variance.max()
    plateau = np.flatnonzero(variance >= best - 1e-9 * max(abs(best), 1.0))
    return float(centres[int(np.median(plateau))])


def segment_subject(image: np.ndarray, border_fraction: float = 0.06) -> np.ndarray:
    """Separate the subject from a plain backdrop.

    The backdrop colour is taken as the median of a border band, which is robust
    to a subject who overlaps the frame edge as long as they do not fill it. The
    per-pixel distance from that colour is then split by Otsu — no fixed
    threshold, so a dark subject on a light wall and a light subject on a grey
    one both work.
    """
    rgb = np.asarray(image, dtype=np.float32)
    height, width = rgb.shape[:2]
    band_y = max(1, int(height * border_fraction))
    band_x = max(1, int(width * border_fraction))

    border = np.concatenate(
        [
            rgb[:band_y].reshape(-1, 3),
            rgb[-band_y:].reshape(-1, 3),
            rgb[:, :band_x].reshape(-1, 3),
            rgb[:, -band_x:].reshape(-1, 3),
        ]
    )
    background = np.median(border, axis=0)

    distance = np.linalg.norm(rgb - background, axis=2)

    # Otsu alone is wrong here. A clothed person is *multi*-modal — bare skin
    # sits much closer to a pale backdrop than a dark garment does — and a
    # two-class split happily puts skin on the background side, amputating the
    # head and arms. So the backdrop's own spread sets the floor: anything more
    # than a few robust deviations away from the wall is subject, whatever else
    # is in the frame. Otsu is kept only as a ceiling, for the case where the
    # backdrop is textured enough that its spread would swallow the subject.
    border_spread = float(
        np.median(np.abs(np.linalg.norm(border - background, axis=1)))
    )
    threshold = min(otsu_threshold(distance), max(12.0, 6.0 * border_spread))
    if threshold < 8.0:
        return np.zeros((height, width), dtype=bool)

    mask = distance > threshold
    from scipy import ndimage

    structure = np.ones((3, 3), dtype=bool)
    mask = ndimage.binary_opening(mask, structure=structure, iterations=2)
    mask = ndimage.binary_closing(mask, structure=structure, iterations=2)
    return largest_component(mask)


def _outer_runs(profile: SilhouetteProfile, y: int, centre_x: float, sign: float):
    """Foreground runs on one side of the trunk at row ``y``."""
    runs = [run for run in profile.runs_at(y) if not run.contains(centre_x)]
    return [run for run in runs if np.sign(run.centre - centre_x) == sign]


def derive_keypoints(profile: SilhouetteProfile) -> np.ndarray:
    """Place COCO-17 landmarks using silhouette structure plus proportions.

    Levels come from :data:`LANDMARK_LEVELS`, but a level only selects a *band*;
    the actual row is the local extremum of trunk width inside it, and the
    horizontal position comes from the pixels there. The result is a body's own
    shoulders, not the average body's shoulders drawn on top of it.
    """
    points = kp.empty()
    centre_x = profile.centre_x()

    # Head. Its breadth at the level of the eyes anchors the face points.
    head_row = profile.row_of(0.060)
    head_half = max(2.0, profile.span_width(head_row) / 2.0)
    points[kp.NOSE] = (centre_x, profile.row_of(0.075), SCORE_HEAD)
    points[kp.LEFT_EYE] = (centre_x - head_half * 0.40, profile.row_of(0.055), SCORE_HEAD)
    points[kp.RIGHT_EYE] = (centre_x + head_half * 0.40, profile.row_of(0.055), SCORE_HEAD)
    points[kp.LEFT_EAR] = (centre_x - head_half * 0.82, profile.row_of(0.068), SCORE_HEAD * 0.8)
    points[kp.RIGHT_EAR] = (centre_x + head_half * 0.82, profile.row_of(0.068), SCORE_HEAD * 0.8)

    # Shoulders. The acromion is where the silhouette *abruptly widens* below
    # the neck, not where it is widest: with the arms held out, trunk width
    # keeps growing for some rows below the shoulder, so a maximum search walks
    # down the arm and lands well below the joint.
    shoulder_row, shoulder_width = profile.steepest_widening(
        LANDMARK_LEVELS["neck"] - 0.020,
        LANDMARK_LEVELS["shoulder"] + 0.030,
        centre_x,
    )
    # Joint centres sit inboard of the deltoid edge: biacromial breadth is about
    # 78% of the bideltoid breadth the silhouette shows, hence half of that.
    shoulder_dx = shoulder_width * 0.39
    points[kp.LEFT_SHOULDER] = (centre_x - shoulder_dx, shoulder_row, SCORE_TORSO)
    points[kp.RIGHT_SHOULDER] = (centre_x + shoulder_dx, shoulder_row, SCORE_TORSO)

    # Hips: widest trunk row over the pelvis.
    hip_row, hip_width = profile.extreme_in_band(
        LANDMARK_LEVELS["hip"] - 0.045,
        LANDMARK_LEVELS["hip"] + 0.045,
        centre_x,
        mode="max",
    )
    hip_dx = hip_width * 0.26
    points[kp.LEFT_HIP] = (centre_x - hip_dx, hip_row, SCORE_TORSO)
    points[kp.RIGHT_HIP] = (centre_x + hip_dx, hip_row, SCORE_TORSO)

    # Knees and ankles: taken from the separated leg runs where they exist.
    for level, left_index, right_index in (
        (LANDMARK_LEVELS["knee"], kp.LEFT_KNEE, kp.RIGHT_KNEE),
        (LANDMARK_LEVELS["ankle"], kp.LEFT_ANKLE, kp.RIGHT_ANKLE),
    ):
        row = profile.row_of(level)
        runs = profile.runs_at(row)
        if len(runs) >= 2:
            points[left_index] = (runs[0].centre, row, SCORE_LEG)
            points[right_index] = (runs[-1].centre, row, SCORE_LEG)
        elif runs:
            # Legs together: one run, so split it down the middle and say so.
            run = max(runs, key=lambda item: item.width)
            points[left_index] = (run.centre - run.width * 0.25, row, SCORE_LEG * 0.6)
            points[right_index] = (run.centre + run.width * 0.25, row, SCORE_LEG * 0.6)

    # Arms: follow the outermost run on each side downwards from the shoulder.
    for sign, elbow_index, wrist_index in (
        (-1.0, kp.LEFT_ELBOW, kp.LEFT_WRIST),
        (1.0, kp.RIGHT_ELBOW, kp.RIGHT_WRIST),
    ):
        # Walk down the outermost run on this side. The arm is fused with the
        # trunk for the first rows below the shoulder, separates, and then ends
        # at the fingertips — and the *first gap after it separates* is the
        # hand. Running past that gap picks the legs up as if they were arms,
        # which is why the scan stops rather than continuing to a fixed depth.
        path: list[tuple[float, float]] = []
        for row in range(shoulder_row, profile.row_of(0.72)):
            runs = _outer_runs(profile, row, centre_x, sign)
            if runs:
                outermost = max(runs, key=lambda run: abs(run.centre - centre_x))
                path.append((outermost.centre, float(row)))
            elif path:
                break
        if len(path) < 8:
            continue

        # The traced run ends at the fingertips. Anatomical arm length stops at
        # the wrist, so step back along the arm's own direction by a hand
        # length — otherwise every arm measures a hand too long.
        tip_x, tip_y = path[-1]
        origin_x, origin_y = path[max(0, len(path) - 2 - int(0.10 * profile.stature_px))]
        direction = np.array([tip_x - origin_x, tip_y - origin_y], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction /= norm
            wrist_x, wrist_y = (
                np.array([tip_x, tip_y]) - direction * HAND_LENGTH * profile.stature_px
            )
        else:
            wrist_x, wrist_y = tip_x, tip_y
        shoulder_y = float(points[kp.LEFT_SHOULDER if sign < 0 else kp.RIGHT_SHOULDER, 1])
        # The elbow divides the arm at the upper-arm share of its length. Taking
        # the path point nearest that height follows a bent arm instead of
        # assuming the limb is straight.
        target_y = shoulder_y + 0.56 * (wrist_y - shoulder_y)
        elbow_x, elbow_y = min(path, key=lambda point: abs(point[1] - target_y))
        points[elbow_index] = (elbow_x, elbow_y, SCORE_ARM)
        points[wrist_index] = (wrist_x, wrist_y, SCORE_ARM * 0.9)

    return points


class ClassicPerception:
    """Silhouette segmentation plus proportion-guided landmarking."""

    name = "classic"

    def __init__(self, settings: PerceptionSettings) -> None:
        self.settings = settings

    def available(self) -> bool:
        """Always: numpy and scipy are core dependencies."""
        return True

    def analyse(self, image: np.ndarray) -> PerceptionResult:
        height, width = image.shape[:2]
        mask = segment_subject(image)
        fraction = mask.mean() if mask.size else 0.0
        if not (MIN_MASK_FRACTION <= fraction <= MAX_MASK_FRACTION):
            return PerceptionResult(observations=[], backend=self.name)

        try:
            profile = SilhouetteProfile(mask)
        except ValueError:
            return PerceptionResult(observations=[], backend=self.name)

        points = derive_keypoints(profile)
        columns = np.flatnonzero(mask.any(axis=0))
        bbox = BoundingBox(
            x1=float(columns[0]),
            y1=float(profile.top),
            x2=float(columns[-1] + 1),
            y2=float(profile.bottom + 1),
        )
        view, view_confidence = classify_view(points, self.settings.min_keypoint_score)
        observation = PersonObservation(
            bbox=bbox,
            keypoints=points,
            image_size=(width, height),
            # Segmentation is all-or-nothing here, so the "detection score"
            # reports how plausibly human the silhouette's fill ratio is rather
            # than a classifier's confidence.
            detection_score=float(np.clip(profile.fill_ratio() * 2.2, 0.2, 0.95)),
            mask=mask,
            view=view,
            backend=self.name,
        )
        observations = rank_observations([observation], (width, height))
        result = PerceptionResult(observations=observations, backend=self.name)
        observation_meta = {"view_confidence": view_confidence}
        result.observations[0].__dict__.setdefault("meta", observation_meta)
        return result
