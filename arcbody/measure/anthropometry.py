"""Turning one or more observations of a body into measurements.

The estimator runs in this order, and the order is the design:

1. **Scale.** Stature in pixels comes from the silhouette's crown-to-sole
   extent. Divided into the client's stature in centimetres, it gives one scale
   factor for the whole body.
2. **Ratios.** Every landmark is located inside a proportion-guided search band
   by the silhouette's own local extremum, and every breadth and length is
   expressed as a fraction of stature. This is the part a photograph can
   actually determine.
3. **Girths.** Each cross-section is modelled as an ellipse. Its breadth is
   measured from the front; its depth is measured from a side photo when one was
   supplied, and otherwise filled in from the population prior in
   :mod:`arcbody.measure.schema`. The two cases get different sources and
   visibly different intervals.
4. **Centimetres, last.** Ratios multiply by stature only at the end, so an
   unknown stature costs the absolute numbers and nothing else.
"""

from __future__ import annotations

import numpy as np

from arcbody import keypoints as kp
from arcbody.config import MeasurementSettings, PerceptionSettings
from arcbody.measure.quality import confidence_multiplier
from arcbody.measure.schema import (
    DEPTH_OVER_BREADTH_PRIOR,
    GIRTH_SOURCE_BREADTH,
    LANDMARK_EXTREMUM,
    LANDMARK_LEVELS,
    LANDMARK_SEARCH_HALF_BAND,
    RATIO_BOUNDS,
    RATIO_NAMES,
    girth_from_breadth_depth,
)
from arcbody.measure.silhouette import SilhouetteProfile
from arcbody.types import (
    BodyMeasurements,
    MeasurementSource,
    MeasurementValue,
    PersonObservation,
    QualityReport,
    ViewLabel,
)

#: Girth names in the order they are reported.
_GIRTH_ORDER = tuple(GIRTH_SOURCE_BREADTH)

#: Only trunk sections take their depth from a side photo. In profile the arms
#: lie over the chest and the legs hide each other, so a "measured" limb depth is
#: a measurement of two overlapping limbs and is worse than the prior it would
#: replace.
_SIDE_VIEW_TRUNK = frozenset(
    {"neck_breadth", "chest_breadth", "waist_breadth", "hip_breadth"}
)

#: A cross-section flatter or rounder than this is not anatomy, it is occlusion.
#: A measured depth outside the window is discarded in favour of the prior.
_PLAUSIBLE_DEPTH_OVER_BREADTH = (0.45, 1.25)


class ScaleError(ValueError):
    """The client-supplied stature is outside the range the service accepts."""


def validate_stature(stature_cm: float | None, settings: MeasurementSettings) -> float | None:
    """Reject implausible statures rather than scaling every number by them."""
    if stature_cm is None:
        return None
    if not (settings.min_stature_cm <= stature_cm <= settings.max_stature_cm):
        raise ScaleError(
            f"stature {stature_cm} cm is outside the accepted range "
            f"{settings.min_stature_cm}-{settings.max_stature_cm} cm"
        )
    return float(stature_cm)


def _limb_tilt_cosine(points: np.ndarray, proximal: int, distal: int, min_score: float) -> float:
    """Cosine of a limb's angle from vertical, or 1.0 if it cannot be measured.

    A horizontal cut through a limb held at an angle is longer than the limb is
    thick, by exactly ``1 / cos(tilt)``. An A-pose arm at 40 degrees reads 30%
    too wide without this correction, which then propagates squared-ish into its
    girth. Clamped so a nearly horizontal limb cannot blow the correction up.
    """
    if points[proximal, 2] < min_score or points[distal, 2] < min_score:
        return 1.0
    delta = points[distal, :2] - points[proximal, :2]
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return 1.0
    return float(np.clip(abs(delta[1]) / length, 0.55, 1.0))


def _trunk_breadth_ratios(
    profile: SilhouetteProfile, points: np.ndarray, min_score: float
) -> dict[str, float]:
    """Breadths at the four trunk landmarks, as fractions of stature.

    Each landmark is a local extremum inside its band: the waist is the
    narrowest trunk row near the 40% level, the hips the widest near 53%. That
    is what makes the measurement belong to *this* body rather than to the
    average one whose proportions seeded the search.
    """
    centre_x = profile.centre_x()
    stature = profile.stature_px
    ratios: dict[str, float] = {}
    for landmark in ("neck", "chest", "waist", "hip"):
        half_band = LANDMARK_SEARCH_HALF_BAND[landmark]
        level = LANDMARK_LEVELS[landmark]
        _, width = profile.extreme_in_band(
            level - half_band,
            level + half_band,
            centre_x,
            mode=LANDMARK_EXTREMUM[landmark],
            # Below the shoulders, only rows where both arms are clear of the
            # body measure the trunk. Without this a "widest chest" search walks
            # up to where the arms are still fused and reports a chest roughly
            # twice the real one.
            require_isolated=landmark in {"chest", "waist", "hip"},
        )
        ratios[f"{landmark}_breadth"] = float(width / stature)

    shoulder_row, shoulder_width = profile.steepest_widening(
        LANDMARK_LEVELS["neck"] - 0.020, LANDMARK_LEVELS["shoulder"] + 0.030, centre_x
    )
    ratios["shoulder_breadth"] = float(shoulder_width / stature)

    # Thigh: measured immediately below the crotch, where the tape goes, on the
    # first rows at which the two legs are separate runs.
    crotch = profile.crotch_row()
    if crotch is not None:
        thigh_row = int(crotch + 0.012 * stature)
        runs = profile.runs_at(thigh_row)
        if len(runs) >= 2:
            tilt = min(
                _limb_tilt_cosine(points, kp.LEFT_HIP, kp.LEFT_KNEE, min_score),
                _limb_tilt_cosine(points, kp.RIGHT_HIP, kp.RIGHT_KNEE, min_score),
            )
            ratios["thigh_breadth"] = float(
                max(run.width for run in runs) * tilt / stature
            )

    # Upper arm: the outer run beside the chest, which is the arm in an A-pose.
    chest_row = profile.row_of(LANDMARK_LEVELS["chest"])
    for side, proximal, distal in (
        ("left", kp.LEFT_SHOULDER, kp.LEFT_ELBOW),
        ("right", kp.RIGHT_SHOULDER, kp.RIGHT_ELBOW),
    ):
        arm_width = profile.limb_width(chest_row, side, centre_x)
        if arm_width > 0:
            tilt = _limb_tilt_cosine(points, proximal, distal, min_score)
            ratios["upper_arm_breadth"] = float(arm_width * tilt / stature)
            break

    inseam = profile.inseam_px()
    if inseam is not None:
        ratios["inseam"] = float(inseam / stature)

    return ratios


def _skeleton_length_ratios(
    observation: PersonObservation, stature_px: float, min_score: float
) -> dict[str, float]:
    """Joint-to-joint lengths, as fractions of stature."""
    points = observation.keypoints
    ratios: dict[str, float] = {}

    shoulder = kp.midpoint(points, kp.LEFT_SHOULDER, kp.RIGHT_SHOULDER, min_score)
    hip = kp.midpoint(points, kp.LEFT_HIP, kp.RIGHT_HIP, min_score)
    if shoulder is not None and hip is not None:
        ratios["torso_length"] = float(np.linalg.norm(shoulder - hip) / stature_px)

    def limb(joints: tuple[int, ...]) -> float | None:
        total = 0.0
        for start, end in zip(joints[:-1], joints[1:], strict=True):
            if points[start, 2] < min_score or points[end, 2] < min_score:
                return None
            total += float(np.linalg.norm(points[start, :2] - points[end, :2]))
        return total / stature_px

    arms = [
        value
        for value in (
            limb((kp.LEFT_SHOULDER, kp.LEFT_ELBOW, kp.LEFT_WRIST)),
            limb((kp.RIGHT_SHOULDER, kp.RIGHT_ELBOW, kp.RIGHT_WRIST)),
        )
        if value is not None
    ]
    if arms:
        ratios["arm_length"] = float(np.mean(arms))

    legs = [
        value
        for value in (
            limb((kp.LEFT_HIP, kp.LEFT_KNEE, kp.LEFT_ANKLE)),
            limb((kp.RIGHT_HIP, kp.RIGHT_KNEE, kp.RIGHT_ANKLE)),
        )
        if value is not None
    ]
    if legs:
        ratios["leg_length"] = float(np.mean(legs))

    # Knee height is measured from the sole, not between joints, because that is
    # how the garment industry defines it and how a generator will read it.
    if points[kp.LEFT_KNEE, 2] >= min_score:
        ratios["knee_height"] = float(
            (observation.bbox.y2 - points[kp.LEFT_KNEE, 1]) / stature_px
        )
    return ratios


def _shape_vector(breadths: dict[str, float], lengths: dict[str, float]) -> dict[str, float]:
    """Assemble the canonical dimensionless descriptor from parts.

    Missing inputs propagate as missing outputs rather than as zeros: a ratio of
    0.0 would read as a real, extraordinary body instead of an absent one.
    """
    ratios: dict[str, float] = {}

    def ratio(name: str, numerator: float | None, denominator: float | None = None) -> None:
        if numerator is None or numerator <= 0:
            return
        if denominator is None:
            ratios[name] = float(numerator)
        elif denominator > 0:
            ratios[name] = float(numerator / denominator)

    ratio("shoulder_to_stature", breadths.get("shoulder_breadth"))
    ratio("chest_to_stature", breadths.get("chest_breadth"))
    ratio("waist_to_stature", breadths.get("waist_breadth"))
    ratio("hip_to_stature", breadths.get("hip_breadth"))
    ratio("thigh_to_stature", breadths.get("thigh_breadth"))
    ratio("upper_arm_to_stature", breadths.get("upper_arm_breadth"))
    ratio("neck_to_stature", breadths.get("neck_breadth"))
    ratio("torso_to_stature", lengths.get("torso_length"))
    ratio("leg_to_stature", lengths.get("leg_length"))
    ratio("arm_to_stature", lengths.get("arm_length"))
    ratio("waist_to_hip", breadths.get("waist_breadth"), breadths.get("hip_breadth"))
    ratio("shoulder_to_waist", breadths.get("shoulder_breadth"), breadths.get("waist_breadth"))
    ratio("shoulder_to_hip", breadths.get("shoulder_breadth"), breadths.get("hip_breadth"))
    ratio("chest_to_waist", breadths.get("chest_breadth"), breadths.get("waist_breadth"))
    return ratios


def implausible_ratios(ratios: dict[str, float]) -> list[str]:
    """Ratios outside the adult population range — a failed measurement."""
    out = []
    for name, value in ratios.items():
        bounds = RATIO_BOUNDS.get(name)
        if bounds and not (bounds[0] <= value <= bounds[1]):
            out.append(name)
    return sorted(out)


def _classify_somatotype(ratios: dict[str, float]) -> str | None:
    """A coarse build label, from the two ratios that actually separate builds.

    Shoulder-to-waist separates a tapered upper body from a straight one, and
    waist-to-hip separates where mass sits. This is a descriptive label for
    prompt text, not a clinical classification, and it deliberately has an
    "average" bucket rather than forcing every body into an extreme.
    """
    taper = ratios.get("shoulder_to_waist")
    whr = ratios.get("waist_to_hip")
    if taper is None or whr is None:
        return None
    if taper >= 1.62 and whr <= 0.80:
        return "athletic, V-tapered"
    if taper >= 1.50:
        return "lean, broad-shouldered"
    if taper <= 1.20 and whr >= 0.92:
        return "full-figured, straight-waisted"
    if whr <= 0.72:
        return "hourglass, narrow-waisted"
    if taper <= 1.32:
        return "solid, rectangular"
    return "average build"


def estimate(
    observations: list[PersonObservation],
    quality: QualityReport,
    *,
    stature_cm: float | None,
    weight_kg: float | None = None,
    settings: MeasurementSettings | None = None,
    perception: PerceptionSettings | None = None,
) -> BodyMeasurements:
    """Estimate one body from every view supplied of it.

    A front view is required — it carries the breadths. A side view is optional
    and, when present, replaces the depth prior for the trunk girths with a
    measured depth, which is the single biggest accuracy win available from a
    second photo.
    """
    settings = settings or MeasurementSettings()
    perception = perception or PerceptionSettings()
    notes: list[str] = []

    front = _pick(observations, ViewLabel.FRONT) or _pick(observations, ViewLabel.BACK)
    if front is None:
        front = observations[0] if observations else None
    if front is None or not front.has_mask:
        return BodyMeasurements(notes=["no segmented front view; measurement not attempted"])

    profile = SilhouetteProfile(front.mask)
    stature_px = profile.stature_px
    breadths = _trunk_breadth_ratios(
        profile, front.keypoints, perception.min_keypoint_score
    )
    lengths = _skeleton_length_ratios(front, stature_px, perception.min_keypoint_score)
    ratios = _shape_vector(breadths, lengths)

    side = _pick(observations, ViewLabel.SIDE)
    depths: dict[str, float] = {}
    if side is not None and side.has_mask:
        depths = _trunk_breadth_ratios(
            SilhouetteProfile(side.mask), side.keypoints, perception.min_keypoint_score
        )
        usable = sorted(
            name
            for name in _SIDE_VIEW_TRUNK
            if _usable_depth(name, breadths.get(name, 0.0), depths) is not None
        )
        notes.append(
            "side view supplied; depth measured for: " + (", ".join(usable) or "nothing usable")
        )

    bad = implausible_ratios(ratios)
    if bad:
        notes.append(
            "implausible ratios, likely a segmentation failure: " + ", ".join(bad)
        )

    somatotype = _classify_somatotype(ratios)
    measurements = BodyMeasurements(
        ratios={name: ratios[name] for name in RATIO_NAMES if name in ratios},
        stature_cm=stature_cm,
        weight_kg=weight_kg,
        somatotype=somatotype,
        notes=notes,
    )
    if stature_cm is None:
        measurements.notes.append(
            "no stature supplied; proportions only, no centimetre values"
        )
        return measurements

    measurements.values = _to_centimetres(
        breadths, lengths, depths, stature_cm, quality, settings, bool(depths)
    )
    return measurements


def _pick(observations: list[PersonObservation], view: ViewLabel) -> PersonObservation | None:
    for observation in observations:
        if observation.view is view:
            return observation
    return None


def _to_centimetres(
    breadths: dict[str, float],
    lengths: dict[str, float],
    depths: dict[str, float],
    stature_cm: float,
    quality: QualityReport,
    settings: MeasurementSettings,
    two_view: bool,
) -> dict[str, MeasurementValue]:
    """Scale ratios by stature and attach an interval to each."""
    widen = confidence_multiplier(quality)
    values: dict[str, MeasurementValue] = {}

    def add(name: str, value_cm: float, sigma: float, source: MeasurementSource) -> None:
        half = settings.ci_sigmas * sigma * value_cm * widen
        values[name] = MeasurementValue(
            name=name,
            value_cm=round(value_cm, 2),
            ci_low_cm=round(max(0.0, value_cm - half), 2),
            ci_high_cm=round(value_cm + half, 2),
            source=source,
            confidence=round(float(np.clip(1.0 - sigma * widen * 4.0, 0.05, 0.99)), 3),
        )

    for name, ratio in breadths.items():
        if name.endswith("_breadth"):
            add(name, ratio * stature_cm, settings.width_sigma, MeasurementSource.SILHOUETTE)

    add("stature", stature_cm, 0.0, MeasurementSource.CLIENT)
    for name, ratio in lengths.items():
        if name in {"torso_length", "arm_length", "leg_length", "knee_height"}:
            add(name, ratio * stature_cm, settings.width_sigma, MeasurementSource.SKELETON)
    if "inseam" in breadths:
        add(
            "inseam",
            breadths["inseam"] * stature_cm,
            settings.width_sigma,
            MeasurementSource.SILHOUETTE,
        )

    girth_sigma = (
        settings.girth_sigma_two_view if two_view else settings.girth_sigma_single_view
    )
    for girth_name in _GIRTH_ORDER:
        breadth_name = GIRTH_SOURCE_BREADTH[girth_name]
        breadth_ratio = breadths.get(breadth_name)
        if breadth_ratio is None:
            continue
        breadth_cm = breadth_ratio * stature_cm
        measured_depth = _usable_depth(breadth_name, breadth_ratio, depths)
        if measured_depth is not None:
            depth_cm = measured_depth * stature_cm
            source = MeasurementSource.SILHOUETTE
            sigma = girth_sigma
        else:
            depth_cm = breadth_cm * DEPTH_OVER_BREADTH_PRIOR[girth_name]
            source = MeasurementSource.ELLIPSE_PRIOR
            # A girth resting on a population prior is never as good as one with
            # a measured depth, even in a request that supplied a side photo.
            sigma = settings.girth_sigma_single_view
        add(girth_name, girth_from_breadth_depth(breadth_cm, depth_cm), sigma, source)

    return values


def _usable_depth(
    breadth_name: str, breadth_ratio: float, depths: dict[str, float]
) -> float | None:
    """The side-view depth for a section, if it can be believed.

    Two ways it cannot: the section is a limb, which another limb hides in
    profile; or the number implies a cross-section no body has, which means the
    silhouette fused the trunk with an arm.
    """
    if breadth_name not in _SIDE_VIEW_TRUNK:
        return None
    depth = depths.get(breadth_name)
    if depth is None or breadth_ratio <= 0:
        return None
    low, high = _PLAUSIBLE_DEPTH_OVER_BREADTH
    return depth if low <= depth / breadth_ratio <= high else None


def body_mass_index(stature_cm: float, weight_kg: float) -> float:
    """BMI, for the prompt's build description. Not a health assessment."""
    metres = stature_cm / 100.0
    return round(weight_kg / (metres * metres), 1) if metres > 0 else 0.0
