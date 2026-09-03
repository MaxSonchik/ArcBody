"""The measurement vocabulary shared by the estimator, the data generator and
the prompt builder.

Two rules hold everywhere in ArcBody:

* **Ratios are primary.** A photo determines a body's *proportions*; it cannot
  determine its size. Every learned quantity is dimensionless, and centimetres
  only appear once a client-supplied stature multiplies them back in.
* **A girth is never measured, only reconstructed.** One camera sees breadth,
  never depth. Girths come from an elliptical cross-section model whose depth is
  either measured on a second (side) photo or filled in from a population prior.
  The two cases are reported with different sources and different intervals.
"""

from __future__ import annotations

from typing import Final

#: Breadths readable straight off a frontal silhouette.
BREADTHS: Final[tuple[str, ...]] = (
    "neck_breadth",
    "shoulder_breadth",
    "chest_breadth",
    "waist_breadth",
    "hip_breadth",
    "thigh_breadth",
    "upper_arm_breadth",
)

#: Circumferences, reconstructed from a breadth plus a depth.
GIRTHS: Final[tuple[str, ...]] = (
    "neck_girth",
    "chest_girth",
    "waist_girth",
    "hip_girth",
    "thigh_girth",
    "upper_arm_girth",
)

#: Joint-to-joint distances, readable from the skeleton alone.
LENGTHS: Final[tuple[str, ...]] = (
    "stature",
    "torso_length",
    "arm_length",
    "leg_length",
    "inseam",
    "knee_height",
)

MEASUREMENT_NAMES: Final[tuple[str, ...]] = BREADTHS + GIRTHS + LENGTHS

#: The breadth each girth is reconstructed from.
GIRTH_SOURCE_BREADTH: Final[dict[str, str]] = {
    "neck_girth": "neck_breadth",
    "chest_girth": "chest_breadth",
    "waist_girth": "waist_breadth",
    "hip_girth": "hip_breadth",
    "thigh_girth": "thigh_breadth",
    "upper_arm_girth": "upper_arm_breadth",
}

#: Population depth-to-breadth ratios of the body's cross-sections.
#:
#: These are the single largest source of error in a one-photo girth, and they
#: are deliberately sitting in one visible table rather than buried in the
#: estimator. They are round numbers from the standard anthropometric range for
#: adults, not a fit to any particular population: recalibrate them against your
#: own measured cohort before quoting girths as anything but estimates. A limb
#: is close to circular, a waist is the flattest section, hence the spread.
DEPTH_OVER_BREADTH_PRIOR: Final[dict[str, float]] = {
    "neck_girth": 0.92,
    "chest_girth": 0.72,
    "waist_girth": 0.78,
    "hip_girth": 0.74,
    "thigh_girth": 0.95,
    "upper_arm_girth": 0.95,
}

#: Anatomical level of each landmark, as a fraction of stature below the crown.
#:
#: Used as the *centre of a search band*, never as the answer: the estimator
#: looks for the actual local minimum or maximum of the silhouette nearby, so a
#: long-torsoed subject gets their own waist rather than the average one.
LANDMARK_LEVELS: Final[dict[str, float]] = {
    "neck": 0.150,
    "shoulder": 0.182,
    "chest": 0.280,
    "waist": 0.400,
    "hip": 0.530,
    "crotch": 0.540,
    "knee": 0.715,
    "ankle": 0.955,
}

#: Half-width of the band searched around each level, in fractions of stature.
LANDMARK_SEARCH_HALF_BAND: Final[dict[str, float]] = {
    "neck": 0.025,
    "chest": 0.035,
    "waist": 0.055,
    "hip": 0.045,
}

#: Whether the estimator looks for a maximum or a minimum of trunk width.
LANDMARK_EXTREMUM: Final[dict[str, str]] = {
    "neck": "min",
    "chest": "max",
    "waist": "min",
    "hip": "max",
}

#: Dimensionless shape descriptors. This vector is what the encoder's auxiliary
#: head regresses and what survives an unknown stature, so it is also the
#: comparison basis for "did the generator keep the body?".
RATIO_NAMES: Final[tuple[str, ...]] = (
    "shoulder_to_stature",
    "chest_to_stature",
    "waist_to_stature",
    "hip_to_stature",
    "thigh_to_stature",
    "upper_arm_to_stature",
    "neck_to_stature",
    "torso_to_stature",
    "leg_to_stature",
    "arm_to_stature",
    "waist_to_hip",
    "shoulder_to_waist",
    "shoulder_to_hip",
    "chest_to_waist",
)

#: Plausible range of each ratio in an adult population. Anything outside is a
#: failed measurement rather than an unusual body, and is reported as such.
RATIO_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "shoulder_to_stature": (0.180, 0.300),
    "chest_to_stature": (0.130, 0.240),
    "waist_to_stature": (0.100, 0.250),
    "hip_to_stature": (0.140, 0.260),
    "thigh_to_stature": (0.060, 0.140),
    "upper_arm_to_stature": (0.035, 0.090),
    "neck_to_stature": (0.045, 0.095),
    "torso_to_stature": (0.230, 0.395),
    "leg_to_stature": (0.400, 0.560),
    "arm_to_stature": (0.300, 0.420),
    "waist_to_hip": (0.600, 1.150),
    "shoulder_to_waist": (0.900, 2.200),
    "shoulder_to_hip": (0.850, 1.600),
    "chest_to_waist": (0.750, 1.700),
}

RATIO_INDEX: Final[dict[str, int]] = {name: i for i, name in enumerate(RATIO_NAMES)}
NUM_RATIOS: Final[int] = len(RATIO_NAMES)


def ellipse_perimeter(semi_major: float, semi_minor: float) -> float:
    """Ramanujan's second approximation to an ellipse perimeter.

    Accurate to better than 1e-5 relative for the eccentricities a human
    cross-section reaches, which is far below the error in the depth prior it
    consumes — so the approximation is never the limiting factor.
    """
    a, b = max(semi_major, semi_minor), min(semi_major, semi_minor)
    if a <= 0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return 3.141592653589793 * (a + b) * (1.0 + (3.0 * h) / (10.0 + (4.0 - 3.0 * h) ** 0.5))


def girth_from_breadth_depth(breadth: float, depth: float) -> float:
    """Circumference of an elliptical cross-section of the given axes."""
    return ellipse_perimeter(breadth / 2.0, depth / 2.0)
