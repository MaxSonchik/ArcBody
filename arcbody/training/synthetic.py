"""A parametric human-silhouette generator.

Why this exists. Public datasets that pair photographs with tape-measure ground
truth are scarce and mostly licence-encumbered, and the environments ArcBody is
developed in have no GPU. A parametric generator gives three things a scraped
dataset cannot:

* **Exact ground truth.** The body was *defined* by its ratios, so measurement
  error is measurable rather than estimated against another estimate.
* **Controlled identity.** One parameter vector rendered under different poses,
  cameras and outfits is, by construction, the same person — which is precisely
  the supervision an angular-margin loss needs.
* **Controlled nuisance.** Pose, framing, colour and background vary
  independently of shape, so invariance can be trained and then tested for.

What it is not: photorealistic. A model trained on this alone will not transfer
to photographs. It is the substrate for wiring, regression tests and accuracy
audits of the geometric estimator; ``arcbody.training.datasets`` is where a real
corpus plugs in, and ``docs/ARCHITECTURE.md`` records the intended path.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from arcbody import keypoints as kp
from arcbody.measure.schema import RATIO_NAMES, girth_from_breadth_depth
from arcbody.training.rasterise import capsule, ellipse, profile_column
from arcbody.types import ViewLabel

#: Limb lengths as fractions of stature, following the classical body-segment
#: proportions of Drillis & Contini. They are fixed rather than sampled: this
#: generator varies girth and frame, not limb-length pathology.
UPPER_ARM_LENGTH = 0.186
FOREARM_LENGTH = 0.146
HAND_LENGTH = 0.108


@dataclass(frozen=True)
class BodyParams:
    """One body, described entirely in fractions of its own stature.

    Splitting stature out from every other field is the whole point: the shape
    fields are what a photograph can recover, ``stature_cm`` is what it cannot.
    """

    stature_cm: float = 172.0

    # Vertical landmark levels, measured downwards from the crown.
    head_height: float = 0.130
    neck_level: float = 0.152
    shoulder_level: float = 0.182
    chest_level: float = 0.280
    waist_level: float = 0.400
    hip_level: float = 0.530
    crotch_level: float = 0.540
    knee_level: float = 0.715
    ankle_level: float = 0.955

    # Breadths, as seen from the front.
    head_breadth: float = 0.086
    neck_breadth: float = 0.066
    shoulder_breadth: float = 0.240
    chest_breadth: float = 0.175
    waist_breadth: float = 0.150
    hip_breadth: float = 0.190
    thigh_breadth: float = 0.098
    calf_breadth: float = 0.068
    upper_arm_breadth: float = 0.055
    forearm_breadth: float = 0.046

    # Depth over breadth of each cross-section, i.e. the truth the single-view
    # estimator has to guess with a population prior.
    chest_depth_ratio: float = 0.72
    waist_depth_ratio: float = 0.78
    hip_depth_ratio: float = 0.74
    neck_depth_ratio: float = 0.92
    thigh_depth_ratio: float = 0.95
    upper_arm_depth_ratio: float = 0.95

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    # -- ground truth -----------------------------------------------------

    def ratios(self) -> dict[str, float]:
        """The dimensionless shape vector, in canonical ``RATIO_NAMES`` order."""
        torso = self.hip_level - self.shoulder_level
        leg = self.ankle_level - self.crotch_level
        arm = UPPER_ARM_LENGTH + FOREARM_LENGTH
        values = {
            "shoulder_to_stature": self.shoulder_breadth,
            "chest_to_stature": self.chest_breadth,
            "waist_to_stature": self.waist_breadth,
            "hip_to_stature": self.hip_breadth,
            "thigh_to_stature": self.thigh_breadth,
            "upper_arm_to_stature": self.upper_arm_breadth,
            "neck_to_stature": self.neck_breadth,
            "torso_to_stature": torso,
            "leg_to_stature": leg,
            "arm_to_stature": arm,
            "waist_to_hip": self.waist_breadth / self.hip_breadth,
            "shoulder_to_waist": self.shoulder_breadth / self.waist_breadth,
            "shoulder_to_hip": self.shoulder_breadth / self.hip_breadth,
            "chest_to_waist": self.chest_breadth / self.waist_breadth,
        }
        return {name: float(values[name]) for name in RATIO_NAMES}

    def ratio_vector(self) -> np.ndarray:
        ratios = self.ratios()
        return np.array([ratios[name] for name in RATIO_NAMES], dtype=np.float32)

    def measurements_cm(self) -> dict[str, float]:
        """Exact measurements of this body, in centimetres.

        Girths use the same elliptical model the estimator uses, but with the
        subject's *true* depth ratio rather than a population prior — so the gap
        between these and an estimate is exactly the prior's error.
        """
        stature = self.stature_cm
        breadths = {
            "neck_breadth": self.neck_breadth,
            "shoulder_breadth": self.shoulder_breadth,
            "chest_breadth": self.chest_breadth,
            "waist_breadth": self.waist_breadth,
            "hip_breadth": self.hip_breadth,
            "thigh_breadth": self.thigh_breadth,
            "upper_arm_breadth": self.upper_arm_breadth,
        }
        depth_ratios = {
            "neck_girth": (self.neck_breadth, self.neck_depth_ratio),
            "chest_girth": (self.chest_breadth, self.chest_depth_ratio),
            "waist_girth": (self.waist_breadth, self.waist_depth_ratio),
            "hip_girth": (self.hip_breadth, self.hip_depth_ratio),
            "thigh_girth": (self.thigh_breadth, self.thigh_depth_ratio),
            "upper_arm_girth": (self.upper_arm_breadth, self.upper_arm_depth_ratio),
        }
        out = {name: value * stature for name, value in breadths.items()}
        for name, (breadth, ratio) in depth_ratios.items():
            width = breadth * stature
            out[name] = girth_from_breadth_depth(width, width * ratio)
        out["stature"] = stature
        out["torso_length"] = (self.hip_level - self.shoulder_level) * stature
        out["arm_length"] = (UPPER_ARM_LENGTH + FOREARM_LENGTH) * stature
        out["leg_length"] = (self.ankle_level - self.hip_level) * stature
        out["inseam"] = (1.0 - self.crotch_level) * stature
        out["knee_height"] = (1.0 - self.knee_level) * stature
        return out


@dataclass(frozen=True)
class PoseParams:
    """Nuisance variation: how the body is standing and how it was framed."""

    arm_angle_deg: float = 38.0
    arm_bend_deg: float = 6.0
    leg_spread_deg: float = 4.0
    roll_deg: float = 0.0
    subject_height_ratio: float = 0.82
    centre_x_frac: float = 0.5
    centre_y_frac: float = 0.5


@dataclass(frozen=True)
class Appearance:
    """Colours of the background and of the three garment regions."""

    background: tuple[int, int, int] = (222, 224, 228)
    skin: tuple[int, int, int] = (206, 168, 140)
    top: tuple[int, int, int] = (70, 92, 140)
    bottom: tuple[int, int, int] = (54, 58, 70)
    noise: float = 5.0


@dataclass
class RenderedBody:
    """A rendered subject and everything that was true about it by construction."""

    image: np.ndarray
    mask: np.ndarray
    keypoints: np.ndarray
    params: BodyParams
    pose: PoseParams
    view: ViewLabel
    stature_px: float


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def sample_params(rng: np.random.Generator, *, stature_cm: float | None = None) -> BodyParams:
    """Draw a plausible body.

    Three latent factors drive the correlations that matter: overall adiposity
    moves every girth together, frame trades shoulder breadth against waist to
    produce V- and H-shaped torsos, and leg length shifts the crotch. Sampling
    each breadth independently would produce bodies that no ratio prior could
    ever describe, and a model trained on them would learn nothing transferable.
    """
    adiposity = float(rng.normal(0.0, 1.0))
    frame = float(rng.normal(0.0, 1.0))
    leg = float(rng.normal(0.0, 1.0))

    def jitter(scale: float) -> float:
        return float(rng.normal(0.0, scale))

    crotch = float(np.clip(0.540 + 0.022 * leg, 0.495, 0.585))
    return BodyParams(
        stature_cm=float(stature_cm if stature_cm is not None else rng.normal(172.0, 9.0)),
        head_height=0.130 + jitter(0.004),
        neck_level=0.152 + jitter(0.004),
        shoulder_level=0.182 + jitter(0.005),
        chest_level=0.280 + jitter(0.008),
        waist_level=0.400 + jitter(0.010),
        hip_level=crotch - 0.010,
        crotch_level=crotch,
        knee_level=float(np.clip(0.715 + 0.010 * leg, 0.680, 0.750)),
        ankle_level=0.955 + jitter(0.005),
        head_breadth=0.086 + jitter(0.003),
        neck_breadth=float(np.clip(0.066 + 0.006 * adiposity + jitter(0.003), 0.048, 0.092)),
        shoulder_breadth=float(
            np.clip(0.240 + 0.016 * frame + 0.008 * adiposity + jitter(0.006), 0.190, 0.295)
        ),
        chest_breadth=float(
            np.clip(0.175 + 0.018 * adiposity + 0.006 * frame + jitter(0.006), 0.135, 0.235)
        ),
        waist_breadth=float(
            np.clip(0.150 + 0.026 * adiposity - 0.008 * frame + jitter(0.006), 0.105, 0.245)
        ),
        hip_breadth=float(np.clip(0.190 + 0.020 * adiposity + jitter(0.007), 0.145, 0.255)),
        thigh_breadth=float(np.clip(0.098 + 0.011 * adiposity + jitter(0.004), 0.065, 0.138)),
        calf_breadth=float(np.clip(0.068 + 0.006 * adiposity + jitter(0.003), 0.048, 0.096)),
        upper_arm_breadth=float(np.clip(0.055 + 0.007 * adiposity + jitter(0.003), 0.037, 0.088)),
        forearm_breadth=float(np.clip(0.046 + 0.005 * adiposity + jitter(0.002), 0.032, 0.070)),
        chest_depth_ratio=float(np.clip(0.72 + jitter(0.05), 0.58, 0.88)),
        waist_depth_ratio=float(np.clip(0.78 + jitter(0.05), 0.64, 0.94)),
        hip_depth_ratio=float(np.clip(0.74 + jitter(0.05), 0.60, 0.90)),
        neck_depth_ratio=float(np.clip(0.92 + jitter(0.03), 0.82, 1.00)),
        thigh_depth_ratio=float(np.clip(0.95 + jitter(0.03), 0.85, 1.00)),
        upper_arm_depth_ratio=float(np.clip(0.95 + jitter(0.03), 0.85, 1.00)),
    )


def sample_pose(rng: np.random.Generator, *, easy: bool = False) -> PoseParams:
    """Draw a capture: A-pose arms, a stance, framing, and a little camera roll.

    ``easy`` keeps the subject well within the quality gates, which is what the
    measurement-accuracy tests want; the default spread deliberately produces
    some captures that the gates should reject.
    """
    if easy:
        return PoseParams(
            arm_angle_deg=float(rng.uniform(34.0, 46.0)),
            arm_bend_deg=float(rng.uniform(0.0, 8.0)),
            leg_spread_deg=float(rng.uniform(2.0, 6.0)),
            roll_deg=float(rng.uniform(-3.0, 3.0)),
            subject_height_ratio=float(rng.uniform(0.78, 0.88)),
            centre_x_frac=float(rng.uniform(0.46, 0.54)),
            centre_y_frac=0.5,
        )
    return PoseParams(
        arm_angle_deg=float(rng.uniform(22.0, 55.0)),
        arm_bend_deg=float(rng.uniform(-8.0, 18.0)),
        leg_spread_deg=float(rng.uniform(0.0, 10.0)),
        roll_deg=float(rng.uniform(-9.0, 9.0)),
        subject_height_ratio=float(rng.uniform(0.55, 0.92)),
        centre_x_frac=float(rng.uniform(0.36, 0.64)),
        centre_y_frac=float(rng.uniform(0.44, 0.56)),
    )


def sample_appearance(rng: np.random.Generator) -> Appearance:
    """Draw a background and an outfit.

    Backgrounds stay light and desaturated and garments stay clearly distinct
    from them, matching the plain-backdrop capture the service asks for. The
    encoder's job is to be invariant to the outfit; the segmenter's job is to be
    unbothered by the backdrop.
    """

    def colour(low: int, high: int) -> tuple[int, int, int]:
        return tuple(int(v) for v in rng.integers(low, high, size=3))  # type: ignore[return-value]

    return Appearance(
        background=colour(198, 246),
        skin=(
            int(rng.integers(150, 235)),
            int(rng.integers(115, 195)),
            int(rng.integers(95, 170)),
        ),
        top=colour(25, 150),
        bottom=colour(20, 130),
        noise=float(rng.uniform(2.0, 9.0)),
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _limb_chain(
    origin: tuple[float, float], angle_deg: float, lengths: list[float], bend_deg: float
) -> list[tuple[float, float]]:
    """Walk a chain of segments outward from ``origin``, bending at each joint."""
    points = [origin]
    angle = math.radians(angle_deg)
    x, y = origin
    for index, length in enumerate(lengths):
        angle += math.radians(bend_deg) if index else 0.0
        x += length * math.sin(angle)
        y += length * math.cos(angle)
        points.append((x, y))
    return points


def render(
    params: BodyParams,
    pose: PoseParams,
    appearance: Appearance,
    *,
    size: tuple[int, int] = (384, 512),
    view: ViewLabel = ViewLabel.FRONT,
    rng: np.random.Generator | None = None,
) -> RenderedBody:
    """Render one capture of one body.

    The side view reuses the whole layout with depths substituted for breadths,
    which is the same assumption the estimator makes in reverse. That symmetry
    is deliberate: it means a two-view estimate recovers the true girth exactly
    when segmentation is exact, so any residual error in the tests is the
    estimator's, not the generator's.
    """
    rng = rng or np.random.default_rng()
    width, height = size
    stature_px = pose.subject_height_ratio * height
    top_y = (height - stature_px) * pose.centre_y_frac
    centre_x = width * pose.centre_x_frac

    def level(fraction: float) -> float:
        return top_y + fraction * stature_px

    def span(fraction: float) -> float:
        return fraction * stature_px

    sideways = view is ViewLabel.SIDE
    if sideways:
        trunk = {
            "neck": params.neck_breadth * params.neck_depth_ratio,
            "shoulder": params.chest_breadth * params.chest_depth_ratio * 1.12,
            "chest": params.chest_breadth * params.chest_depth_ratio,
            "waist": params.waist_breadth * params.waist_depth_ratio,
            "hip": params.hip_breadth * params.hip_depth_ratio,
        }
        limb = {
            "thigh": params.thigh_breadth * params.thigh_depth_ratio,
            "calf": params.calf_breadth,
            "upper_arm": params.upper_arm_breadth * params.upper_arm_depth_ratio,
            "forearm": params.forearm_breadth,
            "head": params.head_breadth * 1.18,
        }
    else:
        trunk = {
            "neck": params.neck_breadth,
            "shoulder": params.shoulder_breadth,
            "chest": params.chest_breadth,
            "waist": params.waist_breadth,
            "hip": params.hip_breadth,
        }
        limb = {
            "thigh": params.thigh_breadth,
            "calf": params.calf_breadth,
            "upper_arm": params.upper_arm_breadth,
            "forearm": params.forearm_breadth,
            "head": params.head_breadth,
        }

    torso_mask = np.zeros((height, width), dtype=bool)
    skin_mask = np.zeros((height, width), dtype=bool)
    legs_mask = np.zeros((height, width), dtype=bool)

    # Trunk: a continuous half-width profile through the anatomical levels.
    profile_column(
        torso_mask,
        centre_x,
        [
            (level(params.neck_level), span(trunk["neck"]) / 2.0),
            (level(params.shoulder_level), span(trunk["shoulder"]) / 2.0),
            (level(params.chest_level), span(trunk["chest"]) / 2.0),
            (level(params.waist_level), span(trunk["waist"]) / 2.0),
            (level(params.hip_level), span(trunk["hip"]) / 2.0),
            (level(params.crotch_level), span(trunk["hip"]) / 2.0 * 0.94),
        ],
    )

    # Head and neck.
    head_centre = (centre_x, level(params.head_height / 2.0))
    ellipse(
        skin_mask,
        head_centre,
        span(limb["head"]) / 2.0,
        span(params.head_height) / 2.0,
    )
    capsule(
        skin_mask,
        (centre_x, level(params.head_height * 0.9)),
        (centre_x, level(params.shoulder_level)),
        span(trunk["neck"]) / 2.0,
    )

    # Arms, hanging outward from just inside the shoulder tips.
    shoulder_dx = (span(trunk["shoulder"]) - span(limb["upper_arm"])) / 2.0
    if sideways:
        shoulder_dx *= 0.15
    shoulder_y = level(params.shoulder_level + 0.012)
    arm_r0 = span(limb["upper_arm"]) / 2.0
    arm_r1 = span(limb["forearm"]) / 2.0
    arm_points: dict[str, list[tuple[float, float]]] = {}
    for side, sign in (("left", -1.0), ("right", 1.0)):
        chain = _limb_chain(
            (centre_x + sign * shoulder_dx, shoulder_y),
            sign * pose.arm_angle_deg,
            [span(UPPER_ARM_LENGTH), span(FOREARM_LENGTH)],
            sign * pose.arm_bend_deg,
        )
        arm_points[side] = chain
        capsule(skin_mask, chain[0], chain[1], arm_r0, arm_r1 * 1.05)
        capsule(skin_mask, chain[1], chain[2], arm_r1 * 1.05, arm_r1 * 0.72)
        # The hand, continuing the forearm's direction. Rendering the arm as if
        # it stopped at the wrist would let a silhouette tracer that stops at
        # the fingertips look accurate here and be a hand too long on a photo.
        forearm = np.array(chain[2]) - np.array(chain[1])
        forearm_norm = float(np.linalg.norm(forearm))
        if forearm_norm > 1e-6:
            hand_tip = np.array(chain[2]) + forearm / forearm_norm * span(HAND_LENGTH)
            capsule(skin_mask, chain[2], tuple(hand_tip), arm_r1 * 0.80, arm_r1 * 0.45)

    # Legs.
    hip_dx = span(trunk["hip"]) * 0.26
    if sideways:
        hip_dx *= 0.20
    hip_y = level(params.crotch_level - 0.015)
    leg_points: dict[str, list[tuple[float, float]]] = {}
    for side, sign in (("left", -1.0), ("right", 1.0)):
        chain = _limb_chain(
            (centre_x + sign * hip_dx, hip_y),
            sign * pose.leg_spread_deg,
            [
                level(params.knee_level) - hip_y,
                level(params.ankle_level) - level(params.knee_level),
            ],
            0.0,
        )
        leg_points[side] = chain
        capsule(legs_mask, chain[0], chain[1], span(limb["thigh"]) / 2.0, span(limb["calf"]) / 2.0)
        capsule(
            legs_mask, chain[1], chain[2], span(limb["calf"]) / 2.0, span(limb["calf"]) / 2.0 * 0.62
        )
        # The sole is placed so the silhouette ends at exactly level(1.0):
        # stature is defined crown-to-sole, and an overshooting foot would put a
        # systematic bias into every ratio measured against the mask extent.
        foot = chain[2]
        sole_radius = span(limb["calf"]) / 2.0 * 0.55
        capsule(
            legs_mask,
            foot,
            (foot[0] + sign * span(0.010), level(1.0) - sole_radius),
            span(limb["calf"]) / 2.0 * 0.62,
            sole_radius,
        )

    body_mask = torso_mask | skin_mask | legs_mask

    # Compose colours. Garments are painted after skin so sleeves and shorts do
    # not bleed through the limbs they cover.
    image = np.empty((height, width, 3), dtype=np.float32)
    image[:] = np.asarray(appearance.background, dtype=np.float32)
    image[skin_mask] = np.asarray(appearance.skin, dtype=np.float32)
    image[legs_mask] = np.asarray(appearance.bottom, dtype=np.float32)
    image[torso_mask] = np.asarray(appearance.top, dtype=np.float32)
    if appearance.noise > 0:
        image += rng.normal(0.0, appearance.noise, size=image.shape).astype(np.float32)

    points = _ground_truth_keypoints(
        params, pose, level, centre_x, head_centre, arm_points, leg_points, sideways
    )

    rgb = np.clip(image, 0, 255).astype(np.uint8)
    if abs(pose.roll_deg) > 1e-3:
        rgb, body_mask, points = _apply_roll(rgb, body_mask, points, pose.roll_deg, appearance)

    return RenderedBody(
        image=rgb,
        mask=body_mask,
        keypoints=points,
        params=params,
        pose=pose,
        view=view,
        stature_px=float(stature_px),
    )


def _ground_truth_keypoints(
    params: BodyParams,
    pose: PoseParams,
    level,
    centre_x: float,
    head_centre: tuple[float, float],
    arm_points: dict[str, list[tuple[float, float]]],
    leg_points: dict[str, list[tuple[float, float]]],
    sideways: bool,
) -> np.ndarray:
    """COCO-17 points, read straight off the geometry that was drawn."""
    points = kp.empty()
    head_rx = params.head_breadth * 0.5
    stature_px = pose.subject_height_ratio  # only ratios are needed below

    def put(index: int, x: float, y: float, score: float = 1.0) -> None:
        points[index] = (x, y, score)

    head_half = (head_centre[1] - level(0.0)) or 1.0
    put(kp.NOSE, head_centre[0], head_centre[1] + head_half * 0.25)
    eye_dx = head_rx * 0.45 / max(stature_px, 1e-6) * 0.0  # placeholder, set below
    del eye_dx
    eye_offset = (level(head_rx) - level(0.0)) * 0.45
    put(kp.LEFT_EYE, head_centre[0] - eye_offset, head_centre[1] - head_half * 0.1)
    put(kp.RIGHT_EYE, head_centre[0] + eye_offset, head_centre[1] - head_half * 0.1)
    put(kp.LEFT_EAR, head_centre[0] - eye_offset * 1.9, head_centre[1])
    put(kp.RIGHT_EAR, head_centre[0] + eye_offset * 1.9, head_centre[1])

    put(kp.LEFT_SHOULDER, *arm_points["left"][0])
    put(kp.RIGHT_SHOULDER, *arm_points["right"][0])
    put(kp.LEFT_ELBOW, *arm_points["left"][1])
    put(kp.RIGHT_ELBOW, *arm_points["right"][1])
    put(kp.LEFT_WRIST, *arm_points["left"][2])
    put(kp.RIGHT_WRIST, *arm_points["right"][2])
    put(kp.LEFT_HIP, *leg_points["left"][0])
    put(kp.RIGHT_HIP, *leg_points["right"][0])
    put(kp.LEFT_KNEE, *leg_points["left"][1])
    put(kp.RIGHT_KNEE, *leg_points["right"][1])
    put(kp.LEFT_ANKLE, *leg_points["left"][2])
    put(kp.RIGHT_ANKLE, *leg_points["right"][2])

    if sideways:
        # In profile the far-side joints are occluded; saying otherwise would
        # teach the quality gates that a side view is fully observed.
        for index in (kp.RIGHT_EAR, kp.RIGHT_SHOULDER, kp.RIGHT_ELBOW, kp.RIGHT_WRIST):
            points[index, 2] = 0.35
    return points


def _apply_roll(
    image: np.ndarray,
    mask: np.ndarray,
    points: np.ndarray,
    roll_deg: float,
    appearance: Appearance,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate the capture about the image centre, keypoints included."""
    from PIL import Image

    height, width = mask.shape
    centre = (width / 2.0, height / 2.0)
    rotated_image = np.asarray(
        Image.fromarray(image).rotate(
            roll_deg, resample=Image.Resampling.BILINEAR, fillcolor=appearance.background
        )
    )
    rotated_mask = (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).rotate(
                roll_deg, resample=Image.Resampling.NEAREST, fillcolor=0
            )
        )
        > 127
    )

    # PIL rotates counter-clockwise for a positive angle; the inverse map takes
    # source points to their place in the rotated frame.
    theta = math.radians(-roll_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    moved = points.copy()
    dx = points[:, 0] - centre[0]
    dy = points[:, 1] - centre[1]
    moved[:, 0] = centre[0] + dx * cos_t - dy * sin_t
    moved[:, 1] = centre[1] + dx * sin_t + dy * cos_t
    moved[:, 2] = points[:, 2]
    return rotated_image, rotated_mask, moved


def render_identity(
    params: BodyParams,
    rng: np.random.Generator,
    *,
    count: int,
    size: tuple[int, int] = (384, 512),
    easy: bool = False,
    include_side: bool = False,
) -> list[RenderedBody]:
    """Several captures of one person — the positive pairs an ArcFace head needs."""
    captures = []
    for index in range(count):
        view = ViewLabel.SIDE if include_side and index % 4 == 3 else ViewLabel.FRONT
        captures.append(
            render(
                params,
                sample_pose(rng, easy=easy),
                sample_appearance(rng),
                size=size,
                view=view,
                rng=rng,
            )
        )
    return captures


def with_stature(params: BodyParams, stature_cm: float) -> BodyParams:
    """Same shape, different size — useful for scale-invariance tests."""
    return replace(params, stature_cm=stature_cm)


def cli(argv: list[str] | None = None) -> int:
    """Write a few sample renders to disk for eyeballing."""
    parser = argparse.ArgumentParser(description="Render synthetic ArcBody subjects.")
    parser.add_argument("--out", type=Path, default=Path("outputs/synthetic"))
    parser.add_argument("--identities", type=int, default=4)
    parser.add_argument("--per-identity", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--easy", action="store_true")
    args = parser.parse_args(argv)

    from arcbody.imaging import encode_png

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for identity in range(args.identities):
        params = sample_params(rng)
        for index, capture in enumerate(
            render_identity(params, rng, count=args.per_identity, easy=args.easy, include_side=True)
        ):
            stem = args.out / f"id{identity:03d}_{index:02d}_{capture.view.value}"
            stem.with_suffix(".png").write_bytes(encode_png(capture.image))
            stem.with_name(stem.name + "_mask").with_suffix(".png").write_bytes(
                encode_png(capture.mask)
            )
    print(f"wrote {args.identities * args.per_identity} captures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
