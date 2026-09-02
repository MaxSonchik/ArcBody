"""Control maps: the image-space half of what a generator needs.

Text tells a diffusion model what kind of body to draw. Control maps tell it
*where* — and for body shape that is the half that actually holds. A caption
saying "broad shoulders" is a suggestion the model weighs against its prior; a
pose skeleton and a silhouette are geometry it is conditioned on.

Every map is rendered in the same canonical frame as the encoder's crop, so the
skeleton, the silhouette and the normalised photo overlay each other exactly and
can be stacked as multi-ControlNet inputs without any alignment work by the
caller.

Colours follow the OpenPose convention. That is not decoration: pose adapters
were trained on those exact limb colours, and a skeleton drawn in arbitrary
colours conditions markedly worse than one drawn in the expected ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from arcbody import imaging
from arcbody import keypoints as kp
from arcbody.config import GenAISettings
from arcbody.embed.crop import canonical_box
from arcbody.types import BoundingBox, PersonObservation

#: Joint disc radius and limb stroke width, as fractions of the map's height.
JOINT_RADIUS = 0.012
LIMB_WIDTH = 0.010

#: Keypoint marker colours, again the OpenPose palette.
JOINT_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170),
)


@dataclass
class ControlMaps:
    """The rendered maps, as raw arrays. The API layer PNG-encodes them."""

    pose: np.ndarray
    silhouette: np.ndarray
    normalised_crop: np.ndarray
    frame: BoundingBox

    def as_png_base64(self) -> dict[str, str]:
        return {
            "pose": imaging.encode_png_base64(self.pose),
            "silhouette": imaging.encode_png_base64(self.silhouette),
            "normalised_crop": imaging.encode_png_base64(self.normalised_crop),
        }


def _to_frame(
    points: np.ndarray, frame: BoundingBox, width: int, height: int
) -> np.ndarray:
    """Map image-space keypoints into the control map's pixel grid."""
    scale_x = width / max(frame.width, 1e-6)
    scale_y = height / max(frame.height, 1e-6)
    moved = points.copy()
    moved[:, 0] = (points[:, 0] - frame.x1) * scale_x
    moved[:, 1] = (points[:, 1] - frame.y1) * scale_y
    return moved


def render_pose(
    observation: PersonObservation,
    frame: BoundingBox,
    *,
    width: int,
    height: int,
    min_score: float = 0.3,
) -> np.ndarray:
    """An OpenPose-style skeleton on black."""
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    points = _to_frame(observation.keypoints, frame, width, height)

    limb_width = max(2, int(LIMB_WIDTH * height))
    for index, (start, end) in enumerate(kp.SKELETON):
        if points[start, 2] < min_score or points[end, 2] < min_score:
            continue
        draw.line(
            [tuple(points[start, :2]), tuple(points[end, :2])],
            fill=kp.SKELETON_COLORS[index % len(kp.SKELETON_COLORS)],
            width=limb_width,
        )

    radius = max(2, int(JOINT_RADIUS * height))
    for index in range(kp.NUM_KEYPOINTS):
        if points[index, 2] < min_score:
            continue
        x, y = float(points[index, 0]), float(points[index, 1])
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=JOINT_COLORS[index % len(JOINT_COLORS)],
        )
    return np.asarray(canvas)


def render_silhouette(
    observation: PersonObservation, frame: BoundingBox, *, width: int, height: int
) -> np.ndarray:
    """The body as a white mask on black, in the canonical frame."""
    if observation.mask is None:
        return np.zeros((height, width), dtype=np.uint8)
    patch = imaging.crop_padded(observation.mask, frame.as_tuple())
    resized = imaging.resize(patch, width, height, nearest=True)
    return (resized.astype(np.uint8) * 255)


def build(
    image: np.ndarray,
    observation: PersonObservation,
    settings: GenAISettings | None = None,
    *,
    min_score: float = 0.3,
) -> ControlMaps:
    """Render all three maps for one subject in one shared frame."""
    settings = settings or GenAISettings()
    width = settings.control_map_width
    height = settings.control_map_height
    frame = canonical_box(observation, height / width)

    crop = imaging.crop_padded(image, frame.as_tuple())
    return ControlMaps(
        pose=render_pose(
            observation, frame, width=width, height=height, min_score=min_score
        ),
        silhouette=render_silhouette(observation, frame, width=width, height=height),
        normalised_crop=imaging.resize(crop, width, height),
        frame=frame,
    )
