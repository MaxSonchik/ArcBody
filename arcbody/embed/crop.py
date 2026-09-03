"""Producing the canonical body crop the encoder sees.

Framing is part of the metric. If a person photographed with their arms out got
a wider crop than the same person with their arms down, the letterboxed body
would come out smaller, and the encoder would have to learn to undo a scale
change that carries no information about who they are.

So the crop is built around *stature*, not around the detection box: a fixed 1:2
window centred on the subject, tall enough to hold them head to foot, widened
only when a pose genuinely needs it — and then made taller to match, so the
aspect ratio never changes and the resize never squeezes.
"""

from __future__ import annotations

import numpy as np

from arcbody import imaging
from arcbody.types import BoundingBox, PersonObservation

#: Fraction of the subject's height added as margin on every side.
MARGIN = 0.06

#: ImageNet statistics. The trunk may be initialised from a checkpoint trained
#: with them, and matching the normalisation is free.
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def canonical_box(observation: PersonObservation, aspect: float) -> BoundingBox:
    """A window of fixed ``height / width`` aspect that contains the subject."""
    box = observation.bbox
    width, height = observation.image_size
    centre_x, centre_y = box.centre

    # The window is anchored on the subject's *height* and nothing else, with
    # the width following from the fixed aspect ratio. Anchoring on the
    # detection box instead would make the framing pose-dependent — arms out
    # widens the box, the letterbox then shrinks the body, and the encoder would
    # have to learn to undo a scale change that says nothing about who this is.
    # Stature is the one stable reference the whole service already rests on, so
    # it sets the scale here too. A very wide A-pose loses the outer edge of the
    # hands; the shoulders, waist, hips and legs that carry body shape are
    # always inside. The window is deliberately not clipped to the image —
    # clipping would change the aspect ratio — and the caller pads instead.
    box_height = box.height * (1.0 + 2.0 * MARGIN)
    box_width = box_height / aspect
    del width, height

    return BoundingBox(
        x1=centre_x - box_width / 2.0,
        y1=centre_y - box_height / 2.0,
        x2=centre_x + box_width / 2.0,
        y2=centre_y + box_height / 2.0,
    )


def build_tensor_input(
    image: np.ndarray,
    observation: PersonObservation,
    *,
    input_width: int,
    input_height: int,
) -> np.ndarray:
    """Crop, resize and normalise into the ``(4, H, W)`` array the trunk takes.

    The fourth channel is the silhouette. When segmentation failed it is filled
    with zeros rather than ones: an all-ones mask would assert "the whole crop is
    body", which is a confident lie, while all-zeros is a distinctive pattern the
    network can learn to treat as "no shape information here".
    """
    aspect = input_height / input_width
    box = canonical_box(observation, aspect)

    patch = imaging.crop_padded(image, box.as_tuple())
    resized = imaging.resize(patch, input_width, input_height)
    rgb = (resized.astype(np.float32) / 255.0 - RGB_MEAN) / RGB_STD

    if observation.mask is not None:
        mask_patch = imaging.crop_padded(observation.mask, box.as_tuple())
        mask = imaging.resize(mask_patch, input_width, input_height, nearest=True)
        mask_channel = mask.astype(np.float32)
    else:
        mask_channel = np.zeros((input_height, input_width), dtype=np.float32)

    stacked = np.concatenate([rgb, mask_channel[:, :, None]], axis=2)
    return np.ascontiguousarray(stacked.transpose(2, 0, 1))
