"""The contract every perception backend implements.

A backend's only job is to turn an image into ``PersonObservation`` values:
where the people are, where their joints are, and — when it can — which pixels
belong to them. Everything anthropometric happens downstream, so swapping YOLO
for a different detector changes one file and no measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from arcbody.types import PersonObservation


@dataclass
class PerceptionResult:
    """Everyone found in one image, best subject first."""

    observations: list[PersonObservation] = field(default_factory=list)
    backend: str = "unknown"

    @property
    def subject(self) -> PersonObservation | None:
        return self.observations[0] if self.observations else None


@runtime_checkable
class PerceptionBackend(Protocol):
    """Structural interface, so backends need no common base class."""

    name: str

    def available(self) -> bool:
        """Whether dependencies and weights are present right now."""
        ...

    def analyse(self, image: np.ndarray) -> PerceptionResult:
        """Detect people in a ``uint8`` RGB image."""
        ...


def rank_observations(
    observations: list[PersonObservation],
    image_size: tuple[int, int],
) -> list[PersonObservation]:
    """Order candidates by how much they look like *the* subject of the photo.

    Size dominates — a portrait's subject is the big one — but centredness
    breaks ties, because a bystander at the frame edge can be nearly as tall as
    the person the photo is of.
    """
    width, height = image_size
    centre_x = width / 2.0

    def score(observation: PersonObservation) -> float:
        box = observation.bbox
        relative_area = box.area / max(1.0, width * height)
        offset = abs(box.centre[0] - centre_x) / max(1.0, width / 2.0)
        return (
            relative_area
            * (1.0 - 0.35 * min(1.0, offset))
            * (0.5 + 0.5 * observation.detection_score)
        )

    return sorted(observations, key=score, reverse=True)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected blob, then fill its interior holes.

    Segmentation leaks — a shadow, a reflection, a patterned shirt read as
    background — and a silhouette with a hole in the chest measures narrower
    than the body it came from.
    """
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    if count > 1:
        sizes = ndimage.sum_labels(mask, labels, index=range(1, count + 1))
        mask = labels == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(mask)
