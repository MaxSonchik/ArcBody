"""Decoding request payloads into pipeline inputs."""

from __future__ import annotations

from arcbody import imaging
from arcbody.config import Settings
from arcbody.pipeline import ImageInput
from arcbody.schemas import ImagePayload


def decode_images(payloads: list[ImagePayload], settings: Settings) -> list[ImageInput]:
    """Base64 payloads to decoded images, with the per-request pixel budget applied."""
    return [
        ImageInput(
            image=imaging.decode_base64(
                payload.content_base64, max_pixels=settings.max_image_pixels
            ),
            view=payload.view_label(),
            label=payload.label,
        )
        for payload in payloads
    ]
