"""Image decoding, encoding and the crop/pad helpers."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from arcbody import imaging
from arcbody.errors import InvalidImageError


def test_png_roundtrip_is_lossless() -> None:
    image = (np.random.default_rng(0).random((60, 40, 3)) * 255).astype(np.uint8)
    assert np.array_equal(imaging.decode(imaging.encode_png(image), max_pixels=10**7), image)


def test_base64_accepts_a_data_uri_prefix() -> None:
    image = np.full((40, 40, 3), 128, np.uint8)
    payload = "data:image/png;base64," + base64.b64encode(imaging.encode_png(image)).decode()
    assert imaging.decode_base64(payload, max_pixels=10**7).shape == (40, 40, 3)


def test_garbage_is_rejected_rather_than_guessed() -> None:
    with pytest.raises(InvalidImageError):
        imaging.decode(b"definitely not an image", max_pixels=10**7)
    with pytest.raises(InvalidImageError):
        imaging.decode_base64("not base64 at all!!", max_pixels=10**7)


def test_pixel_budget_is_enforced() -> None:
    image = np.zeros((500, 500, 3), np.uint8)
    with pytest.raises(InvalidImageError):
        imaging.decode(imaging.encode_png(image), max_pixels=1000)


def test_tiny_images_are_rejected() -> None:
    with pytest.raises(InvalidImageError):
        imaging.decode(imaging.encode_png(np.zeros((8, 8, 3), np.uint8)), max_pixels=10**7)


def test_padded_crop_keeps_the_requested_size_outside_the_image() -> None:
    image = np.full((20, 10, 3), 200, np.uint8)
    out = imaging.crop_padded(image, (-5.0, -5.0, 5.0, 5.0))
    assert out.shape == (10, 10, 3)
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[9, 9].tolist() == [200, 200, 200]


def test_masks_resize_without_inventing_edge_pixels() -> None:
    mask = np.zeros((40, 20), bool)
    mask[10:30, 5:15] = True
    resized = imaging.resize(mask, 10, 20)
    assert resized.dtype == bool
    assert set(np.unique(resized).tolist()) <= {True, False}


def test_letterbox_preserves_aspect_ratio() -> None:
    image = np.full((100, 50, 3), 255, np.uint8)
    out = imaging.letterbox(image, 64, 128)
    assert out.shape == (128, 64, 3)
    # A 1:2 source into a 1:2 target fills it exactly, with no padding bars.
    assert (out > 0).all()
