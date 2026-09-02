"""Image decoding and the small raster helpers the rest of the package needs.

Pillow is the only image dependency here; OpenCV is optional and confined to the
YOLO perception backend. Everything downstream works on ``uint8`` RGB arrays of
shape ``(H, W, 3)``.
"""

from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from arcbody.errors import InvalidImageError

#: Below this, a body crop carries too few pixels for any silhouette width to
#: be meaningful.
MIN_SIDE_PX = 32


def decode(data: bytes, *, max_pixels: int) -> np.ndarray:
    """Decode image bytes into a ``uint8`` RGB array.

    EXIF orientation is applied, because phone cameras store portrait shots
    rotated and a sideways subject fails every quality gate for the wrong reason.
    """
    if not data:
        raise InvalidImageError("empty image payload")
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width * height > max_pixels:
                raise InvalidImageError(
                    "image exceeds the pixel budget",
                    pixels=width * height,
                    max_pixels=max_pixels,
                )
            oriented = ImageOps.exif_transpose(image) or image
            array = np.asarray(oriented.convert("RGB"), dtype=np.uint8)
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"could not decode image: {exc}") from exc

    if array.ndim != 3 or array.shape[2] != 3:
        raise InvalidImageError(f"expected an RGB image, got shape {array.shape}")
    if min(array.shape[0], array.shape[1]) < MIN_SIDE_PX:
        raise InvalidImageError(
            "image is too small to analyse",
            width=int(array.shape[1]),
            height=int(array.shape[0]),
            min_side=MIN_SIDE_PX,
        )
    return array


def decode_base64(payload: str, *, max_pixels: int) -> np.ndarray:
    """Decode a base64 string, tolerating a ``data:image/...;base64,`` prefix."""
    text = payload.strip()
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError(f"payload is not valid base64: {exc}") from exc
    return decode(raw, max_pixels=max_pixels)


def load(path: str | Path, *, max_pixels: int = 40_000_000) -> np.ndarray:
    """Read an image from disk."""
    return decode(Path(path).read_bytes(), max_pixels=max_pixels)


def encode_png(array: np.ndarray) -> bytes:
    """Encode a ``uint8`` array (grayscale or RGB) as PNG bytes."""
    data = np.asarray(array)
    if data.dtype == bool:
        data = (data.astype(np.uint8) * 255)
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    mode = "L" if data.ndim == 2 else "RGB"
    buffer = io.BytesIO()
    Image.fromarray(data, mode=mode).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def encode_png_base64(array: np.ndarray) -> str:
    """PNG-encode and base64 it, ready to drop into a JSON response."""
    return base64.b64encode(encode_png(array)).decode("ascii")


def crop(array: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Crop with integer rounding, clipped to the array bounds.

    Always returns at least a 1x1 region so callers never handle empty arrays.
    """
    height, width = array.shape[:2]
    x1 = int(np.clip(np.floor(box[0]), 0, width - 1))
    y1 = int(np.clip(np.floor(box[1]), 0, height - 1))
    x2 = int(np.clip(np.ceil(box[2]), x1 + 1, width))
    y2 = int(np.clip(np.ceil(box[3]), y1 + 1, height))
    return array[y1:y2, x1:x2]


def crop_padded(
    array: np.ndarray, box: tuple[float, float, float, float], fill: int = 0
) -> np.ndarray:
    """Crop a box that may extend past the image, padding instead of clipping.

    Clipping would silently change the crop's aspect ratio, and the encoder's
    framing contract depends on that ratio being exactly constant. Padding keeps
    the geometry and marks the missing area as "not body", which is true.
    """
    x1 = int(np.floor(box[0]))
    y1 = int(np.floor(box[1]))
    x2 = max(x1 + 1, int(np.ceil(box[2])))
    y2 = max(y1 + 1, int(np.ceil(box[3])))

    height, width = array.shape[:2]
    shape = (y2 - y1, x2 - x1) + array.shape[2:]
    canvas = (
        np.zeros(shape, dtype=bool)
        if array.dtype == bool
        else np.full(shape, fill, dtype=array.dtype)
    )

    src_x1, src_y1 = max(0, x1), max(0, y1)
    src_x2, src_y2 = min(width, x2), min(height, y2)
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[src_y1 - y1 : src_y2 - y1, src_x1 - x1 : src_x2 - x1] = array[
            src_y1:src_y2, src_x1:src_x2
        ]
    return canvas


def resize(array: np.ndarray, width: int, height: int, *, nearest: bool = False) -> np.ndarray:
    """Resize to exactly ``width`` x ``height``.

    Masks must use nearest-neighbour: bilinear resampling of a boolean mask
    invents half-pixels along the silhouette edge, which is exactly where the
    width measurements are taken.
    """
    data = np.asarray(array)
    is_mask = data.dtype == bool
    if is_mask:
        data = data.astype(np.uint8) * 255
    mode = "L" if data.ndim == 2 else "RGB"
    resample = Image.Resampling.NEAREST if (nearest or is_mask) else Image.Resampling.BILINEAR
    resized = Image.fromarray(data.astype(np.uint8), mode=mode).resize(
        (int(width), int(height)), resample=resample
    )
    out = np.asarray(resized)
    return out > 127 if is_mask else out


def letterbox(
    array: np.ndarray, width: int, height: int, *, fill: int = 0
) -> np.ndarray:
    """Resize preserving aspect ratio, padding the remainder with ``fill``.

    The encoder is trained on letterboxed crops so that a wide subject is never
    squeezed into looking narrower than they are — aspect ratio *is* the signal.
    """
    data = np.asarray(array)
    source_h, source_w = data.shape[:2]
    scale = min(width / source_w, height / source_h)
    new_w = max(1, int(round(source_w * scale)))
    new_h = max(1, int(round(source_h * scale)))
    resized = resize(data, new_w, new_h)

    shape = (height, width) if data.ndim == 2 else (height, width, data.shape[2])
    canvas = np.full(shape, fill, dtype=resized.dtype)
    if data.dtype == bool:
        canvas = np.zeros(shape, dtype=bool)
    top = (height - new_h) // 2
    left = (width - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas
