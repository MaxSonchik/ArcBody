"""A minimal rasteriser: capsules and ellipses into a boolean mask.

Pillow can draw shapes, but not tapered capsules, and the synthetic bodies need
limbs whose radius varies from joint to joint. Distance fields are a few lines
of numpy, exact at the sub-pixel level, and keep the renderer dependency-free.
"""

from __future__ import annotations

import numpy as np


def _subgrid(
    shape: tuple[int, int],
    x_values: tuple[float, ...],
    y_values: tuple[float, ...],
    pad: float,
):
    """Index grids for just the bounding box a shape can touch."""
    height, width = shape
    xa = int(max(0, np.floor(min(x_values) - pad)))
    xb = int(min(width, np.ceil(max(x_values) + pad) + 1))
    ya = int(max(0, np.floor(min(y_values) - pad)))
    yb = int(min(height, np.ceil(max(y_values) + pad) + 1))
    if xb <= xa or yb <= ya:
        return None
    ys, xs = np.mgrid[ya:yb, xa:xb]
    return slice(ya, yb), slice(xa, xb), xs.astype(np.float64), ys.astype(np.float64)


def capsule(
    mask: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    r0: float,
    r1: float | None = None,
) -> np.ndarray:
    """Paint a segment thickened by a radius that tapers from ``r0`` to ``r1``.

    This is the workhorse for limbs: an upper arm is genuinely a cone-ish
    cylinder, and drawing it with a constant radius makes every elbow the same
    girth as its shoulder.
    """
    r1 = r0 if r1 is None else r1
    region = _subgrid(mask.shape, (p0[0], p1[0]), (p0[1], p1[1]), max(r0, r1) + 1.0)
    if region is None:
        return mask
    rows, cols, xs, ys = region

    ax, ay = float(p0[0]), float(p0[1])
    bx, by = float(p1[0]), float(p1[1])
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        t = np.zeros_like(xs)
    else:
        t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / length_sq, 0.0, 1.0)

    nearest_x = ax + t * dx
    nearest_y = ay + t * dy
    distance = np.hypot(xs - nearest_x, ys - nearest_y)
    radius = r0 + t * (r1 - r0)
    mask[rows, cols] |= distance <= radius
    return mask


def ellipse(
    mask: np.ndarray, centre: tuple[float, float], rx: float, ry: float, angle: float = 0.0
) -> np.ndarray:
    """Paint a filled, optionally rotated ellipse."""
    reach = max(rx, ry) + 1.0
    region = _subgrid(mask.shape, (centre[0],), (centre[1],), reach)
    if region is None:
        return mask
    rows, cols, xs, ys = region

    dx = xs - float(centre[0])
    dy = ys - float(centre[1])
    if angle:
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        dx, dy = dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a
    mask[rows, cols] |= (dx / max(rx, 1e-6)) ** 2 + (dy / max(ry, 1e-6)) ** 2 <= 1.0
    return mask


def profile_column(
    mask: np.ndarray,
    centre_x: float,
    levels: list[tuple[float, float]],
) -> np.ndarray:
    """Paint a vertical body built from ``(row, half_width)`` control points.

    The trunk is not a stack of primitives but a continuous profile: half-widths
    are linearly interpolated between anatomical levels, so a waist narrower
    than both the chest above it and the hips below it comes out as an actual
    taper instead of a step.
    """
    if len(levels) < 2:
        raise ValueError("a profile needs at least two levels")
    ordered = sorted(levels, key=lambda level: level[0])
    rows = np.array([level[0] for level in ordered], dtype=np.float64)
    widths = np.array([level[1] for level in ordered], dtype=np.float64)

    y_start = int(max(0, np.floor(rows[0])))
    y_end = int(min(mask.shape[0], np.ceil(rows[-1]) + 1))
    if y_end <= y_start:
        return mask

    ys = np.arange(y_start, y_end, dtype=np.float64)
    half = np.interp(ys, rows, widths)
    xs = np.arange(mask.shape[1], dtype=np.float64)[None, :]
    mask[y_start:y_end] |= np.abs(xs - float(centre_x)) <= half[:, None]
    return mask
