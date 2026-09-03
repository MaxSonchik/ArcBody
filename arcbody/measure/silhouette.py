"""Mask geometry: turning a binary silhouette into measurable structure.

Everything anthropometric that is *not* a joint-to-joint distance is read off
the silhouette here — breadths, the crotch level, the narrowest point of the
waist. The module is deliberately free of any perception import so both the
classic backend (which derives landmarks from these primitives) and the
measurement stage (which reads breadths at landmark levels) can depend on it.

Capture assumption, stated once and enforced by the quality gates: the subject
stands upright, roughly fronto-parallel, with the arms held away from the torso
(an "A-pose"). With the arms glued to the body, a chest row and an upper-arm row
are the same run of pixels and no amount of post-processing separates them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A row must hold at least this many pixels to count as part of the body;
#: single stray pixels are segmentation noise, not anatomy.
MIN_RUN_PX = 2


@dataclass(frozen=True)
class Run:
    """A horizontal stretch of foreground pixels, ``end`` exclusive."""

    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def centre(self) -> float:
        return (self.start + self.end - 1) / 2.0

    def contains(self, x: float) -> bool:
        return self.start <= x < self.end


def row_runs(row: np.ndarray, min_px: int = MIN_RUN_PX) -> list[Run]:
    """Split one mask row into its foreground runs, shortest ones dropped."""
    if not row.any():
        return []
    padded = np.concatenate(([False], row.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs = [Run(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]
    return [run for run in runs if run.width >= min_px]


class SilhouetteProfile:
    """Row-wise structure of one person's silhouette.

    Rows are indexed in image space, but every band query takes a *fraction of
    stature* measured downwards from the crown, which is how anthropometric
    landmark levels are actually specified.
    """

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = np.asarray(mask, dtype=bool)
        if self.mask.ndim != 2:
            raise ValueError(f"mask must be 2-D, got shape {self.mask.shape}")

        rows_with_body = np.flatnonzero(self.mask.any(axis=1))
        if rows_with_body.size == 0:
            raise ValueError("mask is empty; there is no silhouette to profile")

        self.top = int(rows_with_body[0])
        self.bottom = int(rows_with_body[-1])
        self._runs: dict[int, list[Run]] = {}

    # -- basic extents ----------------------------------------------------

    @property
    def stature_px(self) -> float:
        """Crown-to-heel extent in pixels. The scale reference for everything."""
        return float(self.bottom - self.top + 1)

    def runs_at(self, y: int) -> list[Run]:
        y = int(np.clip(y, 0, self.mask.shape[0] - 1))
        cached = self._runs.get(y)
        if cached is None:
            cached = row_runs(self.mask[y])
            self._runs[y] = cached
        return cached

    def row_of(self, fraction: float) -> int:
        """Image row at ``fraction`` of stature below the crown."""
        return int(round(self.top + fraction * (self.stature_px - 1)))

    def fraction_of(self, y: float) -> float:
        """Inverse of :meth:`row_of`."""
        return float((y - self.top) / max(1.0, self.stature_px - 1))

    # -- widths -----------------------------------------------------------

    def span_width(self, y: int) -> float:
        """Leftmost to rightmost foreground pixel, arms included."""
        runs = self.runs_at(y)
        if not runs:
            return 0.0
        return float(runs[-1].end - runs[0].start)

    def torso_width(self, y: int, centre_x: float) -> float:
        """Width of the run straddling ``centre_x`` — the trunk, not the arms.

        When the arms are held clear of the body a torso row reads as three runs
        (arm | trunk | arm) and picking the central one is exactly right. When
        they touch, the single fused run is returned and the caller sees an
        implausibly wide trunk, which the plausibility check downstream catches.
        """
        runs = self.runs_at(y)
        if not runs:
            return 0.0
        for run in runs:
            if run.contains(centre_x):
                return float(run.width)
        nearest = min(runs, key=lambda run: abs(run.centre - centre_x))
        return float(nearest.width)

    def limb_width(self, y: int, side: str, centre_x: float) -> float:
        """Width of the outermost run on one side of the trunk."""
        runs = [run for run in self.runs_at(y) if not run.contains(centre_x)]
        if not runs:
            return 0.0
        if side == "left":
            candidates = [run for run in runs if run.centre < centre_x]
        else:
            candidates = [run for run in runs if run.centre > centre_x]
        if not candidates:
            return 0.0
        return float(max(candidates, key=lambda run: run.width).width)

    # -- landmark search --------------------------------------------------

    def _band_rows(self, low: float, high: float) -> range:
        start = max(self.top, self.row_of(min(low, high)))
        end = min(self.bottom, self.row_of(max(low, high)))
        return range(start, end + 1)

    def trunk_is_isolated(self, y: int, centre_x: float) -> bool:
        """True when the trunk run at ``y`` has foreground on both sides of it.

        That only happens when both arms are held clear of the body, which is
        exactly the condition under which a chest or waist row measures the
        trunk rather than trunk-plus-arms.
        """
        runs = self.runs_at(y)
        if len(runs) < 3:
            return False
        return any(run.contains(centre_x) for run in runs[1:-1])

    def extreme_in_band(
        self,
        low: float,
        high: float,
        centre_x: float,
        *,
        mode: str,
        use_torso: bool = True,
        require_isolated: bool = False,
    ) -> tuple[int, float]:
        """Row of the widest (``mode="max"``) or narrowest torso width in a band.

        Returns ``(row, width_px)``. Searching for a local extremum rather than
        trusting a fixed proportion is what makes the waist a *waist* on a long-
        torsoed subject instead of a point 40% down an average body.
        """
        best_row, best_width = -1, (-np.inf if mode == "max" else np.inf)
        rows = list(self._band_rows(low, high))
        if require_isolated:
            isolated = [y for y in rows if self.trunk_is_isolated(y, centre_x)]
            # With the arms down no row qualifies. Rather than silently measuring
            # trunk-plus-arms as if it were a chest, fall back to the whole band
            # and let the plausibility check downstream flag the result.
            rows = isolated or rows
        for y in rows:
            width = self.torso_width(y, centre_x) if use_torso else self.span_width(y)
            if width <= 0:
                continue
            better = width > best_width if mode == "max" else width < best_width
            if better:
                best_row, best_width = y, width
        if best_row < 0:
            fallback = self.row_of((low + high) / 2.0)
            return fallback, self.torso_width(fallback, centre_x)
        return best_row, float(best_width)

    def inseam_px(self) -> float | None:
        """Crotch height above the sole, in pixels.

        Read straight off the silhouette. Deriving it from the hip-knee-ankle
        chain instead needs a fudge factor for the pelvis, and a fudge factor is
        a place for a systematic error to hide.
        """
        crotch = self.crotch_row()
        if crotch is None:
            return None
        return float(self.bottom - crotch + 1)

    def steepest_widening(
        self, low: float, high: float, centre_x: float, *, lag: float = 0.012
    ) -> tuple[int, float]:
        """Row where trunk width grows fastest, and the width just below it.

        This is how a shoulder is found on a silhouette: the neck is narrow, the
        acromion is wide, and the transition between them is the sharpest
        positive width gradient anywhere on the upper body. ``lag`` is the
        comparison distance in fractions of stature — wide enough to step over
        single-row segmentation jitter, short enough not to smear the edge.
        """
        step = max(1, int(round(lag * self.stature_px)))
        best_row, best_gain = -1, -np.inf
        for y in self._band_rows(low, high):
            above = self.torso_width(y - step, centre_x)
            below = self.torso_width(y + step, centre_x)
            if below <= 0:
                continue
            gain = below - above
            if gain > best_gain:
                best_row, best_gain = y, gain
        if best_row < 0:
            fallback = self.row_of((low + high) / 2.0)
            return fallback, self.torso_width(fallback, centre_x)
        # Report the width a little below the transition, where the deltoids are
        # fully in section rather than half-way through the taper.
        return best_row, self.torso_width(best_row + step, centre_x)

    def crotch_row(self) -> int | None:
        """Highest row at which the legs are still separate.

        Scanned upwards from the feet: the first row that is no longer split in
        two is the crotch. Returns ``None`` when the legs never separate, which
        happens with closed-leg poses and long skirts.
        """
        lower_limit = self.row_of(0.40)
        split_seen = False
        for y in range(self.bottom, lower_limit - 1, -1):
            count = len(self.runs_at(y))
            if count >= 2:
                split_seen = True
            elif split_seen:
                return y + 1
        return None

    def centre_x(self) -> float:
        """Horizontal centre of the trunk, from the pelvis where arms never reach."""
        rows = self._band_rows(0.45, 0.60)
        centres = [
            max(runs, key=lambda run: run.width).centre
            for runs in (self.runs_at(y) for y in rows)
            if runs
        ]
        if centres:
            return float(np.median(centres))
        columns = np.flatnonzero(self.mask.any(axis=0))
        return float((columns[0] + columns[-1]) / 2.0) if columns.size else 0.0

    def area_px(self) -> int:
        return int(self.mask.sum())

    def fill_ratio(self) -> float:
        """Silhouette area over its bounding box — a cheap sanity signal.

        A human standing with arms out fills roughly a quarter to a half of the
        box. Values near 1.0 mean the "person" is a rectangle: a failed
        segmentation, a wall, a mirror frame.
        """
        columns = np.flatnonzero(self.mask.any(axis=0))
        if columns.size == 0:
            return 0.0
        box = self.stature_px * float(columns[-1] - columns[0] + 1)
        return float(self.area_px() / box) if box > 0 else 0.0
