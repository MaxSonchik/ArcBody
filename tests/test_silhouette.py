"""Silhouette geometry primitives."""

from __future__ import annotations

import numpy as np
import pytest

from arcbody.measure.silhouette import SilhouetteProfile, row_runs


def figure() -> np.ndarray:
    """A blocky torso with the arms held clear and two separate legs."""
    mask = np.zeros((400, 200), bool)
    cx = 100
    mask[20:60, cx - 14 : cx + 14] = True  # head
    mask[60:76, cx - 9 : cx + 9] = True  # neck
    mask[76:150, cx - 34 : cx + 34] = True  # chest
    mask[150:200, cx - 24 : cx + 24] = True  # waist
    mask[200:240, cx - 40 : cx + 40] = True  # hips
    mask[80:110, cx - 70 : cx - 44] = True  # arms, with a real gap
    mask[80:110, cx + 44 : cx + 70] = True
    mask[240:380, cx - 32 : cx - 4] = True  # legs
    mask[240:380, cx + 4 : cx + 32] = True
    return mask


def test_runs_drop_single_pixel_noise() -> None:
    row = np.zeros(50, bool)
    row[10:20] = True
    row[30] = True
    assert [(r.start, r.end) for r in row_runs(row)] == [(10, 20)]


def test_empty_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        SilhouetteProfile(np.zeros((10, 10), bool))


def test_stature_spans_crown_to_sole() -> None:
    profile = SilhouetteProfile(figure())
    assert profile.top == 20
    assert profile.bottom == 379
    assert profile.stature_px == 360


def test_torso_width_ignores_the_arms() -> None:
    profile = SilhouetteProfile(figure())
    centre = profile.centre_x()
    row = 90  # a row crossing chest and both arms
    assert len(profile.runs_at(row)) == 3
    assert profile.torso_width(row, centre) == 68
    assert profile.span_width(row) == 140


def test_trunk_isolation_detects_clear_arms() -> None:
    profile = SilhouetteProfile(figure())
    centre = profile.centre_x()
    assert profile.trunk_is_isolated(90, centre)
    assert not profile.trunk_is_isolated(180, centre)


def test_crotch_is_the_highest_row_with_separate_legs() -> None:
    profile = SilhouetteProfile(figure())
    assert profile.crotch_row() == 240


def test_inseam_measures_from_the_crotch_to_the_sole() -> None:
    profile = SilhouetteProfile(figure())
    assert profile.inseam_px() == pytest.approx(140, abs=1)


def test_extreme_search_finds_the_local_minimum() -> None:
    profile = SilhouetteProfile(figure())
    centre = profile.centre_x()
    row, width = profile.extreme_in_band(0.36, 0.50, centre, mode="min")
    assert width == 48  # the waist block, not the chest above or hips below
    assert 150 <= row < 200


def test_steepest_widening_finds_the_shoulder_not_the_widest_row() -> None:
    profile = SilhouetteProfile(figure())
    centre = profile.centre_x()
    row, _ = profile.steepest_widening(0.10, 0.22, centre)
    # The shoulder is the neck-to-chest transition at row 76, not the hips.
    assert 66 <= row <= 92


def test_fill_ratio_flags_a_solid_rectangle() -> None:
    solid = np.ones((100, 50), bool)
    assert SilhouetteProfile(solid).fill_ratio() == pytest.approx(1.0)
    assert SilhouetteProfile(figure()).fill_ratio() < 0.6
