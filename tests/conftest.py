"""Shared fixtures.

Two decisions shape the whole suite. The perception backend is pinned to
``classic`` so tests never depend on downloading YOLO weights or on a detector's
nondeterminism, and the gallery is pointed at a temporary file so no test can
see another's data. Both are set through the environment before any ArcBody
module reads settings.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("ARCBODY_PERCEPTION__BACKEND", "classic")
os.environ.setdefault("ARCBODY_LOG_LEVEL", "CRITICAL")

from arcbody.config import Settings  # noqa: E402
from arcbody.perception.classic import ClassicPerception  # noqa: E402
from arcbody.training.synthetic import (  # noqa: E402
    BodyParams,
    RenderedBody,
    render,
    sample_appearance,
    sample_params,
    sample_pose,
)
from arcbody.types import ViewLabel  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    """A generator seeded per test, so failures reproduce."""
    return np.random.default_rng(1234)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        perception={"backend": "classic"},
        gallery={"database_path": tmp_path / "gallery.sqlite3"},
    )


@pytest.fixture
def perception(settings: Settings) -> ClassicPerception:
    return ClassicPerception(settings.perception)


@pytest.fixture
def subject(rng: np.random.Generator) -> BodyParams:
    return sample_params(rng)


@pytest.fixture
def capture(subject: BodyParams, rng: np.random.Generator) -> RenderedBody:
    """One clean frontal photo of ``subject``."""
    return render(
        subject,
        sample_pose(rng, easy=True),
        sample_appearance(rng),
        size=(384, 512),
        rng=rng,
    )


@pytest.fixture
def shoot(rng: np.random.Generator):
    """Factory for extra captures of any subject."""

    def _shoot(
        params: BodyParams, *, view: ViewLabel = ViewLabel.FRONT, easy: bool = True
    ) -> RenderedBody:
        return render(
            params,
            sample_pose(rng, easy=easy),
            sample_appearance(rng),
            size=(384, 512),
            view=view,
            rng=rng,
        )

    return _shoot
