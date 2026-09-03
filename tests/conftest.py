"""Shared fixtures.

Two decisions shape the whole suite. The perception backend is pinned to
``classic`` so tests never depend on downloading YOLO weights or on a detector's
nondeterminism, and the gallery is pointed at a temporary file so no test can
see another's data. Both are set through the environment before any ArcBody
module reads settings.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


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

@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> Iterator[TestClient]:
    """An unsecured app over a private database.

    No API keys are configured, so authentication is off — which is itself
    worth testing, because the service must say so rather than pass silently.
    """
    from fastapi.testclient import TestClient as _TestClient

    from arcbody.api.deps import reset_pipeline
    from arcbody.api.jobs import reset_job_store
    from arcbody.api.main import create_app
    from arcbody.api.security import reset_rate_limiter
    from arcbody.config import reset_settings_cache

    monkeypatch.setenv("ARCBODY_PERCEPTION__BACKEND", "classic")
    monkeypatch.setenv("ARCBODY_GALLERY__DATABASE_PATH", str(tmp_path / "api.sqlite3"))
    monkeypatch.delenv("ARCBODY_SECURITY__API_KEYS", raising=False)
    reset_settings_cache()
    reset_pipeline()
    reset_job_store()
    reset_rate_limiter()
    with _TestClient(create_app()) as test_client:
        yield test_client
    reset_pipeline()
    reset_job_store()
    reset_rate_limiter()
    reset_settings_cache()
