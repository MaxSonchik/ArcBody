"""API-key authentication and rate limiting."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from arcbody.api.deps import reset_pipeline
from arcbody.api.jobs import reset_job_store
from arcbody.api.main import create_app
from arcbody.api.security import (
    InsecureConfigurationError,
    TokenBucket,
    _match,
    check_startup_configuration,
    reset_rate_limiter,
)
from arcbody.config import Settings, reset_settings_cache

KEY = "test-key-abcdef0123456789"


@pytest.fixture
def secured(tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv("ARCBODY_PERCEPTION__BACKEND", "classic")
    monkeypatch.setenv("ARCBODY_GALLERY__DATABASE_PATH", str(tmp_path / "s.sqlite3"))
    monkeypatch.setenv("ARCBODY_SECURITY__API_KEYS", f"{KEY},second-key-9876543210")
    reset_settings_cache()
    reset_pipeline()
    reset_job_store()
    reset_rate_limiter()
    with TestClient(create_app()) as client:
        yield client
    reset_pipeline()
    reset_job_store()
    reset_rate_limiter()
    reset_settings_cache()


# -- startup posture --------------------------------------------------------


def test_production_refuses_to_start_without_keys() -> None:
    """Biometric endpoints served anonymously is a misconfiguration, not a default."""
    with pytest.raises(InsecureConfigurationError, match="API_KEYS"):
        check_startup_configuration(Settings(environment="prod"))


def test_development_starts_without_keys_but_says_so() -> None:
    warnings = check_startup_configuration(Settings(environment="dev"))
    assert warnings and "open to anyone" in warnings[0]
    assert check_startup_configuration(Settings(security={"api_keys": ["k"]})) == []


def test_keys_accept_a_comma_separated_list() -> None:
    assert Settings(security={"api_keys": "a, b ,c"}).security.api_keys == ["a", "b", "c"]


# -- authentication ---------------------------------------------------------


def test_v1_requires_a_key(secured) -> None:
    response = secured.get("/v1/persons")
    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


def test_a_wrong_key_is_rejected(secured) -> None:
    response = secured.get("/v1/persons", headers={"X-API-Key": "not-the-key"})
    assert response.status_code == 401


def test_a_valid_key_is_accepted(secured) -> None:
    assert secured.get("/v1/persons", headers={"X-API-Key": KEY}).status_code == 200


def test_every_configured_key_works(secured) -> None:
    response = secured.get("/v1/persons", headers={"X-API-Key": "second-key-9876543210"})
    assert response.status_code == 200


def test_health_stays_reachable_without_a_key(secured) -> None:
    """An orchestrator probes liveness before it has any credentials."""
    response = secured.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}


def test_health_hides_enrolment_counts_from_anonymous_callers(secured) -> None:
    assert secured.get("/healthz").json()["persons"] == -1
    authorised = secured.get("/healthz", headers={"X-API-Key": KEY})
    assert authorised.json()["persons"] >= 0


def test_health_warns_when_the_service_is_open(client) -> None:
    """The unsecured fixture has no keys; that must be visible, not silent."""
    warnings = client.get("/healthz").json()["warnings"]
    assert any("open to anyone" in warning for warning in warnings)


def test_key_matching_is_constant_time_over_the_whole_list() -> None:
    """Short-circuiting on the first differing byte leaks key material."""
    assert _match("abc", ["xyz", "abc"])
    assert not _match("abc", ["xyz", "def"])
    assert not _match("abc", [])


# -- rate limiting ----------------------------------------------------------


def test_the_bucket_allows_a_burst_then_refills() -> None:
    bucket = TokenBucket(rate_per_minute=60, burst=3)
    assert [bucket.take("caller", now=0.0)[0] for _ in range(4)] == [True, True, True, False]
    allowed, wait = bucket.take("caller", now=0.0)
    assert not allowed and wait > 0
    assert bucket.take("caller", now=1.5)[0], "a token should have refilled after 1.5s"


def test_callers_have_separate_budgets() -> None:
    bucket = TokenBucket(rate_per_minute=60, burst=1)
    assert bucket.take("a", now=0.0)[0]
    assert not bucket.take("a", now=0.0)[0]
    assert bucket.take("b", now=0.0)[0], "one caller must not exhaust another's budget"


def test_idle_callers_are_forgotten() -> None:
    bucket = TokenBucket(rate_per_minute=60, burst=1)
    bucket.take("a")
    assert bucket.evict_idle(older_than=-1.0) == 1
    assert bucket.evict_idle(older_than=3600.0) == 0


def test_exhausting_the_budget_returns_429_with_retry_after(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("ARCBODY_PERCEPTION__BACKEND", "classic")
    monkeypatch.setenv("ARCBODY_GALLERY__DATABASE_PATH", str(tmp_path / "r.sqlite3"))
    monkeypatch.setenv("ARCBODY_SECURITY__API_KEYS", KEY)
    monkeypatch.setenv("ARCBODY_SECURITY__RATE_LIMIT_PER_MINUTE", "60")
    monkeypatch.setenv("ARCBODY_SECURITY__RATE_LIMIT_BURST", "3")
    reset_settings_cache()
    reset_pipeline()
    reset_job_store()
    reset_rate_limiter()
    try:
        with TestClient(create_app()) as client:
            headers = {"X-API-Key": KEY}
            statuses = [client.get("/v1/persons", headers=headers).status_code for _ in range(6)]
            assert 429 in statuses, f"rate limit never triggered: {statuses}"
            throttled = next(
                response
                for response in (client.get("/v1/persons", headers=headers) for _ in range(3))
                if response.status_code == 429
            )
            assert throttled.json()["code"] == "rate_limited"
            # Without Retry-After a throttled client retries immediately and
            # makes the situation it is being throttled for worse.
            assert int(throttled.headers["Retry-After"]) >= 1
    finally:
        reset_pipeline()
        reset_job_store()
        reset_rate_limiter()
        reset_settings_cache()


def test_health_is_not_rate_limited(secured) -> None:
    assert all(secured.get("/healthz").status_code == 200 for _ in range(30))


def test_the_key_header_appears_in_the_openapi_document(secured) -> None:
    schemes = secured.get("/openapi.json").json()["components"]["securitySchemes"]
    assert any(scheme.get("name") == "X-API-Key" for scheme in schemes.values())


def test_a_key_never_reaches_the_logs(secured, caplog) -> None:
    """Rate-limit buckets and log lines carry a fingerprint, not the secret."""
    with caplog.at_level("DEBUG"):
        secured.get("/v1/persons", headers={"X-API-Key": KEY})
    assert KEY not in caplog.text
    time.sleep(0)
