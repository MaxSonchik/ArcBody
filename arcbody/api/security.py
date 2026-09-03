"""API-key authentication and per-caller rate limiting.

Body embeddings and measurements are biometric data. An open ``/v1`` surface is
therefore treated as a misconfiguration, not a convenience: in ``prod`` the
service refuses to start without keys, and everywhere else it starts with auth
off but says so on ``/healthz`` and in the log. A quiet fallback to "no
authentication" is the failure mode this module exists to prevent.

Two deliberate limits, stated rather than hidden:

* Keys are compared in constant time but stored as plaintext in the
  environment. That is proportionate for a single-tenant service; a multi-tenant
  one wants hashed keys and per-key scopes.
* The rate limiter is per process. Behind several replicas each gets its own
  budget, so the configured number is a per-worker ceiling, not a global one.
  Making it global needs a shared store, and pretending otherwise would be
  worse than documenting it.
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
from dataclasses import dataclass

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader

from arcbody.config import Settings, get_settings
from arcbody.errors import ArcBodyError

logger = logging.getLogger(__name__)

#: Identifier used for rate limiting when authentication is disabled.
ANONYMOUS = "anonymous"


class UnauthorizedError(ArcBodyError):
    """No key, or a key that is not on the list."""

    code = "unauthorized"
    http_status = 401


class RateLimitedError(ArcBodyError):
    """The caller has spent their budget."""

    code = "rate_limited"
    http_status = 429


class InsecureConfigurationError(RuntimeError):
    """Production was asked to run without authentication."""


def check_startup_configuration(settings: Settings) -> list[str]:
    """Validate the security posture at boot. Returns warnings, or raises.

    Called from the app's lifespan so a misconfigured production deployment
    fails at startup — where someone is watching — rather than on the first
    request from someone who should not have been able to make it.
    """
    if settings.security.api_keys:
        return []
    if settings.environment == "prod":
        raise InsecureConfigurationError(
            "ARCBODY_SECURITY__API_KEYS is empty in a production environment. "
            "The /v1 endpoints return and store biometric data and must not be "
            "served anonymously. Set keys, or set ARCBODY_ENVIRONMENT to dev."
        )
    message = (
        "no API keys configured: the /v1 endpoints are open to anyone who can reach "
        "this process. Acceptable for local development only."
    )
    logger.warning(message)
    return [message]


@dataclass
class Caller:
    """Who is making this request, for rate limiting and logs."""

    identity: str
    authenticated: bool

    @property
    def is_anonymous(self) -> bool:
        return not self.authenticated


class TokenBucket:
    """Per-caller token bucket.

    A bucket rather than a fixed window because bursts are the normal shape of
    this traffic — a client enrols a handful of photos at once, then goes quiet
    — and a fixed window either rejects that burst or permits twice the rate
    across a window boundary.
    """

    def __init__(self, *, rate_per_minute: int, burst: int) -> None:
        self.rate_per_second = max(rate_per_minute, 1) / 60.0
        self.capacity = float(max(burst, 1))
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def take(self, key: str, *, now: float | None = None) -> tuple[bool, float]:
        """Spend one token. Returns ``(allowed, seconds_until_next_token)``."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._state.get(key, (self.capacity, moment))
            tokens = min(self.capacity, tokens + (moment - last) * self.rate_per_second)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, moment)
                return True, 0.0
            self._state[key] = (tokens, moment)
            return False, (1.0 - tokens) / self.rate_per_second

    def evict_idle(self, *, older_than: float = 3600.0) -> int:
        """Forget callers that have been quiet, so the map cannot grow forever."""
        cutoff = time.monotonic() - older_than
        with self._lock:
            stale = [key for key, (_, last) in self._state.items() if last < cutoff]
            for key in stale:
                del self._state[key]
        return len(stale)


_limiter: TokenBucket | None = None
_limiter_lock = threading.Lock()


def get_rate_limiter(settings: Settings | None = None) -> TokenBucket:
    global _limiter
    if _limiter is None:
        with _limiter_lock:
            if _limiter is None:
                resolved = settings or get_settings()
                _limiter = TokenBucket(
                    rate_per_minute=resolved.security.rate_limit_per_minute,
                    burst=resolved.security.rate_limit_burst,
                )
    return _limiter


def reset_rate_limiter() -> None:
    """Drop the shared limiter. Used by tests and the app's shutdown hook."""
    global _limiter
    with _limiter_lock:
        _limiter = None


#: Declared so the key field appears in the OpenAPI document and in /docs.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _match(presented: str, accepted: list[str]) -> bool:
    """Constant-time membership test.

    ``in`` on a list short-circuits on the first differing byte, which leaks
    key material through response timing. Every candidate is compared.
    """
    matched = False
    for candidate in accepted:
        if hmac.compare_digest(presented, candidate):
            matched = True
    return matched


def authenticate(
    request: Request,
    presented: str | None = Depends(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> Caller:
    """Resolve the caller, rejecting bad keys and exhausted budgets.

    Attached to the ``/v1`` routers as a router-level dependency so a new route
    cannot be added unprotected by forgetting to decorate it.
    """
    accepted = settings.security.api_keys
    if accepted:
        header = request.headers.get(settings.security.header_name) or presented
        if not header:
            raise UnauthorizedError(
                f"missing {settings.security.header_name} header",
                header=settings.security.header_name,
            )
        if not _match(header, accepted):
            raise UnauthorizedError("the supplied API key is not recognised")
        # Keys are secrets: identify the caller by a short fingerprint so logs
        # and rate-limit buckets never carry the key itself.
        caller = Caller(identity=f"key:{header[:4]}…{len(header)}", authenticated=True)
    else:
        client = request.client.host if request.client else ANONYMOUS
        caller = Caller(identity=f"ip:{client}", authenticated=False)

    if settings.security.rate_limit_enabled:
        allowed, wait = get_rate_limiter(settings).take(caller.identity)
        if not allowed:
            raise RateLimitedError(
                "rate limit exceeded",
                retry_after_seconds=round(wait, 2),
                limit_per_minute=settings.security.rate_limit_per_minute,
            )
    return caller


def authenticate_optional(
    request: Request, settings: Settings = Depends(get_settings)
) -> Caller | None:
    """Resolve the caller without rejecting anyone, and without spending budget.

    For endpoints that must stay reachable by an orchestrator's health check but
    should still reveal more to an authenticated operator than to the internet.
    """
    accepted = settings.security.api_keys
    if not accepted:
        return Caller(identity=ANONYMOUS, authenticated=False)
    header = request.headers.get(settings.security.header_name)
    if header and _match(header, accepted):
        return Caller(identity=f"key:{header[:4]}\u2026{len(header)}", authenticated=True)
    return None
