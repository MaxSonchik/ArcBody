"""Domain errors.

Every error carries a stable ``code`` so the HTTP layer can map it to a status
code without string-matching messages, and so clients can branch on it.
"""

from __future__ import annotations


class ArcBodyError(Exception):
    """Base class for every error ArcBody raises deliberately."""

    code = "arcbody_error"
    http_status = 500

    def __init__(self, message: str, **details: object) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "details": self.details}


class InvalidImageError(ArcBodyError):
    """The payload could not be decoded as an image, or is unusably small."""

    code = "invalid_image"
    http_status = 422


class NoPersonFoundError(ArcBodyError):
    """No person was detected in the image."""

    code = "no_person_found"
    http_status = 422


class AmbiguousSubjectError(ArcBodyError):
    """Several people were detected and none is clearly the subject."""

    code = "ambiguous_subject"
    http_status = 422


class InsufficientQualityError(ArcBodyError):
    """The subject was found but fails the gates required for measurement."""

    code = "insufficient_quality"
    http_status = 422


class BackendUnavailableError(ArcBodyError):
    """A perception or encoder backend was requested but its deps/weights are missing."""

    code = "backend_unavailable"
    http_status = 503


class PersonNotFoundError(ArcBodyError):
    """The requested person has no enrolled body profile."""

    code = "person_not_found"
    http_status = 404


class JobNotFoundError(ArcBodyError):
    """The requested batch job id is unknown."""

    code = "job_not_found"
    http_status = 404
