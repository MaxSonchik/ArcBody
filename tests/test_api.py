"""The HTTP surface, end to end.

These run the real pipeline over rendered subjects — no mocks — because the
contract worth testing is "a photo goes in and a usable body profile comes out",
and a mocked pipeline would assert only that the routing works.
"""

from __future__ import annotations

import base64
import time

import numpy as np
import pytest

from arcbody import imaging
from arcbody.training.synthetic import render, sample_appearance, sample_params, sample_pose
from arcbody.types import ViewLabel


@pytest.fixture
def photos(rng):
    """A subject and a factory for base64 photos of them."""
    params = sample_params(rng)

    def shoot(view: ViewLabel = ViewLabel.FRONT) -> str:
        capture = render(
            params,
            sample_pose(rng, easy=True),
            sample_appearance(rng),
            size=(384, 512),
            view=view,
            rng=rng,
        )
        return base64.b64encode(imaging.encode_png(capture.image)).decode()

    return params, shoot


@pytest.fixture
def body(photos):
    params, shoot = photos
    return {
        "images": [
            {"content_base64": shoot(), "view": "front"},
            {"content_base64": shoot(ViewLabel.SIDE), "view": "side"},
        ],
        "stature_cm": float(params.stature_cm),
        "weight_kg": 72.0,
    }


def test_health_reports_its_own_caveats(client) -> None:
    payload = client.get("/healthz").json()
    assert payload["perception_backend"] == "classic"
    # Running untrained or on the fallback backend is "up but not fully useful",
    # and an operator should learn that here rather than from a user complaint.
    assert payload["status"] == "degraded"
    assert payload["warnings"]


def test_analyze_returns_measurements_prompt_and_maps(client, body) -> None:
    response = client.post("/v1/analyze", json=body)
    assert response.status_code == 200
    payload = response.json()

    waist = payload["measurements"]["values"]["waist_girth"]
    assert waist["ci_low_cm"] <= waist["value_cm"] <= waist["ci_high_cm"]
    assert waist["source"] == "silhouette", "a side view must replace the depth prior"
    assert payload["prompt"]["prompt"]
    assert payload["prompt"]["negative_prompt"]
    assert set(payload["control_maps"]) >= {"pose", "silhouette", "normalised_crop"}
    assert payload["embedding_dim"] == len(payload["embedding"])
    assert payload["quality"]["passed"]


def test_a_single_view_girth_declares_its_prior(client, photos) -> None:
    _, shoot = photos
    response = client.post(
        "/v1/analyze",
        json={"images": [{"content_base64": shoot(), "view": "front"}], "stature_cm": 175.0},
    )
    assert response.json()["measurements"]["values"]["waist_girth"]["source"] == "ellipse_prior"


def test_omitting_stature_yields_proportions_only(client, photos) -> None:
    _, shoot = photos
    payload = client.post(
        "/v1/analyze",
        json={"images": [{"content_base64": shoot()}], "include_control_maps": False},
    ).json()
    assert payload["measurements"]["ratios"]
    assert payload["measurements"]["values"] == {}


def test_the_embedding_can_be_suppressed(client, body) -> None:
    payload = client.post(
        "/v1/analyze", json={**body, "include_embedding": False, "include_control_maps": False}
    ).json()
    assert payload["embedding"] is None
    assert payload["embedding_dim"] > 0


def test_enrol_identify_and_delete(client, body) -> None:
    created = client.post("/v1/persons/subject-1/enroll", json={**body, "external_face_id": "f-1"})
    assert created.status_code == 201
    assert created.json()["profile_count"] == 1
    assert created.json()["external_face_id"] == "f-1"

    found = client.post("/v1/identify", json={**body, "top_k": 3, "include_control_maps": False})
    assert found.status_code == 200
    assert found.json()["matches"][0]["person_id"] == "subject-1"

    listed = client.get("/v1/persons").json()
    assert listed["total_persons"] == 1 and listed["total_profiles"] == 1

    assert client.delete("/v1/persons/subject-1").status_code == 204
    assert client.get("/v1/persons/subject-1").status_code == 404
    assert client.delete("/v1/persons/subject-1").status_code == 404


def test_generation_check_scores_against_the_enrolled_body(client, body) -> None:
    client.post("/v1/persons/subject-1/enroll", json=body)
    payload = client.post(
        "/v1/persons/subject-1/check-generation",
        json={**body, "include_control_maps": False, "include_embedding": False},
    ).json()
    assert payload["verdict"] in {"consistent", "drifted", "different_body"}
    assert -1.0 <= payload["similarity"] <= 1.0
    assert any("untrained" in note for note in payload["notes"]), (
        "an untrained encoder must disclaim its own similarity score"
    )


def test_generation_check_needs_an_enrolled_person(client, body) -> None:
    assert client.post("/v1/persons/ghost/check-generation", json=body).status_code == 404


def test_multipart_upload_works(client, photos) -> None:
    _, shoot = photos
    image = base64.b64decode(shoot())
    response = client.post(
        "/v1/analyze/upload",
        files={"files": ("front.png", image, "image/png")},
        data={"stature_cm": "175", "include_control_maps": "false"},
    )
    assert response.status_code == 200
    assert response.json()["measurements"]["values"]["stature"]["value_cm"] == 175.0


def test_errors_carry_a_stable_code(client, body, photos) -> None:
    _, shoot = photos
    bad = client.post("/v1/analyze", json={"images": [{"content_base64": "bm90"}]})
    assert bad.status_code == 422 and bad.json()["code"] == "invalid_image"

    blank = base64.b64encode(
        imaging.encode_png(np.full((400, 300, 3), 205, np.uint8))
    ).decode()
    empty = client.post("/v1/analyze", json={"images": [{"content_base64": blank}]})
    assert empty.status_code == 422 and empty.json()["code"] == "no_person_found"

    assert client.get("/v1/persons/nobody").json()["code"] == "person_not_found"
    assert client.post("/v1/analyze", json={**body, "stature_cm": 300}).status_code == 422


def test_batch_isolates_a_failing_item(client, photos) -> None:
    _, shoot = photos
    blank = base64.b64encode(
        imaging.encode_png(np.full((400, 300, 3), 205, np.uint8))
    ).decode()
    submitted = client.post(
        "/v1/batch",
        json={
            "items": [
                {
                    "reference": "good",
                    "images": [{"content_base64": shoot(), "view": "front"}],
                    "stature_cm": 175.0,
                    "person_id": "batch-1",
                },
                {"reference": "blank", "images": [{"content_base64": blank}]},
            ]
        },
    )
    assert submitted.status_code == 202
    job_id = submitted.json()["job_id"]

    for _ in range(200):
        job = client.get(f"/v1/batch/{job_id}").json()
        if job["status"] == "completed":
            break
        time.sleep(0.1)
    assert job["status"] == "completed"
    assert job["completed"] == 1 and job["failed"] == 1

    results = {item["reference"]: item for item in job["results"]}
    assert results["good"]["person_id"] == "batch-1"
    assert results["blank"]["error"]["code"] == "no_person_found"
    assert client.get("/v1/persons").json()["total_persons"] == 1


def test_a_malformed_batch_fails_at_submission(client) -> None:
    """Better to reject the job than to accept it and fail one item deep."""
    response = client.post(
        "/v1/batch", json={"items": [{"reference": "x", "images": [{"content_base64": "bm90"}]}]}
    )
    assert response.status_code == 422


def test_unknown_job_is_a_404(client) -> None:
    assert client.get("/v1/batch/deadbeef").status_code == 404


def test_openapi_documents_every_route(client) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for path in (
        "/healthz",
        "/v1/analyze",
        "/v1/analyze/upload",
        "/v1/identify",
        "/v1/batch",
        "/v1/persons/{person_id}/enroll",
        "/v1/persons/{person_id}/check-generation",
    ):
        assert path in paths, f"{path} is missing from the OpenAPI document"
