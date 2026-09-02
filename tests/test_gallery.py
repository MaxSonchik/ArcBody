"""The enrolled-profile store."""

from __future__ import annotations

import numpy as np
import pytest

from arcbody.config import GallerySettings
from arcbody.errors import PersonNotFoundError
from arcbody.gallery.store import Gallery


@pytest.fixture
def gallery(tmp_path):
    store = Gallery(GallerySettings(database_path=tmp_path / "g.sqlite3"))
    yield store
    store.close()


def unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


@pytest.fixture
def populated(gallery, rng):
    centres = {name: unit(rng.normal(size=32)) for name in ("alice", "bob", "carol")}
    for name, centre in centres.items():
        for _ in range(2):
            gallery.enrol(
                name,
                unit(centre + rng.normal(0, 0.05, 32)),
                ratios={"waist_to_hip": 0.8},
                quality=0.9,
                external_face_id=f"face-{name}",
            )
    return gallery, centres


def test_enrolment_is_additive(populated) -> None:
    gallery, _ = populated
    assert gallery.count() == (3, 6)
    assert gallery.get("alice").profile_count == 2

    gallery.enrol("alice", unit(np.ones(32)))
    assert gallery.get("alice").profile_count == 3
    assert gallery.count()[0] == 3, "a second profile must not create a second person"


def test_the_fused_signature_is_a_unit_vector(populated) -> None:
    gallery, _ = populated
    assert np.linalg.norm(gallery.get("alice").embedding) == pytest.approx(1.0, abs=1e-5)


def test_identification_ranks_the_right_person_first(populated, rng) -> None:
    gallery, centres = populated
    query = unit(centres["bob"] + rng.normal(0, 0.05, 32))
    matches = gallery.identify(query, top_k=3)
    assert [match.person_id for match in matches][0] == "bob"
    assert matches[0].similarity > 0.85
    assert matches[0].external_face_id == "face-bob"
    assert matches[0].similarity >= matches[1].similarity


def test_identification_respects_a_minimum(populated, rng) -> None:
    gallery, centres = populated
    query = unit(centres["bob"] + rng.normal(0, 0.05, 32))
    assert gallery.identify(query, top_k=3, minimum=0.99) == []


def test_verification_scores_one_person(populated, rng) -> None:
    gallery, centres = populated
    query = unit(centres["bob"] + rng.normal(0, 0.05, 32))
    assert gallery.verify("bob", query) > 0.85
    assert gallery.verify("alice", query) < gallery.verify("bob", query)


def test_a_changed_embedding_dimension_is_an_error_not_a_wrong_score(populated) -> None:
    gallery, _ = populated
    with pytest.raises(ValueError, match="dimension"):
        gallery.verify("bob", np.zeros(8, dtype=np.float32))


def test_forgetting_is_a_real_delete(populated, rng) -> None:
    """Body embeddings are biometric data; erasure must actually erase."""
    gallery, centres = populated
    assert gallery.forget("bob") == 1
    assert gallery.count() == (2, 4)
    with pytest.raises(PersonNotFoundError):
        gallery.get("bob")
    query = unit(centres["bob"] + rng.normal(0, 0.05, 32))
    assert "bob" not in [match.person_id for match in gallery.identify(query, top_k=3)]
    assert gallery.forget("bob") == 0


def test_unknown_person_raises(gallery) -> None:
    with pytest.raises(PersonNotFoundError):
        gallery.get("nobody")


def test_empty_gallery_identifies_nothing(gallery) -> None:
    assert gallery.identify(unit(np.ones(32))) == []


def test_a_degenerate_query_returns_nothing(populated) -> None:
    gallery, _ = populated
    assert gallery.identify(np.zeros(32, dtype=np.float32)) == []


def test_an_empty_embedding_cannot_be_enrolled(gallery) -> None:
    with pytest.raises(ValueError):
        gallery.enrol("x", np.zeros(0, dtype=np.float32))
