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


# -- the search index -------------------------------------------------------
#
# The index is maintained incrementally. Rebuilding it on every write was
# correct but slow enough to matter: a query following an enrolment cost 190 ms
# against 0.22 ms in steady state, and enrolment-heavy traffic is exactly what a
# batch job produces. These tests pin the behaviour the optimisation must keep.


def test_a_newly_enrolled_person_is_findable_immediately(gallery, rng) -> None:
    query = unit(rng.normal(size=32))
    gallery.enrol("first", query)
    assert [m.person_id for m in gallery.identify(query, top_k=1)] == ["first"]

    later = unit(rng.normal(size=32))
    gallery.enrol("second", later)
    assert [m.person_id for m in gallery.identify(later, top_k=1)] == ["second"]
    assert len(gallery.identify(query, top_k=5)) == 2


def test_a_second_profile_updates_the_existing_row(gallery, rng) -> None:
    """Enrolment is additive, so the fused signature must move, not duplicate."""
    base = unit(rng.normal(size=32))
    gallery.enrol("alice", base)
    before = gallery.identify(base, top_k=5)
    assert len(before) == 1 and before[0].profile_count == 1

    gallery.enrol("alice", unit(base + rng.normal(0, 0.3, 32)))
    after = gallery.identify(base, top_k=5)
    assert len(after) == 1, "a second profile must not create a second index row"
    assert after[0].profile_count == 2


def test_deleting_the_middle_of_the_index_keeps_the_rest_findable(gallery, rng) -> None:
    """Deletion swaps the last row into the gap; the mapping must follow."""
    vectors = {name: unit(rng.normal(size=32)) for name in ("a", "b", "c", "d")}
    for name, vector in vectors.items():
        gallery.enrol(name, vector)
    gallery.identify(vectors["a"], top_k=4)  # force the index to exist

    gallery.forget("b")
    for name in ("a", "c", "d"):
        top = gallery.identify(vectors[name], top_k=1)
        assert top and top[0].person_id == name, f"{name} became unfindable"
    assert "b" not in [m.person_id for m in gallery.identify(vectors["b"], top_k=4)]


def test_enrolling_beyond_the_initial_capacity_still_works(gallery, rng) -> None:
    """The backing array grows geometrically; every person stays searchable."""
    vectors = {}
    for index in range(120):
        name = f"p{index:03d}"
        vectors[name] = unit(rng.normal(size=32))
        gallery.enrol(name, vectors[name])
        if index == 0:
            gallery.identify(vectors[name], top_k=1)  # build the index early
    assert gallery.count()[0] == 120
    for name in ("p000", "p060", "p119"):
        assert gallery.identify(vectors[name], top_k=1)[0].person_id == name


def test_a_changed_embedding_dimension_rebuilds_rather_than_corrupts(
    gallery, rng
) -> None:
    """A new encoder checkpoint changes the width; the metric must not silently mix."""
    gallery.enrol("old", unit(rng.normal(size=32)))
    gallery.identify(unit(rng.normal(size=32)), top_k=1)

    wide = unit(rng.normal(size=64))
    gallery.enrol("new", wide)
    # Querying at the new width finds the new person and ignores the old rows,
    # rather than returning a score computed across mismatched dimensions.
    matches = gallery.identify(wide, top_k=5)
    assert all(match.person_id == "new" for match in matches)


def test_face_links_survive_an_incremental_update(gallery, rng) -> None:
    vector = unit(rng.normal(size=32))
    gallery.enrol("alice", vector, external_face_id="face-1")
    assert gallery.identify(vector, top_k=1)[0].external_face_id == "face-1"
    gallery.enrol("alice", unit(vector + rng.normal(0, 0.05, 32)))
    assert gallery.identify(vector, top_k=1)[0].external_face_id == "face-1"
