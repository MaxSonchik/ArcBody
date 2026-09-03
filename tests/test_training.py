"""Data generation, losses and evaluation metrics.

These guard the parts that make a training run trustworthy rather than the run
itself: that the generator's ground truth is self-consistent, that the sampler
produces batches a triplet loss can use, and that the metrics say what they
claim.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from arcbody.measure.schema import NUM_RATIOS, RATIO_NAMES
from arcbody.measure.silhouette import SilhouetteProfile
from arcbody.training.datasets import SyntheticBodyDataset, collate
from arcbody.training.losses import PKSampler, batch_hard_triplet_loss
from arcbody.training.metrics import (
    identification_metrics,
    ratio_error,
    verification_metrics,
)
from arcbody.training.rasterise import capsule, ellipse, profile_column
from arcbody.training.synthetic import render, sample_appearance, sample_pose
from arcbody.types import ViewLabel

# -- the generator ----------------------------------------------------------


def test_rendered_widths_match_the_parameters_they_were_built_from(subject, capture) -> None:
    """The generator's ground truth must be true, or every accuracy test lies."""
    profile = SilhouetteProfile(capture.mask)
    stature = profile.stature_px
    centre = profile.centre_x()
    for level, name in (
        (subject.waist_level, "waist_breadth"),
        (subject.neck_level, "neck_breadth"),
    ):
        measured = profile.torso_width(profile.row_of(level), centre) / stature
        expected = getattr(subject, name)
        assert abs(measured - expected) / expected < 0.06, name


def test_stature_in_the_mask_matches_the_stature_rendered(capture) -> None:
    profile = SilhouetteProfile(capture.mask)
    assert profile.stature_px == pytest.approx(capture.stature_px, rel=0.02)


def test_ground_truth_girths_use_the_subject_s_own_depth(subject) -> None:
    from arcbody.measure.schema import girth_from_breadth_depth

    truth = subject.measurements_cm()
    breadth = subject.waist_breadth * subject.stature_cm
    assert truth["waist_girth"] == pytest.approx(
        girth_from_breadth_depth(breadth, breadth * subject.waist_depth_ratio)
    )


def test_the_ratio_vector_is_ordered_canonically(subject) -> None:
    vector = subject.ratio_vector()
    assert vector.shape == (NUM_RATIOS,)
    ratios = subject.ratios()
    assert list(ratios) == list(RATIO_NAMES)
    assert vector[0] == pytest.approx(ratios[RATIO_NAMES[0]])


def test_a_side_render_is_narrower_than_a_front_render(subject, rng) -> None:
    pose = sample_pose(rng, easy=True)
    appearance = sample_appearance(rng)
    front = render(subject, pose, appearance, size=(384, 512), view=ViewLabel.FRONT, rng=rng)
    side = render(subject, pose, appearance, size=(384, 512), view=ViewLabel.SIDE, rng=rng)
    assert side.mask.sum() < front.mask.sum()


def test_the_hand_is_rendered(subject, rng) -> None:
    """Regression: without hands, a tracer that stops at the fingertips looked
    accurate here and was a hand too long on a photograph."""
    from arcbody import keypoints as kp

    capture = render(
        subject, sample_pose(rng, easy=True), sample_appearance(rng), size=(384, 512), rng=rng
    )
    wrist = capture.keypoints[kp.LEFT_WRIST, :2]
    profile = SilhouetteProfile(capture.mask)
    # Foreground must continue past the wrist, out along the arm.
    assert capture.mask[int(wrist[1]) : int(wrist[1] + 0.05 * profile.stature_px)].any()


# -- the rasteriser ---------------------------------------------------------


def test_a_capsule_tapers_between_its_radii() -> None:
    mask = np.zeros((200, 200), bool)
    capsule(mask, (50, 50), (50, 150), 10, 3)
    assert mask[50].sum() == pytest.approx(20, abs=2)
    assert mask[150].sum() == pytest.approx(6, abs=2)


def test_an_ellipse_has_the_expected_area() -> None:
    mask = np.zeros((100, 100), bool)
    ellipse(mask, (50, 50), 20, 10)
    assert mask.sum() == pytest.approx(np.pi * 20 * 10, rel=0.02)


def test_a_profile_column_interpolates_between_levels() -> None:
    mask = np.zeros((100, 100), bool)
    profile_column(mask, 50, [(10, 5), (50, 20), (90, 5)])
    assert mask[10].sum() == pytest.approx(11, abs=2)
    assert mask[50].sum() == pytest.approx(41, abs=2)
    assert mask[30].sum() > mask[10].sum()


# -- the dataset ------------------------------------------------------------


def test_the_dataset_is_deterministic() -> None:
    first = SyntheticBodyDataset(identities=4, per_identity=3, seed=0)
    second = SyntheticBodyDataset(identities=4, per_identity=3, seed=0)
    assert torch.equal(first[5].tensor, second[5].tensor)


def test_captures_of_one_identity_share_its_ratios_but_not_its_pixels() -> None:
    dataset = SyntheticBodyDataset(identities=4, per_identity=3, seed=0)
    first, second = dataset[0], dataset[1]
    assert first.identity == second.identity
    assert torch.allclose(first.ratios, second.ratios)
    assert not torch.equal(first.tensor, second.tensor)


def test_collate_stacks_a_batch() -> None:
    dataset = SyntheticBodyDataset(identities=4, per_identity=3, seed=0)
    batch = collate([dataset[i] for i in range(6)])
    assert batch["tensor"].shape == (6, 4, 256, 128)
    assert batch["ratios"].shape == (6, NUM_RATIOS)
    assert batch["ratio_mask"].shape == (6, NUM_RATIOS)


# -- losses -----------------------------------------------------------------


def test_the_sampler_fills_every_batch_with_positive_pairs() -> None:
    labels = [index // 4 for index in range(80)]
    sampler = PKSampler(labels, identities_per_batch=3, instances=4, seed=0)
    from collections import Counter

    for batch in sampler:
        counts = Counter(labels[index] for index in batch)
        assert len(counts) == 3
        assert set(counts.values()) == {4}
        break


def test_the_sampler_refuses_an_impossible_request() -> None:
    with pytest.raises(ValueError, match="identities"):
        PKSampler([0, 0, 1, 1], identities_per_batch=8, instances=2)


def test_the_sampler_reshuffles_between_epochs() -> None:
    labels = [index // 4 for index in range(400)]
    sampler = PKSampler(labels, identities_per_batch=3, instances=4, seed=0)
    sampler.set_epoch(0)
    first = next(iter(sampler))
    sampler.set_epoch(1)
    assert first != next(iter(sampler))


def test_triplet_loss_rewards_separated_clusters() -> None:
    torch.manual_seed(0)
    centres = torch.nn.functional.normalize(torch.randn(3, 16))
    tight = torch.cat([centres[i].repeat(4, 1) + 0.01 * torch.randn(4, 16) for i in range(3)])
    labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    assert float(batch_hard_triplet_loss(tight, labels, margin=0.3)) == pytest.approx(0.0)
    scrambled = labels[torch.randperm(12)]
    assert float(batch_hard_triplet_loss(tight, scrambled, margin=0.3)) > 0.5


def test_triplet_loss_is_zero_without_a_negative() -> None:
    embeddings = torch.randn(6, 8)
    assert float(batch_hard_triplet_loss(embeddings, torch.zeros(6, dtype=torch.long))) == 0.0


# -- metrics ----------------------------------------------------------------


@pytest.fixture
def clustered(rng):
    centres = rng.normal(size=(8, 16))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    embeddings, labels = [], []
    for index, centre in enumerate(centres):
        for _ in range(5):
            vector = centre + rng.normal(0, 0.12, 16)
            embeddings.append(vector / np.linalg.norm(vector))
            labels.append(index)
    return np.array(embeddings), np.array(labels)


def test_verification_separates_clusters(clustered) -> None:
    embeddings, labels = clustered
    metrics = verification_metrics(embeddings, labels)
    assert metrics.auc > 0.95
    assert metrics.eer < 0.1
    assert metrics.tar_at_far["far_0.01"] > 0.8


def test_verification_of_noise_is_chance(clustered, rng) -> None:
    _, labels = clustered
    noise = rng.normal(size=(len(labels), 16))
    assert 0.35 < verification_metrics(noise, labels).auc < 0.65


def test_identification_ranks_perfectly_on_clean_clusters(clustered) -> None:
    embeddings, labels = clustered
    gallery_index = np.arange(0, len(labels), 5)
    query_index = np.setdiff1d(np.arange(len(labels)), gallery_index)
    metrics = identification_metrics(
        embeddings[gallery_index],
        labels[gallery_index],
        embeddings[query_index],
        labels[query_index],
    )
    assert metrics.rank1 == 1.0
    assert metrics.mean_average_precision == 1.0


def test_ratio_error_is_a_masked_percentage() -> None:
    predicted = np.array([[1.1, 5.0]])
    target = np.array([[1.0, 2.0]])
    assert ratio_error(predicted, target, np.array([[1, 0]])) == pytest.approx(10.0)
    assert np.isnan(ratio_error(predicted, target, np.array([[0, 0]])))
