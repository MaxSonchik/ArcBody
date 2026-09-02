"""The encoder: crop framing, the margin head, fusion and comparison."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from arcbody.config import EmbeddingSettings
from arcbody.embed.arcface import ArcMarginProduct
from arcbody.embed.crop import build_tensor_input, canonical_box
from arcbody.embed.encoder import BodyEncoder, cosine_similarity, fuse, view_agreement
from arcbody.embed.model import ArcBodyNet
from arcbody.measure.schema import NUM_RATIOS
from arcbody.training.synthetic import PoseParams, render, sample_appearance


def test_the_crop_frame_has_a_fixed_aspect_ratio(capture, perception) -> None:
    observation = perception.analyse(capture.image).subject
    box = canonical_box(observation, 2.0)
    assert box.height / box.width == pytest.approx(2.0, rel=1e-6)


def test_crop_scale_does_not_follow_the_pose(subject, perception, rng) -> None:
    """Regression: framing used to widen with the arms, shrinking the body.

    A pose-dependent crop makes the encoder learn to undo a scale change that
    carries no information about whose body it is.
    """
    coverage = []
    for angle in (22.0, 38.0, 55.0):
        capture = render(
            subject,
            PoseParams(arm_angle_deg=angle, subject_height_ratio=0.85),
            sample_appearance(rng),
            size=(384, 512),
            rng=rng,
        )
        observation = perception.analyse(capture.image).subject
        tensor = build_tensor_input(
            capture.image, observation, input_width=128, input_height=256
        )
        coverage.append(float(tensor[3].mean()))
    assert max(coverage) - min(coverage) < 0.10


def test_the_crop_carries_the_silhouette_as_a_fourth_channel(capture, perception) -> None:
    observation = perception.analyse(capture.image).subject
    tensor = build_tensor_input(
        capture.image, observation, input_width=128, input_height=256
    )
    assert tensor.shape == (4, 256, 128)
    assert tensor[3].min() >= 0.0 and tensor[3].max() <= 1.0
    assert 0.05 < tensor[3].mean() < 0.6


def test_a_missing_mask_is_zeroed_not_filled(capture, perception) -> None:
    observation = perception.analyse(capture.image).subject
    observation.mask = None
    tensor = build_tensor_input(
        capture.image, observation, input_width=128, input_height=256
    )
    assert tensor[3].sum() == 0.0


def test_the_margin_always_penalises_the_target_class() -> None:
    torch.manual_seed(0)
    head = ArcMarginProduct(16, 5, scale=30.0, margin=0.3)
    embeddings = torch.randn(4, 16)
    labels = torch.tensor([0, 1, 2, 3])
    logits = head(embeddings, labels)
    cosine = head.cosine(embeddings)
    rows = torch.arange(4)
    assert bool((logits[rows, labels] < head.scale * cosine[rows, labels]).all())
    # Non-target logits are the plain scaled cosine.
    assert torch.allclose(logits[0, 1], head.scale * cosine[0, 1])


def test_the_margin_stays_finite_at_extreme_angles() -> None:
    head = ArcMarginProduct(8, 3, scale=64.0, margin=0.5)
    extreme = torch.nn.functional.normalize(torch.randn(2, 8)) * 1e4
    assert bool(torch.isfinite(head(extreme, torch.tensor([0, 1]))).all())


def test_set_margin_keeps_derived_constants_in_step() -> None:
    head = ArcMarginProduct(8, 3, margin=0.3)
    head.set_margin(0.1)
    assert head.margin == pytest.approx(0.1)
    assert head._cos_m == pytest.approx(math.cos(0.1))
    assert head._threshold == pytest.approx(math.cos(math.pi - 0.1))


def test_sub_centres_widen_the_weight_without_changing_the_output_shape() -> None:
    head = ArcMarginProduct(16, 5, sub_centers=3)
    assert head.weight.shape == (15, 16)
    assert head.cosine(torch.randn(2, 16)).shape == (2, 5)


def test_the_network_emits_a_unit_embedding_and_a_ratio_vector() -> None:
    net = ArcBodyNet(64)
    output = net.encode(torch.randn(2, 4, 256, 128))
    assert output.embedding.shape == (2, 64)
    assert output.ratios.shape == (2, NUM_RATIOS)
    assert torch.allclose(output.embedding.norm(dim=1), torch.ones(2), atol=1e-5)


def test_the_encoder_is_reproducible_without_a_checkpoint(capture, perception) -> None:
    """An untrained trunk must at least return the same vector every restart."""
    observation = perception.analyse(capture.image).subject
    settings = EmbeddingSettings(dim=64)
    first = BodyEncoder(settings).encode(capture.image, observation)
    second = BodyEncoder(settings).encode(capture.image, observation)
    assert not first.trained
    assert np.allclose(first.embedding, second.embedding)


def test_fusion_returns_a_unit_vector_and_reports_disagreement() -> None:
    base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    near = np.array([0.99, 0.14, 0.0], dtype=np.float32)
    opposite = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    assert np.linalg.norm(fuse([base, near])) == pytest.approx(1.0, abs=1e-5)
    assert view_agreement([base, near]) > 0.98
    assert view_agreement([base, opposite]) < 0.05
    assert view_agreement([base]) == 1.0
    with pytest.raises(ValueError):
        fuse([])


def test_cosine_similarity_handles_degenerate_input() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0
