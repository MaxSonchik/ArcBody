"""Does a trained checkpoint actually tell bodies apart?

Skipped when no checkpoint is present, because the rest of the suite must run
on a fresh clone. When one *is* present this is the test that matters most: an
encoder that returns well-formed vectors carrying no identity information would
pass every other test in this repository.

It measures through the service path — segmentation first, then encoding — on
subjects the model has never seen, because that is what the API does. The
training loop's own validation feeds the network perfect masks from the
renderer and therefore flatters it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arcbody.config import EmbeddingSettings, PerceptionSettings
from arcbody.embed.encoder import BodyEncoder, cosine_similarity
from arcbody.perception.classic import ClassicPerception
from arcbody.training.metrics import identification_metrics, verification_metrics
from arcbody.training.synthetic import render, sample_appearance, sample_params, sample_pose
from arcbody.types import ViewLabel

CHECKPOINT = Path("weights/arcbody.pt")

#: Floors, not targets. An untrained trunk scores about 0.50 AUC on this cohort,
#: so anything meaningfully above chance proves the training signal reached the
#: embedding. Raise these as the model improves; they exist to catch a
#: regression to noise, not to certify production quality.
MIN_AUC = 0.60
MIN_RANK1 = 0.10

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason=f"no encoder checkpoint at {CHECKPOINT}; run 'make train' first",
)


@pytest.fixture(scope="module")
def cohort():
    """Embed unseen subjects through segmentation, as the API would."""
    perception = ClassicPerception(PerceptionSettings())
    encoder = BodyEncoder(EmbeddingSettings(weights_path=CHECKPOINT))
    rng = np.random.default_rng(90210)

    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    for identity in range(24):
        params = sample_params(rng)
        for index in range(4):
            view = ViewLabel.SIDE if index == 3 else ViewLabel.FRONT
            capture = render(
                params,
                sample_pose(rng, easy=True),
                sample_appearance(rng),
                size=(384, 512),
                view=view,
                rng=rng,
            )
            observation = perception.analyse(capture.image).subject
            if observation is None:
                continue
            embeddings.append(encoder.encode(capture.image, observation).embedding)
            labels.append(identity)
    return np.stack(embeddings), np.array(labels), encoder


def test_the_checkpoint_is_recognised_as_trained(cohort) -> None:
    _, _, encoder = cohort
    assert encoder.trained, "a checkpoint on disk must set the trained flag"


def test_embeddings_separate_unseen_bodies(cohort) -> None:
    embeddings, labels, _ = cohort
    metrics = verification_metrics(embeddings, labels)
    assert metrics.auc >= MIN_AUC, (
        f"held-out verification AUC {metrics.auc:.3f} is at or near chance; "
        "the identity signal is not reaching the embedding"
    )


def test_the_right_person_ranks_first(cohort) -> None:
    embeddings, labels, _ = cohort
    first_seen: dict[int, int] = {}
    for index, label in enumerate(labels):
        first_seen.setdefault(int(label), index)
    gallery = np.array(sorted(first_seen.values()))
    queries = np.setdiff1d(np.arange(len(labels)), gallery)
    metrics = identification_metrics(
        embeddings[gallery], labels[gallery], embeddings[queries], labels[queries]
    )
    assert metrics.rank1 >= MIN_RANK1


def test_same_body_scores_above_different_bodies(cohort) -> None:
    """The property the /check-generation verdict rests on."""
    embeddings, labels, _ = cohort
    same, different = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            score = cosine_similarity(embeddings[i], embeddings[j])
            (same if labels[i] == labels[j] else different).append(score)
    assert np.mean(same) > np.mean(different), (
        f"mean same-body similarity {np.mean(same):.3f} does not exceed "
        f"mean different-body similarity {np.mean(different):.3f}"
    )


def test_the_encoder_honours_the_checkpoint_crop_size() -> None:
    """Serving a different crop size than was trained is a silent quality loss."""
    encoder = BodyEncoder(
        EmbeddingSettings(weights_path=CHECKPOINT, input_width=64, input_height=128)
    )
    import torch

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert encoder.input_width == payload["input_width"]
    assert encoder.input_height == payload["input_height"]
