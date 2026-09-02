"""Losses and the sampler that makes one of them possible.

The angular-margin head alone converges slowly here, and the reason is
structural rather than a bug: it is a 900-way classifier that sees each identity
about four times per epoch, so each class centre gets four gradient steps an
epoch and takes a very long time to find its place. The auxiliary shape head
learns quickly from the same trunk, which is how we know the features carry body
shape and the problem is the head's sample efficiency.

The standard person re-ID remedy is to pair the identity loss with a batch-hard
triplet loss over batches built as P identities x K instances. Every batch then
supplies O(P*K^2) direct comparisons instead of 48 classification targets, and
the two losses shape complementary things — the classifier organises the space
globally, the triplet term sharpens local neighbourhoods. The BNNeck exists
precisely to let them coexist: triplet on the features before it, classification
on the features after.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
import torch.nn.functional as F


class PKSampler(torch.utils.data.Sampler[list[int]]):
    """Yields batches of ``identities_per_batch`` people, ``instances`` shots each.

    A randomly shuffled batch of 48 drawn from 900 identities with 4 captures
    each almost never contains a matching pair, which leaves a triplet loss with
    nothing to work on. This sampler guarantees every batch is full of them.
    """

    def __init__(
        self,
        labels: list[int],
        *,
        identities_per_batch: int,
        instances: int,
        batches: int | None = None,
        seed: int = 0,
    ) -> None:
        self.identities_per_batch = identities_per_batch
        self.instances = instances
        self.seed = seed
        self.epoch = 0

        self.by_identity: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            self.by_identity.setdefault(int(label), []).append(index)
        self.identities = sorted(self.by_identity)
        if len(self.identities) < identities_per_batch:
            raise ValueError(
                f"need at least {identities_per_batch} identities, got {len(self.identities)}"
            )
        self.batches = batches or max(1, len(labels) // (identities_per_batch * instances))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng((self.seed, self.epoch))
        for _ in range(self.batches):
            chosen = rng.choice(
                len(self.identities), size=self.identities_per_batch, replace=False
            )
            batch: list[int] = []
            for position in chosen:
                pool = self.by_identity[self.identities[int(position)]]
                # Sample with replacement when an identity has fewer captures
                # than K, so a thin class still contributes a positive pair.
                replace = len(pool) < self.instances
                batch.extend(
                    int(index)
                    for index in rng.choice(pool, size=self.instances, replace=replace)
                )
            yield batch


def batch_hard_triplet_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3
) -> torch.Tensor:
    """Soft-margin triplet loss over the hardest pair for each anchor.

    For every sample, the *furthest* same-identity neighbour and the *nearest*
    different-identity one are selected within the batch. Mining the hardest
    pairs rather than averaging over all of them is what makes the term keep
    contributing once the easy triplets are satisfied.
    """
    if embeddings.size(0) < 2:
        return embeddings.sum() * 0.0

    normalised = F.normalize(embeddings, dim=1)
    distances = torch.cdist(normalised, normalised, p=2)

    same = labels.view(-1, 1) == labels.view(1, -1)
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = same & ~identity
    negative_mask = ~same

    # An anchor whose identity is alone in the batch has no positive; excluding
    # it is correct, and asserting it away would crash on a ragged final batch.
    usable = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not usable.any():
        return embeddings.sum() * 0.0

    hardest_positive = (distances - 1e9 * (~positive_mask).float()).amax(dim=1)
    hardest_negative = (distances + 1e9 * (~negative_mask).float()).amin(dim=1)

    difference = (hardest_positive - hardest_negative)[usable]
    if margin > 0:
        return F.relu(difference + margin).mean()
    # Soft margin: no hyper-parameter, and it keeps producing gradient after the
    # hard margin would have saturated at zero.
    return F.softplus(difference).mean()
