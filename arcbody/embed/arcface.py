"""The additive angular margin head, applied to bodies.

This is the piece ArcBody borrows wholesale from ArcFace, and the reason the
name fits. The idea transfers because it is not about faces at all: it is about
learning an embedding whose *angles* are meaningful, by demanding that a sample
sit closer to its own class centre than to any other by a fixed angular margin.
Bodies need that more than faces do, because the nuisance variation is worse —
one person photographed in a coat and in a swimsuit differs far more in pixels
than two people in the same outfit.

The head exists only during training. At inference the embedding is taken
straight from the trunk and compared by cosine similarity, so the class count
never constrains deployment and enrolling a new person needs no retraining.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ArcMarginProduct(nn.Module):
    """Angular-margin classification logits.

    Args:
        in_features: embedding dimension.
        out_features: number of training identities.
        scale: inverse temperature ``s``. The margin only bites once the cosine
            logits are sharpened; ``s`` around 30 is the usual working point.
        margin: additive angular margin ``m`` in radians.
        sub_centers: centres per identity. More than one lets a single person
            occupy several clusters, which matters for bodies: a person in
            winter clothing and the same person in gym wear are genuinely
            different-looking, and forcing them into one centre drags the whole
            class towards the mean of both.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        scale: float = 30.0,
        margin: float = 0.30,
        sub_centers: int = 1,
    ) -> None:
        super().__init__()
        if out_features < 1:
            raise ValueError("out_features must be at least 1")
        if sub_centers < 1:
            raise ValueError("sub_centers must be at least 1")

        self.in_features = in_features
        self.out_features = out_features
        self.scale = float(scale)
        self.sub_centers = int(sub_centers)

        self.weight = nn.Parameter(torch.empty(out_features * sub_centers, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.set_margin(margin)

    def set_margin(self, margin: float) -> None:
        """Change the angular margin, refreshing its derived constants.

        Training ramps the margin up over the first epochs, so this is a normal
        operation rather than a reconfiguration hook — and doing it through a
        method keeps the four dependent constants from drifting out of step with
        the margin they were derived from.
        """
        self.margin = float(margin)
        # Constants for the stable cos(theta + m) expansion.
        self._cos_m = math.cos(self.margin)
        self._sin_m = math.sin(self.margin)
        # Beyond this cosine, theta + m exceeds pi and cos(theta + m) stops
        # increasing with theta, which would reverse the gradient. The standard
        # remedy is a linear continuation past the threshold.
        self._threshold = math.cos(math.pi - self.margin)
        self._offset = math.sin(math.pi - self.margin) * self.margin

    def cosine(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Cosine similarity to every class centre, sub-centres maxed out."""
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if self.sub_centers > 1:
            cosine = cosine.view(-1, self.out_features, self.sub_centers).amax(dim=2)
        return cosine

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Margin-penalised logits, ready for cross-entropy."""
        cosine = self.cosine(embeddings).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-9))
        target = cosine * self._cos_m - sine * self._sin_m
        target = torch.where(cosine > self._threshold, target, cosine - self._offset)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        return self.scale * (one_hot * target + (1.0 - one_hot) * cosine)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scale={self.scale}, margin={self.margin}, sub_centers={self.sub_centers}"
        )
