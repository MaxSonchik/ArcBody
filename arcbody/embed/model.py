"""``ArcBodyNet``: one trunk, two heads.

The embedding head is the product. The shape head is the honesty check: it
regresses the same dimensionless ratios the geometric estimator measures, from
the same crop, without ever seeing the silhouette analysis. When the two agree,
the measurement is corroborated by an independent path; when they disagree, the
request says so instead of picking a winner silently.

Training it as an auxiliary task also regularises the embedding towards
body shape and away from appearance shortcuts — the network cannot satisfy the
ratio head by memorising a jacket.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from arcbody.embed.backbone import BodyBackbone
from arcbody.measure.schema import NUM_RATIOS


@dataclass
class ArcBodyOutput:
    """What one forward pass produces."""

    embedding: torch.Tensor
    ratios: torch.Tensor


class ArcBodyNet(nn.Module):
    """Body-shape encoder with an auxiliary ratio head.

    Args:
        embedding_dim: width of the vector that leaves the service.
        in_channels: 4 for RGB plus mask.
        dropout: applied to the pooled feature only, never to the embedding —
            dropping components of a vector that is about to be L2-normalised
            and compared by angle would inject noise straight into the metric.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        *,
        in_channels: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.backbone = BodyBackbone(in_channels=in_channels)

        pooled_dim = self.backbone.out_channels * 2  # average and max, concatenated
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Linear(pooled_dim, self.embedding_dim, bias=False)
        # BNNeck: batch-norm between the metric embedding and the classifier, so
        # the classification loss shapes a space that cosine similarity can
        # actually use at inference. Standard practice in re-ID, and the reason
        # the deployed vector is taken from *before* this layer.
        self.neck = nn.BatchNorm1d(self.embedding_dim)
        self.neck.bias.requires_grad_(False)

        self.shape_head = nn.Sequential(
            nn.Linear(pooled_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, NUM_RATIOS),
        )

    def pool(self, features: torch.Tensor) -> torch.Tensor:
        average = F.adaptive_avg_pool2d(features, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(features, 1).flatten(1)
        return torch.cat([average, maximum], dim=1)

    def forward(self, x: torch.Tensor) -> ArcBodyOutput:
        pooled = self.pool(self.backbone(x))
        embedding = self.embedding(self.dropout(pooled))
        return ArcBodyOutput(embedding=embedding, ratios=self.shape_head(pooled))

    def classifier_input(self, embedding: torch.Tensor) -> torch.Tensor:
        """The BNNeck output the angular-margin head consumes during training."""
        return self.neck(embedding)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> ArcBodyOutput:
        """Inference: an L2-normalised embedding and the predicted ratios."""
        self.eval()
        out = self.forward(x)
        return ArcBodyOutput(
            embedding=F.normalize(out.embedding, dim=1),
            ratios=out.ratios,
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
