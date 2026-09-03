"""A compact residual trunk sized for CPU inference.

Person re-identification conventions drive the shape of this network more than
image-classification ones do:

* the input is a tall 1:2 crop, because bodies are;
* the last stage keeps stride 1, preserving spatial detail that a waist-to-hip
  transition lives in and that a stride-2 stage would blur away;
* average and max pooling are concatenated, because a body's identity is partly
  a global statistic (overall build) and partly a peak response (the one
  distinctive silhouette feature).

Widths are deliberately modest. The service's stated budget is one to two
seconds per photo on a CPU, shared with two perception models, and a heavier
trunk would spend that budget without a dataset large enough to justify it.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a projection shortcut when shape changes."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.activation(out + residual)


class BodyBackbone(nn.Module):
    """Feature trunk for body crops.

    Args:
        in_channels: 4 by default — RGB plus the silhouette mask. Handing the
            mask to the network explicitly is the single cheapest way to make
            the embedding about *shape* rather than about clothing: the trunk
            gets the body's outline for free instead of having to learn to
            segment before it can compare.
        widths: channel count per stage.
    """

    def __init__(
        self,
        in_channels: int = 4,
        widths: tuple[int, int, int, int] = (64, 128, 256, 384),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = widths[-1]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0] // 2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(widths[0] // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = self._stage(widths[0] // 2, widths[0], stride=1)
        self.stage2 = self._stage(widths[0], widths[1], stride=2)
        self.stage3 = self._stage(widths[1], widths[2], stride=2)
        # Stride 1 in the final stage: re-ID's "last stride" trick.
        self.stage4 = self._stage(widths[2], widths[3], stride=1)

    @staticmethod
    def _stage(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=stride),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.stage4(x)
