# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Light feature pyramid from strides 4, 8 and 16 back to stride 4 (MOD-REQ-011)."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from theoros_models.layers.blocks import ConvBN, DepthwiseBN


class FeaturePyramid(nn.Module):
    """Top-down fusion of three backbone stages.

    Each stage is projected to ``fields`` by a pointwise convolution; coarser
    levels are upsampled by nearest-neighbour doubling, which commutes with
    every symmetry of the square, and added. A depthwise and a pointwise
    convolution smooth the stride 4 result.

    Args:
        in_fields: Fields of the stride 4, 8 and 16 inputs.
        fields: Output fields.
    """

    def __init__(self, in_fields: Sequence[int], fields: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(ConvBN(count, fields, activation=False) for count in in_fields)
        self.smooth_depthwise = DepthwiseBN(fields)
        self.smooth_pointwise = ConvBN(fields, fields)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        """Fuse the stage outputs into one stride 4 map.

        Args:
            features: Maps (B, 8 * F_i, H / s_i, W / s_i) for s_i = 4, 8 and 16.

        Returns:
            (B, 8 * fields, H / 4, W / 4).
        """
        top: Tensor = self.lateral[-1](features[-1])
        for lateral, feature in zip(
            reversed(self.lateral[:-1]), reversed(features[:-1]), strict=True
        ):
            top = lateral(feature) + F.interpolate(top, scale_factor=2.0, mode="nearest")
        fused: Tensor = self.smooth_pointwise(self.smooth_depthwise(top))
        return fused
