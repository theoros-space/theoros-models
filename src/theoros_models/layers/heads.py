# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Dense task heads (MOD-REQ-064).

Every head maps the stride 4 pyramid output to trivial channels, so its
output is an ordinary image that rotates and mirrors with the input.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from theoros_models.layers.blocks import ConvBN, DepthwiseBN
from theoros_models.layers.d4 import D4InvariantConv1x1


@dataclass(frozen=True)
class HeadSpec:
    """Output definition of one head.

    Attributes:
        name: Head identifier.
        channels: Meaning of each output channel, in order.
    """

    name: str
    channels: tuple[str, ...]


HEAD_SPECS = (
    HeadSpec("cloud", ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")),
    HeadSpec("land_water", ("water",)),
    HeadSpec("vessel", ("centre", "log_length_m", "log_width_m")),
    HeadSpec("fire", ("active_fire",)),
)
"""Cloud channels are class logits; land and water, the vessel centre and
active fire are single logits; vessel length and width are natural
logarithms of metres, read at the centre peaks."""

HEAD_NAMES = tuple(spec.name for spec in HEAD_SPECS)


class DenseHead(nn.Module):
    """Depthwise filter, pointwise mixing and an invariant output projection.

    Args:
        in_fields: Pyramid fields.
        hidden_fields: Hidden fields.
        out_channels: Trivial output channels.
    """

    def __init__(self, in_fields: int, hidden_fields: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = DepthwiseBN(in_fields)
        self.pointwise = ConvBN(in_fields, hidden_fields)
        self.output = D4InvariantConv1x1(hidden_fields, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Map (B, 8 * in_fields, H, W) to (B, out_channels, H, W)."""
        y: Tensor = self.output(self.pointwise(self.depthwise(x)))
        return y
