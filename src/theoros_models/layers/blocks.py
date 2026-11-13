# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""D4-equivariant convolution units and inverted residual blocks.

All feature maps are regular fields (see :mod:`theoros_models.layers.d4`).
Stride 1 layers use 3 x 3 kernels with padding 1. Stride 2 layers use 2 x 2
kernels without padding: on an even-sized map their sampling grid is mapped
onto itself by every symmetry of the square, which a 3 x 3 kernel at stride 2
is not, so downsampling stays exactly equivariant.
"""

from __future__ import annotations

from torch import Tensor, nn

from theoros_models.layers.d4 import D4BatchNorm2d, D4Conv2d, D4DepthwiseConv2d


class ConvBN(nn.Module):
    """Group convolution, batch normalisation and optional ReLU6.

    Args:
        in_fields: Input fields.
        out_fields: Output fields.
        kernel_size: 1 for pointwise or 3 for a stride 1 spatial kernel.
        activation: Apply ReLU6 after normalisation.
    """

    def __init__(
        self, in_fields: int, out_fields: int, kernel_size: int = 1, *, activation: bool = True
    ) -> None:
        super().__init__()
        self.conv = D4Conv2d(in_fields, out_fields, kernel_size, padding=kernel_size // 2)
        self.norm = D4BatchNorm2d(out_fields)
        self.act: nn.Module = nn.ReLU6() if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Map (B, 8 * in_fields, H, W) to (B, 8 * out_fields, H, W)."""
        y: Tensor = self.act(self.norm(self.conv(x)))
        return y


class DepthwiseBN(nn.Module):
    """Depthwise group convolution, batch normalisation and ReLU6.

    Args:
        fields: Fields, unchanged.
        stride: 1 (3 x 3 kernel, padding 1) or 2 (2 x 2 kernel, no padding).
    """

    def __init__(self, fields: int, stride: int = 1) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        kernel, padding = (3, 1) if stride == 1 else (2, 0)
        self.conv = D4DepthwiseConv2d(fields, kernel, stride=stride, padding=padding)
        self.norm = D4BatchNorm2d(fields)
        self.act = nn.ReLU6()

    def forward(self, x: Tensor) -> Tensor:
        """Map (B, 8 * fields, H, W) to (B, 8 * fields, H / stride, W / stride)."""
        y: Tensor = self.act(self.norm(self.conv(x)))
        return y


class InvertedResidual(nn.Module):
    """Expand, depthwise filter and project, with a residual when shapes match.

    Squeeze-and-excitation is not used: its global pooling and per-channel
    gating add operators outside the flight set for little gain at this size.

    Args:
        in_fields: Input fields.
        out_fields: Output fields.
        expansion: Width multiplier of the hidden layer.
        stride: 1 or 2.
    """

    def __init__(self, in_fields: int, out_fields: int, expansion: int, stride: int = 1) -> None:
        super().__init__()
        hidden = in_fields * expansion
        self.expand = ConvBN(in_fields, hidden)
        self.depthwise = DepthwiseBN(hidden, stride)
        self.project = ConvBN(hidden, out_fields, activation=False)
        self.residual = stride == 1 and in_fields == out_fields

    def forward(self, x: Tensor) -> Tensor:
        """Map (B, 8 * in_fields, H, W) to (B, 8 * out_fields, H / stride, W / stride)."""
        y: Tensor = self.project(self.depthwise(self.expand(x)))
        return x + y if self.residual else y
