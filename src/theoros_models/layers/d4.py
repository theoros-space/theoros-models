# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""The dihedral group D4 and convolutions equivariant to it (MOD-REQ-007, MOD-REQ-062).

D4 is the group of the 8 symmetries of the square. Element ``g = (m, r)``
with index ``4 * m + r`` acts on an image by an optional mirror of the
width axis followed by ``r`` counter-clockwise quarter turns.

Feature maps come in two kinds:

* trivial fields, shape (B, C, H, W): the group acts on the pixels only;
* regular fields, shape (B, C * 8, H, W), channel ``8 * c + h`` holding the
  response of field ``c`` to group element ``h``: the group acts on the
  pixels and permutes the 8 channels of each field, ``g . f(h) = T_g f(g^-1 h)``.

Every layer here stores only its unique (training) parameters and builds
the plain convolution weight it is equivalent to through
``expanded_weight``. The forward pass uses that weight, so the compiler's
D4 expansion is the same function and introduces no numerical change.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

ORDER = 8


def transform(x: Tensor, g: int) -> Tensor:
    """Apply group element ``g`` to the last two (spatial) axes.

    Args:
        x: Tensor of shape (..., H, W) with H equal to W.
        g: Group element index in 0 to 7.

    Returns:
        The mirrored then rotated tensor, same shape.
    """
    mirror, quarter_turns = divmod(g, 4)
    if mirror:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, quarter_turns, dims=(-2, -1))


def _tables() -> tuple[np.ndarray, np.ndarray]:
    # Derive the multiplication table from the action itself on an
    # asymmetric pattern, so it matches ``transform`` by construction.
    pattern = torch.arange(9.0).reshape(3, 3)
    images = [transform(pattern, g) for g in range(ORDER)]
    table = np.zeros((ORDER, ORDER), dtype=np.int64)
    for a in range(ORDER):
        for b in range(ORDER):
            composed = transform(images[b], a)
            table[a, b] = next(c for c in range(ORDER) if torch.equal(images[c], composed))
    inverse = np.array([int(np.flatnonzero(table[g] == 0)[0]) for g in range(ORDER)])
    return table, inverse


MULTIPLY, INVERSE = _tables()
"""``MULTIPLY[a, b]`` is the index of ``a b`` (apply ``b`` first); ``INVERSE[g]`` of ``g^-1``."""


def act_trivial(x: Tensor, g: int) -> Tensor:
    """Act with ``g`` on trivial fields (B, C, H, W): pixels only."""
    return transform(x, g)


def act_regular(x: Tensor, g: int) -> Tensor:
    """Act with ``g`` on regular fields (B, C * 8, H, W).

    Args:
        x: Regular feature map.
        g: Group element index.

    Returns:
        ``T_g f(g^-1 h)`` for every field and group channel ``h``, same shape.
    """
    batch, channels, height, width = x.shape
    fields = x.reshape(batch, channels // ORDER, ORDER, height, width)
    order = torch.as_tensor(MULTIPLY[INVERSE[g]], device=x.device)
    return transform(fields[:, :, order], g).reshape(batch, channels, height, width)


def _stack_transforms(kernel: Tensor) -> Tensor:
    # (..., k, k) -> (8, ..., k, k) with T_h applied for every h.
    return torch.stack([transform(kernel, h) for h in range(ORDER)])


class D4LiftConv2d(nn.Module):
    """Convolution from trivial fields to regular fields.

    Output field ``o`` at group channel ``h`` correlates the input with the
    filter ``T_h psi_o``.

    Args:
        in_channels: Trivial input channels.
        out_fields: Regular output fields; the output has ``8 * out_fields`` channels.
        kernel_size: Square kernel size.
        stride: Stride; the sampling grid must be symmetric under D4.
        padding: Zero padding on every side.
    """

    def __init__(
        self, in_channels: int, out_fields: int, kernel_size: int, stride: int = 1, padding: int = 0
    ) -> None:
        super().__init__()
        self.in_channels, self.out_fields = in_channels, out_fields
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.weight = nn.Parameter(torch.empty(out_fields, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")

    def expanded_weight(self) -> Tensor:
        """Return the plain weight, shape (8 * out_fields, in_channels, k, k)."""
        stacked = _stack_transforms(self.weight)  # (8, O, I, k, k)
        return stacked.transpose(0, 1).reshape(-1, self.in_channels, *self.weight.shape[-2:])

    def forward(self, x: Tensor) -> Tensor:
        """Map (B, in_channels, H, W) to (B, 8 * out_fields, H', W')."""
        return F.conv2d(x, self.expanded_weight(), stride=self.stride, padding=self.padding)


class D4Conv2d(nn.Module):
    """Group convolution between regular fields.

    The expanded weight is ``W[o, h, c, h'] = T_h psi[o, c, h^-1 h']``.

    Args:
        in_fields: Regular input fields.
        out_fields: Regular output fields.
        kernel_size: Square kernel size.
        stride: Stride; the sampling grid must be symmetric under D4.
        padding: Zero padding on every side.
    """

    def __init__(
        self, in_fields: int, out_fields: int, kernel_size: int, stride: int = 1, padding: int = 0
    ) -> None:
        super().__init__()
        self.in_fields, self.out_fields = in_fields, out_fields
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.weight = nn.Parameter(
            torch.empty(out_fields, in_fields, ORDER, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight.view(out_fields, -1, kernel_size, kernel_size))
        index = np.stack([MULTIPLY[INVERSE[h]] for h in range(ORDER)])
        self.register_buffer("_index", torch.as_tensor(index), persistent=False)

    def expanded_weight(self) -> Tensor:
        """Return the plain weight, shape (8 * out_fields, 8 * in_fields, k, k)."""
        k = self.kernel_size
        index = cast(Tensor, self._index)
        # permuted[h, o, c, h'] = psi[o, c, h^-1 h']
        permuted = torch.stack([self.weight[:, :, index[h]] for h in range(ORDER)])
        rotated = torch.stack([transform(permuted[h], h) for h in range(ORDER)])
        return rotated.permute(1, 0, 2, 3, 4, 5).reshape(
            ORDER * self.out_fields, ORDER * self.in_fields, k, k
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map (B, 8 * in_fields, H, W) to (B, 8 * out_fields, H', W')."""
        return F.conv2d(x, self.expanded_weight(), stride=self.stride, padding=self.padding)


class D4DepthwiseConv2d(nn.Module):
    """Depthwise convolution on regular fields.

    Channel ``8 * c + h`` is filtered with ``T_h psi_c``.

    Args:
        fields: Regular fields, unchanged by the layer.
        kernel_size: Square kernel size.
        stride: Stride; the sampling grid must be symmetric under D4.
        padding: Zero padding on every side.
    """

    def __init__(self, fields: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.fields = fields
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.weight = nn.Parameter(torch.empty(fields, 1, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")

    def expanded_weight(self) -> Tensor:
        """Return the plain depthwise weight, shape (8 * fields, 1, k, k)."""
        stacked = _stack_transforms(self.weight)  # (8, C, 1, k, k)
        return stacked.transpose(0, 1).reshape(ORDER * self.fields, 1, *self.weight.shape[-2:])

    def forward(self, x: Tensor) -> Tensor:
        """Map (B, 8 * fields, H, W) to (B, 8 * fields, H', W')."""
        return F.conv2d(
            x,
            self.expanded_weight(),
            stride=self.stride,
            padding=self.padding,
            groups=ORDER * self.fields,
        )


class D4InvariantConv1x1(nn.Module):
    """Pointwise map from regular fields to trivial channels, with bias.

    The weight is shared by the 8 group channels of each field, so the
    output is invariant under the channel permutation and transforms as an
    ordinary image. This sums over the group, the group pooling of the head.

    Args:
        in_fields: Regular input fields.
        out_channels: Trivial output channels.
    """

    def __init__(self, in_fields: int, out_channels: int) -> None:
        super().__init__()
        self.in_fields, self.out_channels = in_fields, out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_fields))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.normal_(self.weight, std=(ORDER * in_fields) ** -0.5)

    def expanded_weight(self) -> Tensor:
        """Return the plain weight, shape (out_channels, 8 * in_fields, 1, 1)."""
        return self.weight.repeat_interleave(ORDER, dim=1)[:, :, None, None]

    def forward(self, x: Tensor) -> Tensor:
        """Map (B, 8 * in_fields, H, W) to (B, out_channels, H, W)."""
        return F.conv2d(x, self.expanded_weight(), self.bias)


class D4BatchNorm2d(nn.BatchNorm2d):
    """Batch normalisation of regular fields with statistics shared per field.

    The 8 group channels of a field share mean, variance, scale and shift, so
    normalisation commutes with the channel permutation.

    Args:
        fields: Regular fields.
    """

    def __init__(self, fields: int) -> None:
        super().__init__(fields)
        self.fields = fields

    def forward(self, x: Tensor) -> Tensor:
        """Normalise (B, 8 * fields, H, W)."""
        batch, channels, height, width = x.shape
        grouped = x.reshape(batch, self.fields, ORDER * height, width)
        return super().forward(grouped).reshape(batch, channels, height, width)

    def expanded_affine(self) -> tuple[Tensor, Tensor]:
        """Return the inference-time per-channel scale and shift, each (8 * fields,).

        ``y = scale * x + shift`` reproduces the layer in evaluation mode;
        the compiler folds it into the preceding convolution.
        """
        running_var: Tensor = self.running_var  # type: ignore[assignment]
        running_mean: Tensor = self.running_mean  # type: ignore[assignment]
        scale = self.weight / torch.sqrt(running_var + self.eps)
        shift = self.bias - running_mean * scale
        return scale.repeat_interleave(ORDER), shift.repeat_interleave(ORDER)
