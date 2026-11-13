# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for the D4 group and its layers (MOD-REQ-007, MOD-REQ-062, MOD-REQ-063)."""

from __future__ import annotations

import itertools

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from tests.torch_helpers import randomise_norms
from theoros_models.layers import d4
from theoros_models.layers.d4 import ORDER

GROUP = range(ORDER)


def _generator() -> torch.Generator:
    return torch.Generator().manual_seed(1234)


def test_multiplication_table_is_a_group() -> None:
    table = d4.MULTIPLY
    assert list(table[0]) == list(GROUP)
    assert list(table[:, 0]) == list(GROUP)
    for row in table:
        assert sorted(row) == list(GROUP)
    for a, b, c in itertools.product(GROUP, repeat=3):
        assert table[table[a, b], c] == table[a, table[b, c]]
    for g in GROUP:
        assert table[g, d4.INVERSE[g]] == 0
    # Rotations commute with each other but not with mirrors.
    assert table[1, 4] != table[4, 1]


def test_table_matches_transform() -> None:
    x = torch.randn(2, 5, 5, generator=_generator())
    for a, b in itertools.product(GROUP, repeat=2):
        assert torch.equal(
            d4.transform(d4.transform(x, b), a), d4.transform(x, int(d4.MULTIPLY[a, b]))
        )


def test_regular_action_is_representation() -> None:
    x = torch.randn(2, 3 * ORDER, 6, 6, generator=_generator())
    assert torch.equal(d4.act_regular(x, 0), x)
    for g, h in itertools.product(GROUP, repeat=2):
        composed = d4.act_regular(d4.act_regular(x, h), g)
        assert torch.equal(composed, d4.act_regular(x, int(d4.MULTIPLY[g, h])))


@pytest.mark.parametrize(("kernel", "stride", "padding"), [(4, 4, 0), (3, 1, 1), (2, 2, 0)])
def test_lift_exact_on_integers(kernel: int, stride: int, padding: int) -> None:
    # With small integer inputs and weights every sum is exact in float64,
    # so equivariance must hold bit for bit (MOD-REQ-007).
    generator = _generator()
    lift = d4.D4LiftConv2d(3, 2, kernel, stride=stride, padding=padding).double()
    with torch.no_grad():
        lift.weight.copy_(torch.randint(-5, 6, lift.weight.shape, generator=generator))
    x = torch.randint(-50, 51, (2, 3, 16, 16), generator=generator).double()
    y = lift(x)
    assert y.shape == (2, 2 * ORDER, (16 + 2 * padding - kernel) // stride + 1, y.shape[-1])
    for g in GROUP:
        assert torch.equal(lift(d4.act_trivial(x, g)), d4.act_regular(y, g))


@pytest.mark.parametrize(
    "layer",
    [
        lambda: d4.D4Conv2d(2, 3, 3, padding=1),
        lambda: d4.D4Conv2d(2, 3, 1),
        lambda: d4.D4Conv2d(2, 3, 2, stride=2),
        lambda: d4.D4DepthwiseConv2d(2, 3, padding=1),
        lambda: d4.D4DepthwiseConv2d(2, 2, stride=2),
    ],
)
def test_group_layers_exact_on_integers(layer: object) -> None:
    generator = _generator()
    module = layer().double()  # type: ignore[operator]
    with torch.no_grad():
        module.weight.copy_(torch.randint(-5, 6, module.weight.shape, generator=generator))
    x = torch.randint(-50, 51, (2, 2 * ORDER, 12, 12), generator=generator).double()
    y = module(x)
    for g in GROUP:
        assert torch.equal(module(d4.act_regular(x, g)), d4.act_regular(y, g))


def test_invariant_projection() -> None:
    generator = _generator()
    module = d4.D4InvariantConv1x1(3, 2).double()
    x = torch.randn(2, 3 * ORDER, 6, 6, generator=generator, dtype=torch.float64)
    y = module(x)
    assert y.shape == (2, 2, 6, 6)
    for g in GROUP:
        torch.testing.assert_close(
            module(d4.act_regular(x, g)), d4.act_trivial(y, g), rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("training", [True, False])
def test_batch_norm_equivariant(training: bool) -> None:
    generator = _generator()
    norm = d4.D4BatchNorm2d(3).double()
    randomise_norms(norm, generator)
    norm.train(training)
    x = torch.randn(4, 3 * ORDER, 5, 5, generator=generator, dtype=torch.float64)
    for g in GROUP:
        torch.testing.assert_close(
            norm(d4.act_regular(x, g)), d4.act_regular(norm(x), g), rtol=0, atol=1e-12
        )


def test_batch_norm_shares_statistics_per_field() -> None:
    norm = d4.D4BatchNorm2d(2).double().train()
    x = torch.randn(3, 2 * ORDER, 4, 4, generator=_generator(), dtype=torch.float64)
    y = norm(x).reshape(3, 2, ORDER * 4 * 4).transpose(0, 1).reshape(2, -1)
    torch.testing.assert_close(
        y.mean(dim=1), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0
    )


def test_expanded_affine_matches_eval() -> None:
    generator = _generator()
    norm = d4.D4BatchNorm2d(3).double().eval()
    randomise_norms(norm, generator)
    x = torch.randn(2, 3 * ORDER, 4, 4, generator=generator, dtype=torch.float64)
    scale, shift = norm.expanded_affine()
    assert scale.shape == shift.shape == (3 * ORDER,)
    torch.testing.assert_close(norm(x), x * scale[:, None, None] + shift[:, None, None])


def test_expanded_weight_shapes() -> None:
    assert d4.D4LiftConv2d(5, 3, 4).expanded_weight().shape == (24, 5, 4, 4)
    assert d4.D4Conv2d(2, 3, 3).expanded_weight().shape == (24, 16, 3, 3)
    assert d4.D4DepthwiseConv2d(3, 2).expanded_weight().shape == (24, 1, 2, 2)
    assert d4.D4InvariantConv1x1(3, 4).expanded_weight().shape == (4, 24, 1, 1)


def test_expanded_weight_is_the_forward_function() -> None:
    generator = _generator()
    conv = d4.D4Conv2d(2, 3, 3, padding=1).double()
    x = torch.randn(1, 2 * ORDER, 7, 7, generator=generator, dtype=torch.float64)
    assert torch.equal(conv(x), F.conv2d(x, conv.expanded_weight(), padding=1))


def test_odd_kernel_at_stride_two_breaks_equivariance() -> None:
    # The reason downsampling uses 2 x 2 kernels: a 3 x 3 kernel at stride 2
    # samples a grid that a quarter turn of an even-sized map does not preserve.
    generator = _generator()
    conv = d4.D4LiftConv2d(1, 1, 3, stride=2, padding=1).double()
    x = torch.randn(1, 1, 8, 8, generator=generator, dtype=torch.float64)
    error = (conv(d4.act_trivial(x, 1)) - d4.act_regular(conv(x), 1)).abs().max()
    assert error > 1e-3
