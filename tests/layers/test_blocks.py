# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Equivariance and shape tests for blocks, neck and heads (MOD-REQ-063)."""

from __future__ import annotations

import pytest
import torch

from tests.torch_helpers import randomise_norms
from theoros_models.layers import d4
from theoros_models.layers.blocks import DepthwiseBN, InvertedResidual
from theoros_models.layers.d4 import ORDER
from theoros_models.layers.heads import DenseHead
from theoros_models.layers.neck import FeaturePyramid


def _generator() -> torch.Generator:
    return torch.Generator().manual_seed(99)


@pytest.mark.parametrize(
    ("in_fields", "out_fields", "stride", "residual"),
    [(3, 3, 1, True), (3, 4, 1, False), (3, 4, 2, False), (3, 3, 2, False)],
)
def test_inverted_residual(in_fields: int, out_fields: int, stride: int, residual: bool) -> None:
    generator = _generator()
    block = InvertedResidual(in_fields, out_fields, 2, stride).double().eval()
    randomise_norms(block, generator)
    assert block.residual is residual
    x = torch.randn(2, ORDER * in_fields, 8, 8, generator=generator, dtype=torch.float64)
    y = block(x)
    assert y.shape == (2, ORDER * out_fields, 8 // stride, 8 // stride)
    for g in range(ORDER):
        torch.testing.assert_close(
            block(d4.act_regular(x, g)), d4.act_regular(y, g), rtol=0, atol=1e-10
        )


def test_depthwise_rejects_stride() -> None:
    with pytest.raises(ValueError, match="stride must be 1 or 2"):
        DepthwiseBN(2, stride=3)


def test_pyramid_and_head() -> None:
    generator = _generator()
    neck = FeaturePyramid((2, 3, 4), 3).double().eval()
    head = DenseHead(3, 2, 5).double().eval()
    randomise_norms(neck, generator)
    randomise_norms(head, generator)
    features = [
        torch.randn(2, ORDER * fields, size, size, generator=generator, dtype=torch.float64)
        for fields, size in ((2, 16), (3, 8), (4, 4))
    ]
    y = head(neck(features))
    assert y.shape == (2, 5, 16, 16)
    for g in range(ORDER):
        moved = [d4.act_regular(feature, g) for feature in features]
        torch.testing.assert_close(head(neck(moved)), d4.act_trivial(y, g), rtol=0, atol=1e-10)
