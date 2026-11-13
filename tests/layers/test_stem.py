# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for the spectral hypernetwork stem (MOD-REQ-060)."""

from __future__ import annotations

import torch
from torch.nn import functional as F  # noqa: N812

from theoros_models.layers.stem import SpectralStem, band_features
from theoros_models.model import BandBatch
from theoros_models.sensors import load_catalogue_profile


def _stem() -> SpectralStem:
    torch.manual_seed(5)
    return SpectralStem(6, embed=8, hidden=12, frequencies=4).double()


def _inputs() -> tuple[torch.Tensor, BandBatch]:
    profile = load_catalogue_profile("test_5band")
    bands = BandBatch.from_profile(profile, 2, dtype=torch.float64)
    x = torch.randn(2, 5, 8, 8, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    return x, bands


def test_band_features_shape_and_range() -> None:
    meta = torch.tensor([[[400.0, 100.0, 10.0], [2400.0, 50.0, 30.0]]])
    features = band_features(meta, 3)
    assert features.shape == (1, 2, 8)
    torch.testing.assert_close(features[0, 0, :3], torch.zeros(3))
    torch.testing.assert_close(
        features[0, :, 6:], torch.log(torch.tensor([[1.0, 1.0], [0.5, 3.0]]))
    )


def test_output_shape() -> None:
    x, bands = _inputs()
    assert _stem()(x, bands.meta, bands.mask).shape == (2, 6, 8, 8)


def test_band_order_does_not_matter() -> None:
    stem = _stem()
    x, bands = _inputs()
    order = torch.tensor([3, 0, 4, 1, 2])
    shuffled = stem(x[:, order], bands.meta[:, order], bands.mask[:, order])
    torch.testing.assert_close(shuffled, stem(x, bands.meta, bands.mask), rtol=0, atol=1e-12)


def test_padding_slots_are_ignored() -> None:
    stem = _stem()
    x, bands = _inputs()
    padded_x = torch.cat([x, torch.full((2, 2, 8, 8), 1e6, dtype=torch.float64)], dim=1)
    junk = torch.tensor([[700.0, 5.0, 0.5], [1500.0, 300.0, 900.0]], dtype=torch.float64)
    padded_meta = torch.cat([bands.meta, junk.expand(2, -1, -1)], dim=1)
    padded_mask = torch.cat([bands.mask, torch.zeros(2, 2, dtype=torch.bool)], dim=1)
    kernels, _ = stem.generate(padded_meta, padded_mask)
    assert torch.equal(kernels[:, 5:], torch.zeros_like(kernels[:, 5:]))
    torch.testing.assert_close(
        stem(padded_x, padded_meta, padded_mask),
        stem(x, bands.meta, bands.mask),
        rtol=0,
        atol=1e-12,
    )


def test_equals_plain_convolution() -> None:
    stem = _stem()
    x, bands = _inputs()
    kernels, bias = stem.generate(bands.meta, bands.mask)
    for sample in range(2):
        weight = kernels[sample].T[:, :, None, None]  # (C_s, N, 1, 1)
        expected = F.conv2d(x[sample : sample + 1], weight, bias[sample])
        torch.testing.assert_close(
            stem(x, bands.meta, bands.mask)[sample : sample + 1], expected, rtol=0, atol=1e-12
        )


def test_kernels_depend_on_metadata() -> None:
    stem = _stem()
    _, bands = _inputs()
    kernels, _ = stem.generate(bands.meta, bands.mask)
    coarse = bands.meta.clone()
    coarse[..., 2] *= 6.0
    coarse_kernels, _ = stem.generate(coarse, bands.mask)
    assert not torch.allclose(kernels, coarse_kernels)
