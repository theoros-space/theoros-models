# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for TheorosNet assembly, gating and equivariance (MOD-REQ-009, 011, 063, 064)."""

from __future__ import annotations

import pytest
import torch

from tests.torch_helpers import randomise_norms
from theoros_models.arch import TIERS
from theoros_models.layers import d4
from theoros_models.layers.heads import HEAD_SPECS
from theoros_models.model import BandBatch, TheorosNet
from theoros_models.sensors import load_catalogue_profile


def _model(tier: str, dtype: torch.dtype = torch.float32) -> TheorosNet:
    torch.manual_seed(2026)
    model = TheorosNet(TIERS[tier]).to(dtype).eval()
    randomise_norms(model, torch.Generator().manual_seed(7))
    return model


def _batch(
    profile: str, size: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, BandBatch]:
    bands = BandBatch.from_profile(load_catalogue_profile(profile), size, dtype=dtype)
    x = torch.randn(
        size, bands.mask.shape[1], 64, 64, generator=torch.Generator().manual_seed(1), dtype=dtype
    )
    return x, bands


def test_output_shapes() -> None:
    x, bands = _batch("sentinel2_msi", 2)
    with torch.no_grad():
        out = _model("nano")(x, bands)
    assert set(out.heads) == {spec.name for spec in HEAD_SPECS}
    for spec in HEAD_SPECS:
        assert out.heads[spec.name].shape == (2, len(spec.channels), 16, 16)
    assert out.enabled.shape == (2, 4)
    assert bool(out.enabled.all())


def test_full_tile() -> None:
    x = torch.randn(1, 13, 256, 256)
    with torch.no_grad():
        out = _model("nano")(x, BandBatch.from_profile(load_catalogue_profile("sentinel2_msi")))
    assert out.heads["cloud"].shape == (1, 4, 64, 64)


def test_feature_strides() -> None:
    x, bands = _batch("test_5band", 1)
    model = _model("base")
    with torch.no_grad():
        features = model.features(x, bands)
    fields = TIERS["base"].stage_fields
    assert [tuple(f.shape) for f in features] == [
        (1, 8 * fields[0], 16, 16),
        (1, 8 * fields[1], 8, 8),
        (1, 8 * fields[2], 4, 4),
    ]


@pytest.mark.parametrize(
    ("tier", "dtype", "tolerance"),
    [
        ("nano", torch.float64, 1e-9),
        ("base", torch.float64, 1e-9),
        ("nano", torch.float32, 1e-4),
        ("max", torch.float32, 1e-4),
    ],
)
def test_network_equivariance(tier: str, dtype: torch.dtype, tolerance: float) -> None:
    model = _model(tier, dtype)
    x, bands = _batch("sentinel2_msi", 2, dtype)
    with torch.no_grad():
        reference = model(x, bands).heads
        for g in range(d4.ORDER):
            moved = model(d4.act_trivial(x, g), bands).heads
            for name, output in reference.items():
                expected = d4.act_trivial(output, g)
                scale = 1.0 if dtype == torch.float64 else float(expected.abs().max())
                error = float((moved[name] - expected).abs().max())
                assert error <= tolerance * scale, (name, g, error)


def test_gating_skips_unsupported_heads() -> None:
    x, bands = _batch("planetscope_superdove", 2)
    with torch.no_grad():
        out = _model("nano")(x, bands)
    assert "fire" not in out.heads
    assert out.enabled.tolist() == [[True, True, True, False]] * 2


def test_mixed_batch_reports_per_sample() -> None:
    profiles = [
        load_catalogue_profile("planetscope_superdove"),
        load_catalogue_profile("sentinel2_msi"),
    ]
    bands = BandBatch.from_profiles(profiles)
    assert bands.meta.shape == (2, 13, 3)
    assert bands.mask.sum(dim=1).tolist() == [8, 13]
    x = torch.randn(2, 13, 32, 32)
    with torch.no_grad():
        out = _model("nano")(x, bands)
    assert "fire" in out.heads
    assert out.enabled[:, 3].tolist() == [False, True]


def test_input_validation() -> None:
    model = _model("nano")
    _, bands = _batch("test_5band", 2)
    with pytest.raises(ValueError, match="metadata describes"):
        model(torch.zeros(2, 4, 32, 32), bands)
    with pytest.raises(ValueError, match="divisible by 16"):
        model(torch.zeros(2, 5, 32, 24), bands)
    with pytest.raises(ValueError, match="divisible by 16"):
        model(torch.zeros(2, 5, 40, 40), bands)


def test_band_batch_validation() -> None:
    meta = torch.ones(2, 3, 3)
    with pytest.raises(ValueError, match="shape \\(B, N, 3\\)"):
        BandBatch(torch.ones(2, 3), torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="bool tensor"):
        BandBatch(meta, torch.ones(2, 3))
    with pytest.raises(ValueError, match="at least one real band"):
        BandBatch(meta, torch.tensor([[True, False, False], [False, False, False]]))
