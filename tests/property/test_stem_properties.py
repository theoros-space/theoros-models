# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Property tests: the stem reads any valid band set (MOD-REQ-061)."""

from __future__ import annotations

from typing import Any

import torch
from hypothesis import given, settings

from tests.strategies import valid_profiles
from theoros_models.layers.stem import SpectralStem
from theoros_models.model import BandBatch
from theoros_models.sensors import SensorProfile

torch.manual_seed(11)
STEM = SpectralStem(8, embed=16, hidden=32, frequencies=8).eval()


@settings(max_examples=50)
@given(valid_profiles())
def test_any_band_set(data: dict[str, Any]) -> None:
    profile = SensorProfile.model_validate(data)
    bands = BandBatch.from_profile(profile)
    x = torch.rand(1, len(profile.bands), 4, 4)
    with torch.no_grad():
        y = STEM(x, bands.meta, bands.mask)
    assert y.shape == (1, 8, 4, 4)
    assert bool(torch.isfinite(y).all())
