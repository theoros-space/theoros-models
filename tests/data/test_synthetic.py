# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for the synthetic scene generator (MOD-REQ-106)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings

from tests.strategies import valid_profiles
from theoros_models.data import synthetic
from theoros_models.data.sample import IGNORE
from theoros_models.sensors import SensorProfile, load_catalogue_profile


def test_material_spectra_features() -> None:
    grid = np.array([480.0, 665.0, 865.0, 1610.0, 2200.0])
    water, vegetation, soil, cloud, _, fire = synthetic.material_spectra(grid)
    assert vegetation[2] > 4 * vegetation[1]  # red edge
    assert water[2] < 0.01 < water[0]  # water absorbs NIR
    assert fire[4] > 2 * soil[4]  # emission at 2.2 um
    assert fire[4] - soil[4] > 2 * (fire[3] - soil[3])
    assert cloud[:3].min() > 0.7


def test_deterministic() -> None:
    profile = load_catalogue_profile("sentinel2_msi")
    first = synthetic.render_profile_scene(profile, 64, 5)
    second = synthetic.render_profile_scene(profile, 64, 5)
    other = synthetic.render_profile_scene(profile, 64, 6)
    np.testing.assert_array_equal(first.image, second.image)
    assert first.sample_id == "sentinel2_msi-5"
    assert not np.array_equal(first.image, other.image)


def test_labels_match_materials() -> None:
    profile = load_catalogue_profile("sentinel2_msi")
    names = profile.band_names
    for seed in range(8):
        sample = synthetic.render_profile_scene(profile, 128, seed)
        assert sample.cloud is not None
        assert sample.land_water is not None
        clear = sample.cloud == 0
        nir, green = sample.image[names.index("B08")], sample.image[names.index("B03")]
        water = clear & (sample.land_water == 1)
        land = clear & (sample.land_water == 0)
        if water.any() and land.any():
            assert np.median(nir[water]) < np.median(nir[land])
            assert np.median((green - nir)[water]) > 0
        thick = sample.cloud == 1
        if thick.any():
            assert np.all(sample.land_water[thick] == IGNORE)
            assert sample.image[names.index("B02")][thick].mean() > 0.7


def test_vessels_on_water_with_sizes() -> None:
    profile = load_catalogue_profile("planetscope_superdove")
    found = 0
    for seed in range(20):
        sample = synthetic.render_profile_scene(profile, 128, seed)
        assert sample.vessels is not None
        assert sample.land_water is not None
        for row, col, length, width in sample.vessels:
            assert sample.land_water[int(row), int(col)] == 1
            assert 20.0 <= length <= 200.0
            assert length / 8.0 <= width <= length / 5.0
            found += 1
    assert found > 0


def test_task_selection() -> None:
    profile = load_catalogue_profile("test_5band")
    sample = synthetic.render_profile_scene(profile, 32, 1, tasks=("fire",))
    assert sample.labelled_tasks() == ("fire",)
    with pytest.raises(ValueError, match="unknown tasks: smoke"):
        synthetic.render_profile_scene(profile, 32, 1, tasks=("smoke",))


def test_smooth_noise_range() -> None:
    field = synthetic.smooth_noise(np.random.default_rng(0), 50, 4)
    assert field.shape == (50, 50)
    assert field.min() >= 0.0
    assert field.max() <= 1.0
    assert np.abs(np.diff(field, axis=0)).max() < 0.2


@settings(max_examples=25)
@given(valid_profiles())
def test_any_band_set(data: dict[str, Any]) -> None:
    profile = SensorProfile.model_validate(data)
    sample = synthetic.render_profile_scene(profile, 32, 3)
    assert sample.image.shape == (len(profile.bands), 32, 32)
    assert np.isfinite(sample.image).all()
