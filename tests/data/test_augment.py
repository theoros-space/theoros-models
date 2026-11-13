# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for D4-consistent and spectral augmentation (MOD-REQ-107, MOD-REQ-108)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from theoros_models.data import augment
from theoros_models.data.sample import Sample
from theoros_models.data.synthetic import render_profile_scene
from theoros_models.layers import d4
from theoros_models.sensors import load_catalogue_profile


def _sample(size: int = 32) -> Sample:
    return render_profile_scene(load_catalogue_profile("sentinel2_msi"), size, 7)


@pytest.mark.parametrize("g", range(8))
def test_symmetry_matches_model_convention(g: int) -> None:
    array = np.random.default_rng(g).random((3, 6, 6)).astype(np.float32)
    expected = d4.transform(torch.from_numpy(array), g).numpy()
    np.testing.assert_array_equal(augment.transform_array(array, g), expected)


@pytest.mark.parametrize("g", range(8))
def test_vessel_points_follow_pixels(g: int) -> None:
    size = 10
    image = np.zeros((size, size), dtype=np.float32)
    image[2, 7] = 1.0
    moved = augment.transform_array(image, g)
    row, col = np.argwhere(moved == 1.0)[0]
    point = np.array([[2.5, 7.5, 50.0, 8.0]], dtype=np.float32)
    np.testing.assert_allclose(
        augment.transform_points(point, g, size), [[row + 0.5, col + 0.5, 50.0, 8.0]]
    )


def test_apply_symmetry_moves_everything() -> None:
    sample = _sample()
    assert sample.vessels is not None
    vessels = np.array([[3.2, 20.9, 100.0, 15.0]], dtype=np.float32)
    sample = dataclasses.replace(sample, vessels=vessels)
    turned = augment.apply_symmetry(sample, 5)
    np.testing.assert_array_equal(turned.image, augment.transform_array(sample.image, 5))
    for task in ("cloud", "land_water", "fire"):
        np.testing.assert_array_equal(
            getattr(turned, task), augment.transform_array(getattr(sample, task), 5)
        )
    assert turned.vessels is not None
    np.testing.assert_allclose(turned.vessels, augment.transform_points(vessels, 5, 32))
    assert turned.bands == sample.bands
    identity = augment.apply_symmetry(sample, 0)
    np.testing.assert_array_equal(identity.image, sample.image)
    with pytest.raises(ValueError, match="0 to 7"):
        augment.apply_symmetry(sample, 8)


def test_apply_symmetry_without_labels() -> None:
    sample = render_profile_scene(load_catalogue_profile("test_5band"), 16, 1, tasks=())
    assert augment.apply_symmetry(sample, 3).labelled_tasks() == ()


def test_spectral_jitter() -> None:
    sample = _sample()
    rng = np.random.default_rng(0)
    jittered = augment.spectral_jitter(
        sample, rng, gain_sigma=0.0, offset_sigma=0.0, noise_sigma=0.0
    )
    np.testing.assert_allclose(jittered.image, sample.image)
    changed = augment.spectral_jitter(sample, rng)
    ratio = changed.image / np.where(np.abs(sample.image) > 0.05, sample.image, np.nan)
    assert np.nanmedian(np.abs(ratio - 1.0)) < 0.2
    np.testing.assert_array_equal(changed.cloud, sample.cloud)
    assert changed.image.dtype == np.float32


def test_band_dropout_keeps_valid_subsets() -> None:
    sample = _sample()
    rng = np.random.default_rng(3)
    counts = set()
    for _ in range(50):
        dropped = augment.band_dropout(sample, rng)
        names = [band.name for band in dropped.bands]
        assert len(names) >= 3
        assert names == [band.name for band in sample.bands if band.name in names]
        indices = [sample.bands.index(band) for band in dropped.bands]
        np.testing.assert_array_equal(dropped.image, sample.image[indices])
        counts.add(len(names))
    assert len(counts) > 5
    with pytest.raises(ValueError, match="keep_min"):
        augment.band_dropout(sample, rng, keep_min=2)


def test_augmenter() -> None:
    sample = _sample()
    rng = np.random.default_rng(11)
    outputs = [augment.Augmenter(dropout_probability=1.0)(sample, rng) for _ in range(10)]
    assert any(len(out.bands) < 13 for out in outputs)
    plain = augment.Augmenter(symmetry=False, jitter=False, dropout_probability=0.0)(sample, rng)
    np.testing.assert_array_equal(plain.image, sample.image)
    three = render_profile_scene(load_catalogue_profile("test_5band"), 16, 2)
    three = augment.band_dropout(three, np.random.default_rng(0), keep_min=3)
    three = dataclasses.replace(three, image=three.image[:3], bands=three.bands[:3])
    assert len(augment.Augmenter(dropout_probability=1.0)(three, rng).bands) == 3
