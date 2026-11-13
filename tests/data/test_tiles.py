# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for reflectance, normalisation, resampling and tiling (MOD-REQ-100 to 102)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from theoros_models.data import tiles
from theoros_models.sensors.schema import ReflectanceEncoding


def test_reflectance_and_normalisation() -> None:
    encoding = ReflectanceEncoding(scale=1e-4, offset=-0.1)
    reflectance = tiles.to_reflectance(np.array([1000, 3000, 30000], dtype=np.uint16), encoding)
    np.testing.assert_allclose(reflectance, [0.0, 0.2, 2.9], rtol=1e-6)
    assert reflectance.dtype == np.float32
    x = tiles.normalise(reflectance)
    np.testing.assert_allclose(x, [-1.0, 0.0, 6.5], rtol=1e-6)
    np.testing.assert_allclose(tiles.denormalise(x[:2]), reflectance[:2], atol=1e-7)
    np.testing.assert_allclose(tiles.normalise([-0.5]), [-1.0])


def test_band_resampling() -> None:
    band = np.arange(12, dtype=np.uint16).reshape(3, 4)
    up = tiles.upsample_band(band, 2)
    assert up.shape == (6, 8)
    assert up.dtype == np.uint16
    assert up[5, 7] == 11
    np.testing.assert_array_equal(tiles.downsample_band(up, 2), band)
    field = np.random.default_rng(0).random((12, 18))
    assert tiles.downsample_band(field, 6).mean() == pytest.approx(field.mean(), rel=1e-6)
    with pytest.raises(ValueError, match="cannot downsample"):
        tiles.downsample_band(field, 5)
    with pytest.raises(ValueError, match="at least 1"):
        tiles.upsample_band(band, 0)


@given(
    st.integers(min_value=1, max_value=700),
    st.integers(min_value=1, max_value=700),
    st.sampled_from([64, 256]),
    st.integers(min_value=0, max_value=63),
)
def test_windows_cover_image(height: int, width: int, size: int, overlap: int) -> None:
    windows = tiles.tile_windows(height, width, size, overlap)
    covered = np.zeros((max(height, size), max(width, size)), dtype=int)
    for window in windows:
        rows, cols = window.slices()
        covered[rows, cols] += 1
        assert window.row >= 0
        assert window.col >= 0
        assert window.row + size <= max(height, size)
        assert window.col + size <= max(width, size)
    assert covered[:height, :width].min() >= 1


def test_window_validation() -> None:
    with pytest.raises(ValueError, match="overlap"):
        tiles.tile_windows(100, 100, 64, 64)
    with pytest.raises(ValueError, match="empty"):
        tiles.tile_windows(0, 100, 64)


def test_pad_extract_and_stitch_roundtrip() -> None:
    rng = np.random.default_rng(1)
    image = rng.random((2, 150, 300)).astype(np.float32)
    windows = tiles.tile_windows(150, 300, 128, overlap=32)
    padded, valid = tiles.pad_to_tile(image, 128)
    assert padded.shape == (2, 150, 300)
    assert valid.all()
    cut = [tiles.extract(image, window) for window in windows]
    assert all(tile.shape == (2, 128, 128) for tile in cut)
    np.testing.assert_allclose(tiles.stitch(cut, windows, 150, 300), image, rtol=1e-6)


def test_small_image_is_padded() -> None:
    image = np.ones((3, 40, 50), dtype=np.float32)
    padded, valid = tiles.pad_to_tile(image, 64, fill=-1)
    assert padded.shape == (3, 64, 64)
    assert valid.sum() == 40 * 50
    assert padded[0, 63, 63] == -1
    windows = tiles.tile_windows(40, 50, 64)
    assert windows == [tiles.Window(0, 0, 64)]
    stitched = tiles.stitch([tiles.extract(padded, windows[0])], windows, 40, 50)
    np.testing.assert_array_equal(stitched, image)


def test_stitch_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="1 tiles for 2 windows"):
        tiles.stitch([np.zeros((4, 4))], [tiles.Window(0, 0, 4)] * 2, 4, 4)
