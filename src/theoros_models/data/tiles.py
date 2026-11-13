# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Reflectance conversion, normalisation, band resampling and tiling.

The model reads normalised top-of-atmosphere reflectance (MOD-REQ-009,
MOD-REQ-101):

    reflectance = encoding.scale * stored + encoding.offset
    x           = (clip(reflectance, 0, 1.5) - 0.2) / 0.2

The clip is the saturation of the int8 input quantiser; the affine parts are
folded into the stem by the compiler.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from theoros_models.arch import TILE_SIZE
from theoros_models.sensors.schema import ReflectanceEncoding

FloatArray = NDArray[np.float32]

REFLECTANCE_MIN = 0.0
REFLECTANCE_MAX = 1.5
NORMALISATION_OFFSET = 0.2
NORMALISATION_SCALE = 0.2


def to_reflectance(stored: ArrayLike, encoding: ReflectanceEncoding) -> FloatArray:
    """Convert stored pixel values to TOA reflectance.

    Args:
        stored: Stored values of any shape, for example (N, H, W) uint16 DN.
        encoding: The profile's affine encoding.

    Returns:
        Reflectance, float32, same shape.
    """
    values = np.asarray(stored, dtype=np.float64)
    return (encoding.scale * values + encoding.offset).astype(np.float32)


def normalise(reflectance: ArrayLike) -> FloatArray:
    """Clip reflectance to 0 to 1.5 and map it to the model's input scale."""
    clipped = np.clip(np.asarray(reflectance, dtype=np.float32), REFLECTANCE_MIN, REFLECTANCE_MAX)
    return ((clipped - NORMALISATION_OFFSET) / NORMALISATION_SCALE).astype(np.float32)


def denormalise(x: ArrayLike) -> FloatArray:
    """Invert :func:`normalise` for values inside the clip range."""
    values = np.asarray(x, dtype=np.float32)
    return (values * NORMALISATION_SCALE + NORMALISATION_OFFSET).astype(np.float32)


def upsample_band(band: ArrayLike, factor: int) -> NDArray[np.generic]:
    """Repeat every pixel ``factor`` times along both axes (nearest neighbour).

    Args:
        band: (H, W) raster, any dtype.
        factor: Integer factor, at least 1; for example 2 for 20 m to 10 m.

    Returns:
        (H * factor, W * factor) raster of the same dtype.
    """
    if factor < 1:
        raise ValueError(f"factor must be at least 1, got {factor}")
    array = np.asarray(band)
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def downsample_band(band: ArrayLike, factor: int) -> FloatArray:
    """Average non-overlapping ``factor`` x ``factor`` blocks.

    Args:
        band: (H, W) raster with H and W divisible by ``factor``.
        factor: Integer factor, at least 1.

    Returns:
        (H / factor, W / factor) float32 raster with the same mean.
    """
    array = np.asarray(band, dtype=np.float64)
    height, width = array.shape
    if factor < 1 or height % factor or width % factor:
        raise ValueError(f"cannot downsample {height}x{width} by {factor}")
    blocks = array.reshape(height // factor, factor, width // factor, factor)
    return np.asarray(blocks.mean(axis=(1, 3)), dtype=np.float32)


@dataclass(frozen=True)
class Window:
    """A square tile window in pixel coordinates.

    Attributes:
        row: Top row.
        col: Left column.
        size: Edge length in pixels.
    """

    row: int
    col: int
    size: int

    def slices(self) -> tuple[slice, slice]:
        """Return the row and column slices of the window."""
        return slice(self.row, self.row + self.size), slice(self.col, self.col + self.size)


def _starts(length: int, size: int, step: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, step))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def tile_windows(height: int, width: int, size: int = TILE_SIZE, overlap: int = 0) -> list[Window]:
    """Return windows that cover an image completely (MOD-REQ-100).

    Windows advance by ``size - overlap``; the last window in each direction
    is moved back to end at the image edge instead of leaving the image.
    Images smaller than a tile get one window at the origin and must be
    padded with :func:`pad_to_tile`.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        size: Tile edge in pixels.
        overlap: Overlap between neighbouring windows in pixels, below ``size``.

    Returns:
        Windows in row-major order.
    """
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be from 0 to below {size}, got {overlap}")
    if height < 1 or width < 1:
        raise ValueError(f"image must not be empty, got {height}x{width}")
    step = size - overlap
    return [
        Window(row, col, size)
        for row in _starts(height, size, step)
        for col in _starts(width, size, step)
    ]


def pad_to_tile(
    array: NDArray[np.generic], size: int = TILE_SIZE, fill: float = 0
) -> tuple[NDArray[np.generic], NDArray[np.bool_]]:
    """Pad the last two axes of an array to at least ``size``.

    Args:
        array: (..., H, W) array.
        size: Minimum edge length.
        fill: Value written into the padding.

    Returns:
        The padded array and a (H', W') mask that is True on original pixels.
    """
    height, width = array.shape[-2:]
    pad_h, pad_w = max(0, size - height), max(0, size - width)
    widths = [(0, 0)] * (array.ndim - 2) + [(0, pad_h), (0, pad_w)]
    padded = np.pad(array, widths, constant_values=fill)
    valid = np.zeros(padded.shape[-2:], dtype=bool)
    valid[:height, :width] = True
    return padded, valid


def extract(array: NDArray[np.generic], window: Window) -> NDArray[np.generic]:
    """Cut a window out of the last two axes of an array."""
    rows, cols = window.slices()
    return array[..., rows, cols]


def stitch(
    tiles: Sequence[ArrayLike], windows: Sequence[Window], height: int, width: int
) -> FloatArray:
    """Reassemble tile predictions into an image, averaging overlaps.

    Args:
        tiles: (..., size, size) predictions, one per window.
        windows: The windows the tiles were cut from.
        height: Image height.
        width: Image width.

    Returns:
        (..., height, width) float32 image; pixels outside every window are 0.
    """
    if len(tiles) != len(windows):
        raise ValueError(f"{len(tiles)} tiles for {len(windows)} windows")
    first = np.asarray(tiles[0])
    total = np.zeros((*first.shape[:-2], height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float64)
    for tile, window in zip(tiles, windows, strict=True):
        rows, cols = window.slices()
        values = np.asarray(tile, dtype=np.float64)
        visible = values[
            ..., : min(window.size, height - window.row), : min(window.size, width - window.col)
        ]
        total[..., rows, cols] += visible
        count[rows, cols] += 1.0
    return (total / np.maximum(count, 1.0)).astype(np.float32)
