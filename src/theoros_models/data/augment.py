# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Augmentations that respect D4 and the band metadata (MOD-REQ-107, MOD-REQ-108).

Geometric augmentation applies one symmetry of the square to the image, to
every dense label and to the vessel centres, using the same convention as
:func:`theoros_models.layers.d4.transform`. Spectral augmentation perturbs
reflectance the way calibration error and sensor noise would. Band dropout
removes bands so that the stem learns to read band subsets; the metadata
always lists exactly the bands left in the image.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from theoros_models.data.sample import DENSE_TASKS, Sample
from theoros_models.sensors.schema import MIN_BANDS

GROUP_ORDER = 8


def transform_array(array: NDArray[np.generic], g: int) -> NDArray[np.generic]:
    """Apply group element ``g`` to the last two axes of a numpy array.

    Matches :func:`theoros_models.layers.d4.transform`: mirror the width axis
    if ``g >= 4``, then rotate ``g mod 4`` quarter turns counter-clockwise.
    """
    mirror, turns = divmod(g, 4)
    if mirror:
        array = array[..., ::-1]
    return np.ascontiguousarray(np.rot90(array, turns, axes=(-2, -1)))


def transform_points(points: NDArray[np.float32], g: int, size: int) -> NDArray[np.float32]:
    """Apply group element ``g`` to (row, col) points in pixel coordinates.

    Args:
        points: (K, 2+) array whose first two columns are row and column;
            further columns are copied unchanged.
        g: Group element index.
        size: Tile edge in pixels.

    Returns:
        Transformed copy, same shape and dtype.
    """
    out = points.copy()
    mirror, turns = divmod(g, 4)
    rows, cols = out[:, 0].copy(), out[:, 1].copy()
    if mirror:
        cols = size - cols
    for _ in range(turns):
        rows, cols = size - cols, rows
    out[:, 0], out[:, 1] = rows, cols
    return out


def apply_symmetry(sample: Sample, g: int) -> Sample:
    """Return the sample with image, labels and vessels transformed by ``g``."""
    if not 0 <= g < GROUP_ORDER:
        raise ValueError(f"group element must be 0 to 7, got {g}")
    changes: dict[str, object] = {"image": transform_array(sample.image, g)}
    for task in DENSE_TASKS:
        label = getattr(sample, task)
        if label is not None:
            changes[task] = transform_array(label, g)
    if sample.vessels is not None:
        changes["vessels"] = transform_points(sample.vessels, g, sample.size)
    return dataclasses.replace(sample, **changes)  # type: ignore[arg-type]


def spectral_jitter(
    sample: Sample,
    rng: np.random.Generator,
    gain_sigma: float = 0.03,
    offset_sigma: float = 0.005,
    noise_sigma: float = 0.003,
) -> Sample:
    """Perturb reflectance per band with gain, offset and pixel noise.

    Args:
        sample: Input sample.
        rng: Random generator.
        gain_sigma: Standard deviation of the per-band relative gain.
        offset_sigma: Standard deviation of the per-band additive offset (reflectance).
        noise_sigma: Standard deviation of independent pixel noise (reflectance).

    Returns:
        The perturbed sample; labels are unchanged.
    """
    count = len(sample.bands)
    gain = 1.0 + rng.normal(0.0, gain_sigma, size=(count, 1, 1))
    offset = rng.normal(0.0, offset_sigma, size=(count, 1, 1))
    noise = rng.normal(0.0, noise_sigma, size=sample.image.shape)
    image = (sample.image * gain + offset + noise).astype(np.float32)
    return dataclasses.replace(sample, image=image)


def band_dropout(sample: Sample, rng: np.random.Generator, keep_min: int = MIN_BANDS) -> Sample:
    """Keep a random subset of at least ``keep_min`` bands, in their original order.

    The number of kept bands is uniform between ``keep_min`` and the band
    count; which bands are kept is uniform among subsets of that size.
    """
    count = len(sample.bands)
    if not MIN_BANDS <= keep_min <= count:
        raise ValueError(f"keep_min must be from {MIN_BANDS} to {count}, got {keep_min}")
    kept = np.sort(rng.choice(count, size=int(rng.integers(keep_min, count + 1)), replace=False))
    return dataclasses.replace(
        sample,
        image=np.ascontiguousarray(sample.image[kept]),
        bands=tuple(sample.bands[i] for i in kept),
    )


@dataclass(frozen=True)
class Augmenter:
    """Random geometric and spectral augmentation.

    Attributes:
        symmetry: Apply a uniformly random symmetry of the square.
        jitter: Apply :func:`spectral_jitter` with default strengths.
        dropout_probability: Probability of applying :func:`band_dropout`.
    """

    symmetry: bool = True
    jitter: bool = True
    dropout_probability: float = 0.25

    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample:
        """Return an augmented copy of ``sample``."""
        if self.symmetry:
            sample = apply_symmetry(sample, int(rng.integers(GROUP_ORDER)))
        if self.jitter:
            sample = spectral_jitter(sample, rng)
        if len(sample.bands) > MIN_BANDS and rng.random() < self.dropout_probability:
            sample = band_dropout(sample, rng)
        return sample
