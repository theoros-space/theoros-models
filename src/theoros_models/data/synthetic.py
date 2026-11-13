# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Synthetic multi-sensor scenes for tests and pipeline checks (MOD-REQ-106).

Scenes are built from a few analytic material spectra: water, vegetation,
soil, cloud, vessel hull and active fire. The spectra are smooth
approximations chosen for their qualitative features (the red edge of
vegetation, the near-infrared absorption of water, the shortwave emission of
fire); they are not measured library spectra and synthetic scenes carry no
information about real-world accuracy.

Each material is integrated over every band's spectral response
(MOD-REQ-028), so the same scene renders consistently for any sensor.
Pixels are linear mixtures of materials; thin cloud blends over the surface,
shadow darkens it and thick cloud replaces it. Labels follow from the layout
by construction. Everything is a deterministic function of the seed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from theoros_models.data.sample import IGNORE, Sample
from theoros_models.layers.heads import HEAD_NAMES
from theoros_models.sensors import srf
from theoros_models.sensors.schema import Band, SensorProfile

FloatArray = NDArray[np.float64]

MATERIALS = ("water", "vegetation", "soil", "cloud", "vessel", "fire")
NOISE_REFLECTANCE = 0.004
SHADOW_FACTOR = 0.35
THIN_CLOUD_ALPHA = 0.45
VEGETATION_FRACTION = 0.55


def _gauss(grid: FloatArray, centre: float, width: float) -> FloatArray:
    return np.asarray(np.exp(-0.5 * ((grid - centre) / width) ** 2), dtype=np.float64)


def _sigmoid(grid: FloatArray, centre: float, width: float) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-(grid - centre) / width)), dtype=np.float64)


def material_spectra(grid: FloatArray | None = None) -> FloatArray:
    """Return the analytic material spectra as reflectance.

    Args:
        grid: Wavelengths in nm, shape (L,); defaults to the common grid.

    Returns:
        Reflectance of each material in ``MATERIALS`` order, shape (6, L).
    """
    g = srf.wavelength_grid() if grid is None else grid
    moisture = 0.12 * _gauss(g, 1450.0, 60.0) + 0.18 * _gauss(g, 1940.0, 80.0)
    water = 0.07 * np.exp(-(g - 400.0) / 180.0) + 0.004
    vegetation = (
        0.035
        + 0.05 * _gauss(g, 550.0, 30.0)
        + 0.42 * _sigmoid(g, 715.0, 12.0) * (1.0 - 0.55 * _sigmoid(g, 1350.0, 120.0))
        - moisture
    )
    soil = 0.07 + 0.28 * (g - 400.0) / 2000.0 - 0.5 * moisture
    cloud = 0.78 - 0.25 * _sigmoid(g, 1500.0, 150.0) - 0.6 * moisture
    vessel = 0.16 + 0.04 * _sigmoid(g, 700.0, 30.0)
    # Soil plus thermal emission rising steeply through the shortwave infrared.
    fire = soil + 0.9 * np.clip((g - 1250.0) / 1000.0, 0.0, None) ** 2
    spectra = np.stack([water, vegetation, soil, cloud, vessel, fire])
    return np.clip(spectra, 0.002, None)


def material_band_values(bands: Sequence[Band]) -> FloatArray:
    """Return the band-equivalent reflectance of every material, shape (6, N)."""
    grid = srf.wavelength_grid()
    responses = np.stack([srf.band_response(band, grid) for band in bands])
    return srf.band_equivalent(material_spectra(grid), responses, grid)


def smooth_noise(rng: np.random.Generator, size: int, cells: int) -> FloatArray:
    """Return a smooth random field in [0, 1] by bilinear upsampling of coarse noise.

    Args:
        rng: Random generator.
        size: Output edge in pixels.
        cells: Coarse grid cells per edge; larger means finer structure.

    Returns:
        Field of shape (size, size).
    """
    coarse = rng.random((cells + 1, cells + 1))
    position = (np.arange(size) + 0.5) * cells / size
    low = np.minimum(position.astype(int), cells - 1)
    frac = position - low
    rows = coarse[low] * (1.0 - frac)[:, None] + coarse[low + 1] * frac[:, None]
    field = rows[:, low] * (1.0 - frac)[None, :] + rows[:, low + 1] * frac[None, :]
    return np.asarray(field, dtype=np.float64)


def _disc(size: int, row: float, col: float, radius: float) -> NDArray[np.bool_]:
    rr, cc = np.mgrid[0:size, 0:size] + 0.5
    return np.asarray((rr - row) ** 2 + (cc - col) ** 2 <= radius**2, dtype=bool)


def _rectangle(
    size: int, row: float, col: float, extent: tuple[float, float], angle: float
) -> NDArray[np.bool_]:
    length, width = extent
    rr, cc = np.mgrid[0:size, 0:size] + 0.5
    along = (rr - row) * np.sin(angle) + (cc - col) * np.cos(angle)
    across = (rr - row) * np.cos(angle) - (cc - col) * np.sin(angle)
    return np.asarray((np.abs(along) <= length / 2.0) & (np.abs(across) <= width / 2.0), dtype=bool)


def _place_vessels(
    rng: np.random.Generator,
    size: int,
    pixel_m: float,
    *,
    water: NDArray[np.bool_],
    thick: NDArray[np.bool_],
    abundance: FloatArray,
) -> list[tuple[float, float, float, float]]:
    """Draw vessels on open water, paint them into ``abundance`` and return their rows."""
    vessels = []
    for _ in range(rng.poisson(2.0)):
        row, col = rng.uniform(4, size - 4, size=2)
        length_m = rng.uniform(20.0, 200.0)
        width_m = length_m / rng.uniform(5.0, 8.0)
        extent = (length_m / pixel_m, width_m / pixel_m)
        hull = _rectangle(size, row, col, extent, rng.uniform(0, np.pi))
        if not water[int(row), int(col)] or thick[int(row), int(col)]:
            continue
        hull &= water
        abundance[:, hull] = 0.0
        abundance[4, hull] = 1.0
        vessels.append((row, col, length_m, width_m))
    return vessels


def render_scene(
    bands: Sequence[Band],
    sensor: str,
    size: int,
    seed: int,
    *,
    tasks: Sequence[str] = HEAD_NAMES,
    sample_id: str = "",
) -> Sample:
    """Render one synthetic scene for a band set.

    The pixel size is the finest GSD of the band set. The layout depends
    only on ``seed`` and ``size``, so the same seed gives the same scene for
    every sensor up to pixel size.

    Args:
        bands: Bands to render, 3 to 13.
        sensor: Sensor name recorded in the sample.
        size: Tile edge in pixels.
        seed: Seed of the layout and noise.
        tasks: Tasks whose labels are emitted; others are None.
        sample_id: Identifier recorded in the sample.

    Returns:
        The rendered sample with reflectance and labels.
    """
    rng = np.random.default_rng(seed)
    pixel_m = min(band.gsd_m for band in bands)
    water = smooth_noise(rng, size, 3) < rng.uniform(0.2, 0.6)
    vegetated = smooth_noise(rng, size, 6) < VEGETATION_FRACTION
    cloud_field = smooth_noise(rng, size, 4)
    cloud_level = rng.uniform(0.55, 0.95)
    thick = cloud_field > cloud_level
    thin = (cloud_field > cloud_level - 0.12) & ~thick
    shift = rng.integers(-size // 16, size // 16 + 1, size=2)
    shadow = np.roll(thick, tuple(int(s) for s in shift), axis=(0, 1)) & ~thick & ~thin

    abundance = np.zeros((len(MATERIALS), size, size))
    abundance[0] = water
    abundance[1] = ~water & vegetated
    abundance[2] = ~water & ~vegetated

    fire = np.zeros((size, size), dtype=bool)
    for _ in range(rng.poisson(1.0)):
        row, col = rng.uniform(0, size, size=2)
        fire |= _disc(size, row, col, rng.uniform(1.0, 3.0)) & ~water
    abundance[:, fire] = 0.0
    abundance[5, fire] = 1.0

    vessels = _place_vessels(rng, size, pixel_m, water=water, thick=thick, abundance=abundance)

    values = material_band_values(bands)  # (6, N)
    surface = np.einsum("mhw,mn->nhw", abundance, values)
    cloud_values = values[3][:, None, None]
    image = np.where(
        thin, (1 - THIN_CLOUD_ALPHA) * surface + THIN_CLOUD_ALPHA * cloud_values, surface
    )
    image = np.where(shadow, SHADOW_FACTOR * image, image)
    image = np.where(thick, cloud_values, image)
    image = image + rng.normal(0.0, NOISE_REFLECTANCE, size=image.shape)

    cloud_label = np.zeros((size, size), dtype=np.uint8)
    cloud_label[shadow] = 3
    cloud_label[thin] = 2
    cloud_label[thick] = 1
    land_water = water.astype(np.uint8)
    land_water[thick] = IGNORE
    fire_label = fire.astype(np.uint8)
    fire_label[thick] = IGNORE
    vessel_rows = np.asarray(vessels, dtype=np.float32).reshape(-1, 4)
    unknown = sorted(set(tasks) - set(HEAD_NAMES))
    if unknown:
        raise ValueError(f"unknown tasks: {', '.join(unknown)}")
    return Sample(
        image=image.astype(np.float32),
        bands=tuple(bands),
        sensor=sensor,
        sample_id=sample_id,
        cloud=cloud_label if "cloud" in tasks else None,
        land_water=land_water if "land_water" in tasks else None,
        fire=fire_label if "fire" in tasks else None,
        vessels=vessel_rows if "vessel" in tasks else None,
    )


def render_profile_scene(
    profile: SensorProfile, size: int, seed: int, tasks: Sequence[str] = HEAD_NAMES
) -> Sample:
    """Render a scene for every band of a profile; see :func:`render_scene`."""
    return render_scene(
        profile.bands, profile.name, size, seed, tasks=tasks, sample_id=f"{profile.name}-{seed}"
    )
