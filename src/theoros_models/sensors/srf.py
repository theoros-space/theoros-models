# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Spectral response functions: measurement, resampling and band averaging.

All wavelengths are in nanometres. Responses are dimensionless and, once
resampled, normalised to a peak of 1. Spectra are sampled on the common grid
returned by :func:`wavelength_grid` (MOD-REQ-028).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from theoros_models.sensors.schema import Band, SensorProfile

FloatArray = NDArray[np.float64]

GRID_START_NM = 350.0
GRID_STOP_NM = 2500.0
GRID_STEP_NM = 1.0

# A Gaussian with this FWHM has standard deviation FWHM / (2 sqrt(2 ln 2)).
_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def wavelength_grid() -> FloatArray:
    """Return the common wavelength grid.

    Returns:
        Wavelengths from 350 to 2500 nm inclusive at 1 nm spacing, shape (2151,).
    """
    count = round((GRID_STOP_NM - GRID_START_NM) / GRID_STEP_NM) + 1
    return GRID_START_NM + GRID_STEP_NM * np.arange(count, dtype=np.float64)


def trapezoid_weights(grid: FloatArray) -> FloatArray:
    """Return trapezoidal quadrature weights for a strictly increasing grid.

    Args:
        grid: Sample positions, shape (L,), L >= 2.

    Returns:
        Weights w such that sum(w * f) is the trapezoidal integral of f, shape (L,).
    """
    steps = np.diff(grid)
    weights = np.zeros_like(grid)
    weights[:-1] += steps / 2.0
    weights[1:] += steps / 2.0
    return weights


def gaussian_response(centre_nm: float, fwhm_nm: float, grid: FloatArray) -> FloatArray:
    """Evaluate a peak-normalised Gaussian response (MOD-REQ-027).

    Args:
        centre_nm: Centre wavelength in nm.
        fwhm_nm: Full width at half maximum in nm, positive.
        grid: Wavelengths in nm, shape (L,).

    Returns:
        Response at each grid point, peak 1 at the centre, shape (L,).
    """
    sigma = fwhm_nm * _FWHM_TO_SIGMA
    return np.exp(-0.5 * ((grid - centre_nm) / sigma) ** 2)


def tabulated_response(
    wavelengths_nm: ArrayLike, response: ArrayLike, grid: FloatArray
) -> FloatArray:
    """Resample a tabulated response onto a grid by linear interpolation.

    The response is zero outside the tabulated wavelength span and is
    normalised so that its largest tabulated value maps to 1.

    Args:
        wavelengths_nm: Strictly increasing sample wavelengths in nm, shape (S,).
        response: Non-negative responses with a positive peak, shape (S,).
        grid: Target wavelengths in nm, shape (L,).

    Returns:
        Resampled response, shape (L,).
    """
    x = np.asarray(wavelengths_nm, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    resampled = np.interp(grid, x, y / y.max(), left=0.0, right=0.0)
    return np.asarray(resampled, dtype=np.float64)


def centroid_nm(wavelengths_nm: ArrayLike, response: ArrayLike) -> float:
    """Return the response-weighted mean wavelength of a tabulated response.

    Args:
        wavelengths_nm: Strictly increasing sample wavelengths in nm, shape (S,).
        response: Non-negative responses with a positive integral, shape (S,).

    Returns:
        Centroid in nm, using trapezoidal integration.
    """
    x = np.asarray(wavelengths_nm, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    w = trapezoid_weights(x) * y
    return float(np.sum(w * x) / np.sum(w))


def measured_fwhm_nm(wavelengths_nm: ArrayLike, response: ArrayLike) -> float:
    """Measure the full width at half maximum of a tabulated response.

    The width runs between the outermost half-maximum crossings, located by
    linear interpolation between neighbouring samples. A response that is
    still above half maximum at the first or last sample is measured to that
    sample, which under-estimates the width of a truncated table.

    Args:
        wavelengths_nm: Strictly increasing sample wavelengths in nm, shape (S,).
        response: Non-negative responses with a positive peak, shape (S,).

    Returns:
        FWHM in nm.
    """
    x = np.asarray(wavelengths_nm, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    half = y.max() / 2.0
    above = np.flatnonzero(y >= half)
    first, last = int(above[0]), int(above[-1])
    left = x[first]
    if first > 0:
        x0, x1, y0, y1 = x[first - 1], x[first], y[first - 1], y[first]
        left = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
    right = x[last]
    if last < x.size - 1:
        x0, x1, y0, y1 = x[last], x[last + 1], y[last], y[last + 1]
        right = x0 + (y0 - half) * (x1 - x0) / (y0 - y1)
    return float(right - left)


def band_response(band: Band, grid: FloatArray | None = None) -> FloatArray:
    """Return the peak-normalised response of one band on a grid.

    Args:
        band: Band definition; its tabulated SRF is used when present, otherwise
            the Gaussian model of MOD-REQ-027.
        grid: Wavelengths in nm, shape (L,); defaults to :func:`wavelength_grid`.

    Returns:
        Response, shape (L,).
    """
    grid = wavelength_grid() if grid is None else grid
    if band.srf is None:
        return gaussian_response(band.centre_nm, band.fwhm_nm, grid)
    return tabulated_response(band.srf.wavelengths_nm, band.srf.response, grid)


def response_matrix(profile: SensorProfile, grid: FloatArray | None = None) -> FloatArray:
    """Stack the responses of every band of a profile.

    Args:
        profile: Sensor profile with N bands.
        grid: Wavelengths in nm, shape (L,); defaults to :func:`wavelength_grid`.

    Returns:
        Responses in profile band order, shape (N, L).
    """
    grid = wavelength_grid() if grid is None else grid
    return np.stack([band_response(band, grid) for band in profile.bands])


def band_equivalent(
    spectra: ArrayLike, responses: FloatArray, grid: FloatArray | None = None
) -> FloatArray:
    """Compute band-equivalent values of spectra (MOD-REQ-028).

    Each output is the response-weighted mean of the spectrum, integrated with
    the trapezoidal rule. Because the weights are non-negative, the result lies
    between the minimum and maximum of the spectrum over the band's support.

    Args:
        spectra: Spectra sampled on ``grid``, shape (..., L), any unit.
        responses: Band responses on the same grid, shape (N, L), each with a
            positive integral.
        grid: Wavelengths in nm, shape (L,); defaults to :func:`wavelength_grid`.

    Returns:
        Band-equivalent values in the unit of ``spectra``, shape (..., N).
    """
    grid = wavelength_grid() if grid is None else grid
    weights = responses * trapezoid_weights(grid)  # (N, L)
    weights = weights / weights.sum(axis=1, keepdims=True)
    values = np.asarray(spectra, dtype=np.float64) @ weights.T
    return np.asarray(values, dtype=np.float64)
