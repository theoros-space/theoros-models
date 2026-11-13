# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Property tests for band-equivalent reflectance (MOD-REQ-028)."""

from __future__ import annotations

from typing import Any

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from tests.strategies import valid_profiles
from theoros_models.sensors import SensorProfile, srf

GRID = srf.wavelength_grid()
_REFLECTANCE = st.floats(min_value=0.0, max_value=1.5, allow_nan=False)
SPECTRA = hnp.arrays(np.float64, (GRID.size,), elements=_REFLECTANCE)


@given(valid_profiles(), _REFLECTANCE)
def test_constant_spectrum_maps_to_constant(data: dict[str, Any], value: float) -> None:
    responses = srf.response_matrix(SensorProfile.model_validate(data))
    out = srf.band_equivalent(np.full(GRID.size, value), responses)
    np.testing.assert_allclose(out, value, rtol=1e-12, atol=1e-15)


@given(valid_profiles(), SPECTRA)
def test_bounded_by_spectrum(data: dict[str, Any], spectrum: np.ndarray) -> None:
    responses = srf.response_matrix(SensorProfile.model_validate(data))
    out = srf.band_equivalent(spectrum, responses)
    assert np.all(out >= spectrum.min() - 1e-12)
    assert np.all(out <= spectrum.max() + 1e-12)


@given(valid_profiles(), SPECTRA, SPECTRA, _REFLECTANCE)
def test_linear_in_spectrum(
    data: dict[str, Any], first: np.ndarray, second: np.ndarray, weight: float
) -> None:
    responses = srf.response_matrix(SensorProfile.model_validate(data))
    combined = srf.band_equivalent(first + weight * second, responses)
    separate = srf.band_equivalent(first, responses) + weight * srf.band_equivalent(
        second, responses
    )
    np.testing.assert_allclose(combined, separate, rtol=1e-10, atol=1e-12)
