# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for spectral response handling (MOD-REQ-026 to MOD-REQ-028)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from tests.strategies import profile_dict
from theoros_models.sensors import srf
from theoros_models.sensors.schema import Band, SensorProfile


def _gaussian_table(centre: float, fwhm: float, step: float = 0.5) -> dict[str, Any]:
    x = np.arange(centre - 2.0 * fwhm, centre + 2.0 * fwhm + step / 2.0, step)
    y = srf.gaussian_response(centre, fwhm, x)
    return {"wavelengths_nm": x.tolist(), "response": (0.8 * y).tolist()}


def _band(**changes: Any) -> dict[str, Any]:
    band: dict[str, Any] = {"name": "red", "centre_nm": 665.0, "fwhm_nm": 30.0, "gsd_m": 10.0}
    band.update(changes)
    return band


def test_wavelength_grid() -> None:
    grid = srf.wavelength_grid()
    assert grid.shape == (2151,)
    assert grid[0] == 350.0
    assert grid[-1] == 2500.0
    np.testing.assert_array_equal(np.diff(grid), 1.0)


def test_trapezoid_weights_integrate_exactly() -> None:
    grid = np.array([0.0, 1.0, 3.0, 6.0])
    weights = srf.trapezoid_weights(grid)
    np.testing.assert_allclose(weights, [0.5, 1.5, 2.5, 1.5])
    assert np.sum(weights * (2.0 * grid + 1.0)) == pytest.approx(42.0)


def test_gaussian_response_shape() -> None:
    grid = np.array([650.0, 665.0, 680.0])
    np.testing.assert_allclose(srf.gaussian_response(665.0, 30.0, grid), [0.5, 1.0, 0.5])


def test_gaussian_measured_fwhm() -> None:
    table = _gaussian_table(1610.0, 90.0, step=0.25)
    assert srf.measured_fwhm_nm(table["wavelengths_nm"], table["response"]) == pytest.approx(
        90.0, rel=1e-4
    )
    assert srf.centroid_nm(table["wavelengths_nm"], table["response"]) == pytest.approx(1610.0)


def test_measured_fwhm_of_truncated_table() -> None:
    x = [600.0, 610.0, 620.0]
    assert srf.measured_fwhm_nm(x, [1.0, 1.0, 0.0]) == pytest.approx(15.0)
    assert srf.measured_fwhm_nm(x, [0.0, 1.0, 1.0]) == pytest.approx(15.0)
    assert srf.measured_fwhm_nm(x, [1.0, 1.0, 1.0]) == pytest.approx(20.0)


def test_tabulated_response_resampling() -> None:
    grid = np.array([590.0, 600.0, 605.0, 610.0, 630.0])
    resampled = srf.tabulated_response([600.0, 610.0, 620.0], [0.0, 2.0, 0.0], grid)
    np.testing.assert_allclose(resampled, [0.0, 0.0, 0.5, 1.0, 0.0])


def test_band_response_uses_table_when_present() -> None:
    table = {"wavelengths_nm": [640.0, 665.0, 690.0], "response": [0.0, 1.0, 0.0]}
    tabulated = Band.model_validate(_band(srf=table))
    gaussian = Band.model_validate(_band())
    grid = np.array([652.5, 665.0])
    np.testing.assert_allclose(srf.band_response(tabulated, grid), [0.5, 1.0])
    # 12.5 nm from the centre of a 30 nm FWHM Gaussian: 2^-((2 * 12.5 / 30)^2).
    expected = [2.0 ** -((2.0 * 12.5 / 30.0) ** 2), 1.0]
    np.testing.assert_allclose(srf.band_response(gaussian, grid), expected)
    assert srf.band_response(gaussian).shape == (2151,)


def test_consistent_srf_accepted() -> None:
    band = Band.model_validate(_band(srf=_gaussian_table(665.0, 30.0)))
    assert band.srf is not None
    assert len(band.srf.response) == len(band.srf.wavelengths_nm)


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (_gaussian_table(675.0, 30.0), "centroid"),
        (_gaussian_table(665.0, 40.0), "FWHM"),
        (_gaussian_table(665.0, 21.0), "FWHM"),
        ({"wavelengths_nm": [650.0, 665.0], "response": [1.0, 1.0]}, "at least 3"),
        ({"wavelengths_nm": [650.0, 665.0, 680.0], "response": [1.0, 1.0]}, "samples"),
        ({"wavelengths_nm": [650.0, 650.0, 680.0], "response": [0.0, 1.0, 0.0]}, "increasing"),
        ({"wavelengths_nm": [250.0, 665.0, 680.0], "response": [0.0, 1.0, 0.0]}, "within"),
        ({"wavelengths_nm": [650.0, 665.0, 2700.0], "response": [0.0, 1.0, 0.0]}, "within"),
        ({"wavelengths_nm": [650.0, 665.0, 680.0], "response": [-0.1, 1.0, 0.0]}, "non-negative"),
        ({"wavelengths_nm": [650.0, 665.0, 680.0], "response": [0.0, 0.0, 0.0]}, "positive peak"),
    ],
)
def test_inconsistent_srf_rejected(table: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Band.model_validate(_band(srf=table))


def test_response_matrix_and_linear_spectrum() -> None:
    profile = SensorProfile.model_validate(
        profile_dict(
            [
                _band(name="blue", centre_nm=490.0, fwhm_nm=60.0),
                _band(),
                _band(name="swir", centre_nm=1610.0, fwhm_nm=90.0),
            ]
        )
    )
    grid = srf.wavelength_grid()
    responses = srf.response_matrix(profile)
    assert responses.shape == (3, grid.size)
    np.testing.assert_allclose(responses.max(axis=1), 1.0)
    # A spectrum linear in wavelength maps to its value at the centre of a
    # symmetric response; the grid truncates the blue tail beyond 5 sigma.
    values = srf.band_equivalent(1.0e-3 * grid, responses)
    np.testing.assert_allclose(values, [0.490, 0.665, 1.610], rtol=1e-8)
    batch = srf.band_equivalent(np.ones((4, 2, grid.size)), responses, grid)
    assert batch.shape == (4, 2, 3)
