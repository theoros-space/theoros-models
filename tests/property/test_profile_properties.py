# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Property tests for profile validation (MOD-REQ-021 to MOD-REQ-025, MOD-REQ-029)."""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import ValidationError

from tests.strategies import disjoint_bands, profile_dict, valid_profiles
from theoros_models.sensors import SensorProfile, parse_profile
from theoros_models.sensors.schema import MAX_BAND_IOU, Band, interval_iou


@given(valid_profiles())
def test_valid_profiles_accepted(data: dict[str, Any]) -> None:
    profile = SensorProfile.model_validate(data)
    assert 3 <= len(profile.bands) <= 13
    assert parse_profile(yaml.safe_dump(data)).fingerprint() == profile.fingerprint()


@given(st.integers(min_value=0, max_value=16).flatmap(disjoint_bands))
def test_band_count_bounds(bands: list[dict[str, Any]]) -> None:
    data = profile_dict(bands)
    if 3 <= len(bands) <= 13:
        assert len(SensorProfile.model_validate(data).bands) == len(bands)
    else:
        with pytest.raises(ValidationError, match="3 to 13 bands"):
            SensorProfile.model_validate(data)


@given(
    valid_profiles(),
    st.data(),
    st.one_of(
        st.floats(min_value=0.0, max_value=399.9),
        st.floats(min_value=2400.1, max_value=5000.0),
    ),
)
def test_band_outside_range_rejected(
    data: dict[str, Any], draw: st.DataObject, edge: float
) -> None:
    index = draw.draw(st.integers(min_value=0, max_value=len(data["bands"]) - 1))
    band = data["bands"][index]
    # Move one half-power edge outside the range while keeping the centre.
    band["fwhm_nm"] = 2.0 * abs(band["centre_nm"] - edge)
    with pytest.raises(ValidationError):
        SensorProfile.model_validate(data)


@given(valid_profiles(), st.floats(min_value=-10.0, max_value=0.0))
def test_non_positive_fwhm_rejected(data: dict[str, Any], fwhm: float) -> None:
    data["bands"][0]["fwhm_nm"] = fwhm
    with pytest.raises(ValidationError, match="greater than 0"):
        SensorProfile.model_validate(data)


@given(valid_profiles(), st.data())
def test_unsorted_bands_rejected(data: dict[str, Any], draw: st.DataObject) -> None:
    bands = data["bands"]
    order = draw.draw(st.permutations(range(len(bands))).filter(lambda p: list(p) != sorted(p)))
    data["bands"] = [bands[i] for i in order]
    with pytest.raises(ValidationError, match="strictly increasing centre wavelength"):
        SensorProfile.model_validate(data)


@given(
    st.integers(min_value=5000, max_value=20000),
    st.integers(min_value=10, max_value=2000),
    st.integers(min_value=1, max_value=4000),
    st.integers(min_value=10, max_value=2000),
)
def test_near_duplicate_rejected(centre_a: int, width_a: int, offset: int, width_b: int) -> None:
    # Two bands on a 0.1 nm lattice with increasing centres; a third band far
    # in the SWIR completes the minimum band count without overlapping them.
    a = Band(name="a", centre_nm=centre_a / 10, fwhm_nm=width_a / 10, gsd_m=10.0)
    b_centre, b_width = (centre_a + offset) / 10, width_b / 10
    assume(b_centre + b_width / 2 < 2340.0)
    b = Band(name="b", centre_nm=b_centre, fwhm_nm=b_width, gsd_m=10.0)
    far = Band(name="far", centre_nm=2350.0, fwhm_nm=10.0, gsd_m=10.0)
    data = profile_dict([band.model_dump() for band in (a, b, far)])
    if interval_iou(a, b) > MAX_BAND_IOU:
        with pytest.raises(ValidationError, match="near-duplicates"):
            SensorProfile.model_validate(data)
    else:
        assert SensorProfile.model_validate(data).band_names == ("a", "b", "far")


@given(
    valid_profiles(),
    st.one_of(
        st.floats(max_value=0.0999, allow_nan=False),
        st.floats(min_value=1000.001, allow_nan=False),
    ),
)
def test_gsd_out_of_range_rejected(data: dict[str, Any], gsd: float) -> None:
    data["bands"][-1]["gsd_m"] = gsd
    with pytest.raises(ValidationError):
        SensorProfile.model_validate(data)


@given(valid_profiles(), st.data())
def test_fingerprint_sensitivity(data: dict[str, Any], draw: st.DataObject) -> None:
    reference = SensorProfile.model_validate(data).fingerprint()
    index = draw.draw(st.integers(min_value=0, max_value=len(data["bands"]) - 1))
    field = draw.draw(st.sampled_from(["name", "gsd_m", "fwhm_nm", "scale", "offset", "platform"]))
    if field == "name":
        data["bands"][index]["name"] = "renamed"
    elif field == "gsd_m":
        gsd = data["bands"][index]["gsd_m"]
        data["bands"][index]["gsd_m"] = gsd / 2.0 if gsd > 1.0 else gsd * 2.0
    elif field == "fwhm_nm":
        data["bands"][index]["fwhm_nm"] *= 0.5
    elif field in {"scale", "offset"}:
        data["encoding"][field] = data["encoding"][field] + 0.5
    else:
        data["platform"] = "Other"
    assert SensorProfile.model_validate(data).fingerprint() != reference
