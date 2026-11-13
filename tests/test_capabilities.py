# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for capability gating (MOD-REQ-004, MOD-REQ-065)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from theoros_models.capabilities import NEEDS, Window, assess_bands, assess_profile
from theoros_models.layers.heads import HEAD_NAMES
from theoros_models.sensors import load_catalogue_profile


@dataclass(frozen=True)
class _Band:
    name: str
    centre_nm: float
    gsd_m: float


def test_table_covers_every_head_in_order() -> None:
    assert tuple(NEEDS) == HEAD_NAMES
    for needs in NEEDS.values():
        for need in needs:
            assert need.windows
            assert need.reason


@pytest.mark.parametrize(
    ("profile", "enabled"),
    [
        ("sentinel2_msi", ("cloud", "land_water", "vessel", "fire")),
        ("landsat8_oli", ("cloud", "land_water", "fire")),
        ("landsat9_oli2", ("cloud", "land_water", "fire")),
        ("planetscope_superdove", ("cloud", "land_water", "vessel")),
        ("generic_cubesat_4band", ("cloud", "land_water", "vessel")),
        ("test_5band", ("cloud", "vessel", "fire")),
    ],
)
def test_catalogue_capabilities(profile: str, enabled: tuple[str, ...]) -> None:
    assert assess_profile(load_catalogue_profile(profile)).enabled == enabled


def test_disabled_heads_are_explained() -> None:
    report = assess_profile(load_catalogue_profile("planetscope_superdove"))
    assert set(report.disabled) == {"fire"}
    missing = report.disabled["fire"]
    assert len(missing) == 2
    assert missing[0].startswith("SWIR 2.2 um (2080 to 2350 nm): thermal emission")
    assert missing[1].startswith("SWIR 1.6 um (1550 to 1750 nm)")
    text = report.format()
    assert "fire        disabled" in text
    assert "needs SWIR 2.2 um" in text
    assert "cloud       enabled   bands coastal_blue, red" in text


def test_alternative_windows() -> None:
    nir_only = [_Band("g", 560.0, 10.0), _Band("n", 865.0, 10.0), _Band("b", 490.0, 10.0)]
    swir_only = [_Band("g", 560.0, 10.0), _Band("s", 1610.0, 20.0), _Band("b", 490.0, 10.0)]
    assert "land_water" in assess_bands(nir_only).enabled
    assert "land_water" in assess_bands(swir_only).enabled
    report = assess_bands(swir_only)
    land = next(head for head in report.heads if head.head == "land_water")
    assert land.satisfied_by == ("g", "s")


def test_gsd_limit_for_vessels() -> None:
    fine = [_Band("n", 842.0, 10.0)]
    coarse = [_Band("n", 842.0, 10.5)]
    assert "vessel" in assess_bands(fine).enabled
    assert "vessel" not in assess_bands(coarse).enabled
    assert "GSD at most 10 m" in assess_bands(coarse).disabled["vessel"][0]


@pytest.mark.parametrize(
    ("centre", "gsd", "accepted"),
    [
        (760.0, 1.0, True),
        (900.0, 1.0, True),
        (759.9, 1.0, False),
        (900.1, 1.0, False),
        (800.0, 10.0, True),
        (800.0, 10.01, False),
    ],
)
def test_window_bounds_inclusive(centre: float, gsd: float, accepted: bool) -> None:
    assert Window("NIR", 760.0, 900.0, max_gsd_m=10.0).accepts(centre, gsd) is accepted


def test_empty_band_set_disables_everything() -> None:
    report = assess_bands([])
    assert report.enabled == ()
    assert set(report.disabled) == set(HEAD_NAMES)
