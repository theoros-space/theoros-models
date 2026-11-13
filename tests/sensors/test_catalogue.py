# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for the packaged catalogue and profile loading (MOD-REQ-020, MOD-REQ-031)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

from theoros_models.sensors import (
    ProfileError,
    catalogue_names,
    load_catalogue_profile,
    load_profile,
    resolve_profile,
)

EXPECTED = (
    "generic_cubesat_4band",
    "landsat8_oli",
    "landsat9_oli2",
    "planetscope_superdove",
    "sentinel2_msi",
    "test_5band",
)
NOT_REAL = ("generic_cubesat_4band", "test_5band")


def test_catalogue_contents() -> None:
    assert catalogue_names() == EXPECTED


@pytest.mark.parametrize("name", EXPECTED)
def test_catalogue_profile_valid(name: str) -> None:
    profile = load_catalogue_profile(name)
    assert profile.name == name
    assert profile.reference.strip()
    if name in NOT_REAL:
        assert "not a real instrument" in profile.reference


def test_band_counts() -> None:
    counts = {name: len(load_catalogue_profile(name).bands) for name in EXPECTED}
    assert counts == {
        "generic_cubesat_4band": 4,
        "landsat8_oli": 8,
        "landsat9_oli2": 8,
        "planetscope_superdove": 8,
        "sentinel2_msi": 13,
        "test_5band": 5,
    }


def test_fingerprints_distinct() -> None:
    fingerprints = {load_catalogue_profile(name).fingerprint() for name in EXPECTED}
    assert len(fingerprints) == len(EXPECTED)


def test_unknown_catalogue_name() -> None:
    with pytest.raises(ProfileError, match="known: generic_cubesat_4band"):
        load_catalogue_profile("sentinel3_olci")


def test_load_profile_from_file(tmp_path: Path) -> None:
    source = resources.files("theoros_models.sensors") / "catalogue" / "test_5band.yaml"
    path = tmp_path / "custom.yaml"
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    assert load_profile(path) == load_catalogue_profile("test_5band")
    assert resolve_profile(str(path)) == resolve_profile("test_5band")


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ProfileError, match="cannot read file"):
        load_profile(tmp_path / "absent.yaml")


@pytest.mark.parametrize(
    ("text", "message"),
    [("bands: [unclosed", "invalid YAML"), ("- a\n- b\n", "YAML mapping"), ("", "YAML mapping")],
)
def test_malformed_yaml(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProfileError, match=message) as caught:
        load_profile(path)
    assert caught.value.source == str(path)
