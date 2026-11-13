# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for the sensor profile schema (MOD-REQ-020, 023, 029, 030)."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from tests.strategies import profile_dict
from theoros_models.sensors import ProfileError, SensorProfile, parse_profile
from theoros_models.sensors.schema import Band, interval_iou


def _bands() -> list[dict[str, Any]]:
    return [
        {"name": "blue", "centre_nm": 490.0, "fwhm_nm": 60.0, "gsd_m": 10.0},
        {"name": "red", "centre_nm": 665.0, "fwhm_nm": 30.0, "gsd_m": 10.0},
        {"name": "nir", "centre_nm": 842.0, "fwhm_nm": 100.0, "gsd_m": 10.0},
    ]


def _valid() -> dict[str, Any]:
    return profile_dict(_bands(), description="three bands", reference="test")


def _problems(data: dict[str, Any]) -> list[str]:
    with pytest.raises(ProfileError) as caught:
        parse_profile(yaml.safe_dump(data))
    return caught.value.problems


def test_valid_profile_parses() -> None:
    profile = parse_profile(yaml.safe_dump(_valid()))
    assert profile.band_names == ("blue", "red", "nir")
    assert profile.bands[0].lower_nm == 460.0
    assert profile.bands[0].upper_nm == 520.0


def test_profile_is_immutable() -> None:
    profile = SensorProfile.model_validate(_valid())
    with pytest.raises(ValidationError):
        profile.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("path", "field"),
    [((), "colour"), (("encoding",), "gain"), (("bands", 1), "polarisation")],
)
def test_unknown_field_rejected(path: tuple[Any, ...], field: str) -> None:
    data = _valid()
    target: Any = data
    for key in path:
        target = target[key]
    target[field] = 1
    problems = _problems(data)
    assert any(field in problem and "Extra inputs" in problem for problem in problems)


@pytest.mark.parametrize("field", ["schema_version", "name", "platform", "encoding", "bands"])
def test_missing_field_rejected(field: str) -> None:
    data = _valid()
    del data[field]
    problems = _problems(data)
    assert any(problem.startswith(f"{field}:") for problem in problems)


def test_missing_band_field_rejected() -> None:
    data = _valid()
    del data["bands"][2]["gsd_m"]
    assert any(problem.startswith("bands.2.gsd_m:") for problem in _problems(data))


@pytest.mark.parametrize(
    ("field", "value"),
    [("centre_nm", "blue"), ("fwhm_nm", [1, 2]), ("gsd_m", None), ("name", 7)],
)
def test_wrong_type_rejected(field: str, value: object) -> None:
    data = _valid()
    data["bands"][0][field] = value
    assert any(problem.startswith(f"bands.0.{field}:") for problem in _problems(data))


@pytest.mark.parametrize("version", [0, 2, "1"])
def test_schema_version_rejected(version: object) -> None:
    data = _valid()
    data["schema_version"] = version
    assert any(problem.startswith("schema_version:") for problem in _problems(data))


@pytest.mark.parametrize("name", ["Sentinel2", "2sat", "has space", ""])
def test_profile_name_pattern(name: str) -> None:
    data = _valid()
    data["name"] = name
    assert any(problem.startswith("name:") for problem in _problems(data))


def test_duplicate_band_name_rejected() -> None:
    data = _valid()
    data["bands"][2]["name"] = "blue"
    problems = _problems(data)
    assert any("duplicate band names: blue" in problem for problem in problems)


@pytest.mark.parametrize(
    "encoding",
    [
        {"scale": 0.0},
        {"scale": -1.0e-4},
        {"scale": float("nan")},
        {"scale": float("inf")},
        {"scale": 1.0e-4, "offset": float("inf")},
        {"offset": 0.0},
    ],
)
def test_encoding_validation(encoding: dict[str, float]) -> None:
    data = _valid()
    data["encoding"] = encoding
    assert any(problem.startswith("encoding") for problem in _problems(data))


def test_encoding_offset_defaults_to_zero() -> None:
    data = _valid()
    data["encoding"] = {"scale": 2.0e-5}
    assert SensorProfile.model_validate(data).encoding.offset == 0.0


def test_non_finite_band_values_rejected() -> None:
    data = _valid()
    data["bands"][1]["centre_nm"] = float("nan")
    assert any(problem.startswith("bands.1.centre_nm:") for problem in _problems(data))


def test_fingerprint_format() -> None:
    fingerprint = SensorProfile.model_validate(_valid()).fingerprint()
    assert fingerprint.startswith("sha256:")
    digest = fingerprint.removeprefix("sha256:")
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_fingerprint_ignores_formatting() -> None:
    data = _valid()
    reference = parse_profile(yaml.safe_dump(data)).fingerprint()

    reordered = {key: data[key] for key in reversed(list(data))}
    reordered["bands"] = [dict(reversed(list(band.items()))) for band in data["bands"]]
    reordered["description"] = "a different description"
    reordered["reference"] = "a different reference"
    flow = yaml.safe_dump(reordered, default_flow_style=True, sort_keys=False)
    assert parse_profile(flow).fingerprint() == reference

    integers = copy.deepcopy(data)
    integers["bands"][0]["centre_nm"] = 490
    assert parse_profile(yaml.safe_dump(integers)).fingerprint() == reference


def test_fingerprint_is_stable() -> None:
    # Model cards and manifests record fingerprints, so the canonical form must
    # not drift between releases. The expected digest is computed from the
    # canonical JSON written out by hand.
    canonical = (
        '{"bands":['
        '{"centre_nm":490.0,"fwhm_nm":60.0,"gsd_m":10.0,"name":"blue","srf":null},'
        '{"centre_nm":665.0,"fwhm_nm":30.0,"gsd_m":10.0,"name":"red","srf":null},'
        '{"centre_nm":842.0,"fwhm_nm":100.0,"gsd_m":10.0,"name":"nir","srf":null}],'
        '"encoding":{"offset":0.0,"scale":0.0001},'
        '"instrument":"Generated","name":"generated","platform":"Test","schema_version":1}'
    )
    expected = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert SensorProfile.model_validate(_valid()).fingerprint() == expected


def _band(centre: float, fwhm: float) -> Band:
    return Band(name="x", centre_nm=centre, fwhm_nm=fwhm, gsd_m=10.0)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((500.0, 20.0), (500.0, 20.0), 1.0),
        ((500.0, 20.0), (520.0, 20.0), 0.0),
        ((500.0, 20.0), (530.0, 20.0), 0.0),
        ((500.0, 20.0), (505.0, 10.0), 0.5),
        ((832.8, 106.0), (864.7, 21.0), 21.0 / 106.0),
    ],
)
def test_interval_iou(a: tuple[float, float], b: tuple[float, float], expected: float) -> None:
    assert interval_iou(_band(*a), _band(*b)) == pytest.approx(expected)
    assert interval_iou(_band(*b), _band(*a)) == pytest.approx(expected)
