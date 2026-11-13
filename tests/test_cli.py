# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for the theoros command line (MOD-REQ-220)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.strategies import profile_dict
from theoros_models import __version__
from theoros_models.cli import main


def _write(path: Path, data: object) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _bands() -> list[dict[str, float | str]]:
    return [
        {"name": "blue", "centre_nm": 490.0, "fwhm_nm": 60.0, "gsd_m": 10.0},
        {"name": "red", "centre_nm": 665.0, "fwhm_nm": 30.0, "gsd_m": 10.0},
        {"name": "nir", "centre_nm": 842.0, "fwhm_nm": 100.0, "gsd_m": 10.0},
    ]


def test_sensors_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sensors", "list"]) == 0
    out = capsys.readouterr().out
    assert "sentinel2_msi" in out
    assert "13" in out


def test_sensors_show_catalogue(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sensors", "show", "landsat8_oli"]) == 0
    out = capsys.readouterr().out
    assert "fingerprint  sha256:" in out
    assert "reflectance = 2e-05 * value - 0.1" in out
    assert "B9" in out


def test_sensors_show_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path / "p.yaml", profile_dict(_bands()))
    assert main(["sensors", "show", str(path)]) == 0
    assert "reflectance = 0.0001 * value + 0" in capsys.readouterr().out


def test_sensors_show_invalid(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sensors", "show", "no_such_profile"]) == 1
    assert "cannot read file" in capsys.readouterr().err


def test_sensors_validate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = _write(tmp_path / "good.yaml", profile_dict(_bands()))
    bad_data = profile_dict(_bands())
    bad_data["bands"][1]["centre_nm"] = 400.0
    bad = _write(tmp_path / "bad.yaml", bad_data)
    assert main(["sensors", "validate", str(good)]) == 0
    assert "valid, 3 bands, sha256:" in capsys.readouterr().out
    assert main(["sensors", "validate", str(good), str(bad)]) == 1
    captured = capsys.readouterr()
    assert f"{good}: valid" in captured.out
    assert f"{bad}: invalid" in captured.err
    assert "bands.1: Value error, band red: half-power interval" in captured.err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["sensors"])
    assert caught.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_sensors_show_capabilities(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sensors", "show", "planetscope_superdove"]) == 0
    out = capsys.readouterr().out
    assert "capabilities" in out
    assert "fire        disabled" in out


def test_budget(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["budget"]) == 0
    out = capsys.readouterr().out
    for tier in ("nano", "base", "max"):
        assert f"\n{tier:<6}    13" in out
    assert out.count("within budget") == 3
    assert main(["budget", "--tier", "nano", "--bands", "3"]) == 0
    assert "nano       3" in capsys.readouterr().out


def test_budget_reports_violation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("theoros_models.cli.FLIGHT_PARAMETER_LIMIT", 100_000)
    assert main(["budget"]) == 1
    assert capsys.readouterr().out.count("OVER BUDGET") == 2
